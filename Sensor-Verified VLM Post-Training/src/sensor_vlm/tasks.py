"""Turning labeled clips into VLM samples: 4-frame tiles with checkable answers.

A sample shows four frames of one clip, evenly spaced over 1.2-4.8 s, tiled
2x2 (1 top-left, 2 top-right, 3 bottom-left, 4 bottom-right). The model
answers in JSON:

    {"left": [0, 1, 1, 1], "right": [0, 0, 0, 0],
     "object": "dumbbell_5lb", "point": [212, 655]}

left/right   whether that hand is holding or moving an object in frames 1-4
object       the manipulated object, chosen from the clip's object list, or "none"
point        that object's location in the first frame where it is held and
             visible, in
             tile coordinates scaled to 0-1000 (Qwen3-VL's grounding
             convention); null when the object is "none"

Every field is derived from motion capture (`sensor_vlm.labels`), and the
sample carries the target object's mask at the point frame so the pointing
reward needs nothing but the sample itself.
"""

from __future__ import annotations

import base64
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sensor_vlm.hot3d import KEEP_EVERY, MASK_SIZE, load_masks

TILE = 448
SPACINGS = (12, 24, 36, 48)  # frames between tile frames at 30 fps: 0.4-1.6 s
CHANGE_WINDOWS = 3  # per clip, windows where some hand's state changes
STEADY_WINDOWS = 1  # per clip, windows without a change

PROMPT = (
    "The image tiles 4 frames from an egocentric video in time order: "
    "1 top-left, 2 top-right, 3 bottom-left, 4 bottom-right.\n"
    "Objects in the scene: {objects}.\n"
    "For each hand, say in which frames it is holding or moving an object (1 = yes, 0 = no). "
    "Name the object being manipulated (one of the list, or \"none\"), and point to it in the "
    "first frame where it is held and visible, with x and y in 0-1000 image coordinates.\n"
    "Answer with JSON only. Each hand gets 4 entries, one per frame (1-4), each 0 or 1:\n"
    '{{"left": [0/1, 0/1, 0/1, 0/1], "right": [0/1, 0/1, 0/1, 0/1], "object": "<name>", "point": [x, y] or null}}'
)


def window_frames(n_frames: int) -> list[list[int]]:
    """All 4-frame windows on the stored-frame grid."""
    out = []
    for s in SPACINGS:
        for start in range(0, n_frames - 3 * s, KEEP_EVERY):
            out.append([start + k * s for k in range(4)])
    return out


def answer_for(frames: list[int], held: np.ndarray, names: list[str]) -> tuple[dict, int | None, list[int]]:
    """Target answer (point still unset), the object column, and the tile slots where it is held."""
    h = held[frames]  # (4, 2, O)
    left = h[:, 0].any(-1).astype(int).tolist()
    right = h[:, 1].any(-1).astype(int).tolist()
    counts = h.sum(axis=(0, 1))
    if counts.max() == 0:
        return {"left": left, "right": right, "object": "none", "point": None}, None, []
    o = int(counts.argmax())
    slots = np.flatnonzero(h[:, :, o].any(-1)).tolist()
    return {"left": left, "right": right, "object": names[o], "point": None}, o, slots


def mask_point(mask: np.ndarray, quadrant: int) -> list[int] | None:
    """A point on the mask (its pixel nearest the centroid), in 0-1000 tile coordinates."""
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return None
    cy, cx = rows.mean(), cols.mean()
    i = int(np.argmin((rows - cy) ** 2 + (cols - cx) ** 2))
    qx, qy = quadrant % 2, quadrant // 2
    x = (qx + (cols[i] + 0.5) / MASK_SIZE) / 2
    y = (qy + (rows[i] + 0.5) / MASK_SIZE) / 2
    return [round(1000 * x), round(1000 * y)]


def render_tile(clip_dir: Path, frames: list[int], size: int = TILE) -> Image.Image:
    half = size // 2
    tile = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(tile)
    for k, t in enumerate(frames):
        img = Image.open(clip_dir / f"frames/{t:03d}.jpg").convert("RGB").resize((half, half), Image.Resampling.LANCZOS)
        x, y = (k % 2) * half, (k // 2) * half
        tile.paste(img, (x, y))
        draw.rectangle([x, y, x + 17, y + 17], fill="black")
        draw.text((x + 5, y + 3), str(k + 1), fill="white")
    return tile


def encode_mask(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask).tobytes()).decode()


def decode_mask(s: str) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(base64.b64decode(s), dtype=np.uint8))
    return bits[: MASK_SIZE * MASK_SIZE].reshape(MASK_SIZE, MASK_SIZE).astype(bool)


def build_clip_samples(clip_dir: Path, held: np.ndarray, names: list[str], rng: random.Random) -> list[dict]:
    """Choose windows for one clip and return samples (without writing tiles)."""
    meta = json.loads((clip_dir / "meta.json").read_text())
    kept, masks = load_masks(clip_dir)
    kept_index = {int(t): i for i, t in enumerate(kept)}

    change, steady = [], []
    for frames in window_frames(meta["n_frames"]):
        ans, o, slots = answer_for(frames, held, names)
        varies = len(set(ans["left"])) > 1 or len(set(ans["right"])) > 1
        (change if varies else steady).append((frames, ans, o, slots))
    chosen = rng.sample(change, min(CHANGE_WINDOWS, len(change))) + rng.sample(steady, min(STEADY_WINDOWS, len(steady)))

    samples = []
    for frames, ans, o, slots in chosen:
        point_frame, point_mask = None, None
        visible = [k for k in slots if masks[kept_index[frames[k]], o].any()]
        if visible:
            point_frame = visible[0]
            m = masks[kept_index[frames[point_frame]], o]
            ans["point"] = mask_point(m, point_frame)
            point_mask = encode_mask(m)
        objects = names[:]
        rng.shuffle(objects)
        samples.append({
            "id": f"{meta['clip']}_{frames[0]:03d}_{frames[1] - frames[0]:02d}",
            "clip": meta["clip"],
            "participant": meta["participant_id"],
            "frames": frames,
            "objects": objects,
            "prompt": PROMPT.format(objects=", ".join(objects)),
            "answer": ans,
            "point_frame": point_frame,
            "point_mask": point_mask,
        })
    return samples
