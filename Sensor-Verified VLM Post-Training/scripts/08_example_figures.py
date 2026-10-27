"""Illustrative figures for the README: one task tile, and contact labels on a clip.

    docs/figures/example_tile.jpg     a val tile with its target point marked
    docs/figures/labels_example.jpg   five frames of a clip with per-hand contact labels and a timeline

    python scripts/08_example_figures.py --tile-id clip-001868_015_12 --clip clip-001868
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sensor_vlm import labels

INK, LEFT, RIGHT = (11, 11, 11), (42, 120, 214), (235, 104, 52)  # hands in categorical slots 1 and 2


def find_sample(tasks: Path, sample_id: str) -> dict:
    for split in ("train", "val", "test"):
        for line in open(tasks / f"{split}.jsonl"):
            s = json.loads(line)
            if s["id"] == sample_id:
                return s
    raise KeyError(sample_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, default=Path("data/tasks"))
    ap.add_argument("--data", type=Path, default=Path("data/hot3d"))
    ap.add_argument("--tile-id", default="clip-001868_015_12")
    ap.add_argument("--clip", default="clip-001873")
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    s = find_sample(args.tasks, args.tile_id)
    tile = Image.open(args.tasks / s["image"]).convert("RGB")
    d = ImageDraw.Draw(tile)
    if s["answer"]["point"]:
        x, y = (v * tile.width / 1000 for v in s["answer"]["point"])
        d.ellipse([x - 8, y - 8, x + 8, y + 8], outline=(255, 255, 255), width=4)
        d.ellipse([x - 8, y - 8, x + 8, y + 8], outline=RIGHT, width=2)
    tile.save(args.out / "example_tile.jpg", quality=92)
    print(s["id"], json.dumps(s["answer"]))

    clip = args.data / "clips" / args.clip
    out = labels.label_clip(clip, labels.SurfaceIndex(args.data / "object_models"))
    names = labels.object_names(clip)
    frames, w = [0, 36, 72, 108, 147], 220
    canvas = Image.new("RGB", (len(frames) * w, w + 74), (252, 252, 251))
    d = ImageDraw.Draw(canvas)
    for i, t in enumerate(frames):
        canvas.paste(Image.open(clip / f"frames/{t:03d}.jpg").resize((w, w)), (i * w, 0))
        d.text((i * w + 4, w + 4), f"t = {t / 30:.1f} s", fill=INK)
        for h, (side, color) in enumerate((("left", LEFT), ("right", RIGHT))):
            held = [names[o] for o in np.flatnonzero(out["held"][t, h])]
            y = w + 18 + 14 * h
            d.rectangle([i * w + 4, y + 2, i * w + 12, y + 10], fill=color)  # swatch carries identity; text stays ink
            d.text((i * w + 16, y), f"{side}: {', '.join(held) or '-'}", fill=INK)
    for h, (y, color) in enumerate(((w + 54, LEFT), (w + 64, RIGHT))):
        for t in range(out["held"].shape[0]):
            if out["held"][t, h].any():
                x0 = t * canvas.width / out["held"].shape[0]
                d.rectangle([x0, y, x0 + canvas.width / out["held"].shape[0], y + 6], fill=color)
    canvas.save(args.out / "labels_example.jpg", quality=92)


if __name__ == "__main__":
    main()
