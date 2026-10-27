"""Reading HOT3D-Clips and converting each clip to a compact processed form.

A HOT3D-Clips tar holds 150 frames (30 fps, 5 s) of one recording, with
images and JSON annotations from a marker-based motion-capture system:
object 6-DoF poses with 2D boxes and RLE masks, UmeTrack / MANO hand poses,
and camera poses. Aria clips have an RGB stream (214-1, 1408x1408) plus two
monochrome SLAM cameras; Quest 3 clips have two monochrome cameras
(1201-1/2, 1280x1024). We use one stream per device (STREAMS).

We keep only what the VLM tasks and the label derivation need:

    frames/<frame>.jpg  upright frames at 10 fps, 256x256 (RGB; Quest 3 gray)
    kin.npz             30 fps kinematics: object, wrist and camera poses, hand
                        joint angles, visibilities, 2D boxes (upright pixels)
    masks.npz           object masks at the kept frames, 128x128, bit-packed
    meta.json           ids, object names, calibration, hand shape; written
                        last, so its presence marks the clip as done

Both devices' cameras are mounted sideways: images are rotated 90 degrees
clockwise to make them upright, then padded to a square (a no-op for Aria),
and every 2D annotation goes through the same transform.
Poses are 7-vectors [qw, qx, qy, qz, tx, ty, tz] (world-from-local, meters).
"""

from __future__ import annotations

import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

STREAMS = {"Aria": "214-1", "Quest3": "1201-1"}
SRC_SIZE = 1408  # Aria RGB
FPS = 30
KEEP_EVERY = 3  # 30 fps -> 10 fps for stored frames and masks
FRAME_SIZE = 256
MASK_SIZE = 128  # divides both padded sizes: 1408 = 128 * 11, 1280 = 128 * 10
HANDS = ("left", "right")


def pose7(d: dict) -> np.ndarray:
    return np.array(d["quaternion_wxyz"] + d["translation_xyz"], dtype=np.float64)


def upright_square(w: int, h: int) -> tuple[int, int, int]:
    """Side of the upright square and the padding offsets (x, y) for a raw w x h image.

    Rotated clockwise, the image is h wide and w tall; it is centered in a square.
    """
    side = max(w, h)
    return side, (side - h) // 2, (side - w) // 2


def rotate_point(x: float, y: float, w: int = SRC_SIZE, h: int = SRC_SIZE) -> tuple[float, float]:
    """Continuous (x, y) in raw image coords -> upright square coords."""
    _, ox, oy = upright_square(w, h)
    return h - y + ox, x + oy


def rotate_box(box, w: int = SRC_SIZE, h: int = SRC_SIZE) -> list[float]:
    """[x1, y1, x2, y2] in raw image coords -> upright square coords."""
    x1, y1, x2, y2 = box
    _, ox, oy = upright_square(w, h)
    return [h - y2 + ox, x1 + oy, h - y1 + ox, x2 + oy]


def decode_rle(d: dict) -> np.ndarray:
    """HOT3D binary-mask RLE: [start, length, ...], starts 1-based, row-major."""
    rle = np.asarray(d["rle"], dtype=np.int64)
    flat = np.zeros(d["height"] * d["width"], dtype=bool)
    for start, length in zip(rle[0::2] - 1, rle[1::2]):
        flat[start:start + length] = True
    return flat.reshape(d["height"], d["width"])


def upright_pooled_mask(mask: np.ndarray, out: int = MASK_SIZE) -> np.ndarray:
    """Rotate to upright, pad to square and max-pool to out x out (set if any source pixel is)."""
    h, w = mask.shape
    side, ox, oy = upright_square(w, h)
    m = np.zeros((side, side), dtype=bool)
    m[oy:oy + w, ox:ox + h] = np.rot90(mask, k=-1)
    f = side // out
    return m[: out * f, : out * f].reshape(out, f, out, f).any(axis=(1, 3))


def upright_frame(jpeg: bytes, size: int = FRAME_SIZE) -> Image.Image:
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    side, ox, oy = upright_square(img.width, img.height)
    square = Image.new("RGB", (side, side))
    square.paste(img.transpose(Image.Transpose.ROTATE_270), (ox, oy))
    return square.resize((size, size), Image.Resampling.LANCZOS)


def read_clip_tar(path: Path) -> tuple[dict[int, dict], dict]:
    """Per-frame annotations and kept-frame JPEG bytes (key jpeg_<stream>), plus the hand shape."""
    frames: dict[int, dict] = defaultdict(dict)
    hand_shape = {}
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            if m.name == "__hand_shapes.json__":
                hand_shape = json.load(tf.extractfile(m))
                continue
            stem, _, ext = m.name.partition(".")
            if not stem.isdigit():
                continue
            fid = int(stem)
            if ext.startswith("image_") and ext[len("image_"):-len(".jpg")] in STREAMS.values():
                if fid % KEEP_EVERY == 0:
                    frames[fid][f"jpeg_{ext[len('image_'):-len('.jpg')]}"] = tf.extractfile(m).read()
            elif ext.endswith(".json"):
                frames[fid][ext[: -len(".json")]] = json.load(tf.extractfile(m))
    return dict(frames), hand_shape


def process_clip(tar_path: Path, out_dir: Path) -> dict:
    """Convert one HOT3D-Clips tar into the processed layout; returns meta."""
    frames, hand_shape = read_clip_tar(tar_path)
    n = max(frames) + 1
    assert sorted(frames) == list(range(n)), f"{tar_path.name}: non-contiguous frames"
    info0 = frames[0]["info"]
    stream = STREAMS[info0["device"]]
    calib = frames[0]["cameras"][stream]["calibration"]
    img_w, img_h = calib["image_width"], calib["image_height"]

    obj_ids = sorted({int(k) for f in frames.values() for k in f.get("objects", {})})
    names = {}
    col = {oid: i for i, oid in enumerate(obj_ids)}
    n_obj = len(obj_ids)

    nan = np.nan
    t_ns = np.zeros(n, dtype=np.int64)
    obj_pose = np.full((n, n_obj, 7), nan)
    obj_vis = np.full((n, n_obj), nan)
    obj_box = np.full((n, n_obj, 4), nan)
    wrist_pose = np.full((n, 2, 7), nan)
    joint_angles = np.full((n, 2, 22), nan)
    mano_thetas = np.full((n, 2, 15), nan)
    mano_wrist = np.full((n, 2, 6), nan)
    hand_vis = np.full((n, 2), nan)
    hand_box = np.full((n, 2, 4), nan)
    cam_pose = np.full((n, 7), nan)

    kept = list(range(0, n, KEEP_EVERY))
    masks = np.zeros((len(kept), n_obj, MASK_SIZE, MASK_SIZE), dtype=bool)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    for t in range(n):
        f = frames[t]
        t_ns[t] = f["info"]["ref_timestamp_ns"]
        cam_pose[t] = pose7(f["cameras"][stream]["T_world_from_camera"])
        for k, lst in f.get("objects", {}).items():
            o, j = lst[0], col[int(k)]
            names[int(k)] = o["object_name"]
            obj_pose[t, j] = pose7(o["T_world_from_object"])
            obj_vis[t, j] = o["visibilities_modeled"].get(stream, 0.0)
            if stream in o["boxes_amodal"]:
                obj_box[t, j] = rotate_box(o["boxes_amodal"][stream], img_w, img_h)
            if t % KEEP_EVERY == 0 and stream in o.get("masks_modal", {}):
                masks[t // KEEP_EVERY, j] = upright_pooled_mask(decode_rle(o["masks_modal"][stream]))
        for h, side in enumerate(HANDS):
            d = f.get("hands", {}).get(side)
            if d is None:
                continue
            wrist_pose[t, h] = pose7(d["umetrack_pose"]["T_world_from_wrist"])
            joint_angles[t, h] = d["umetrack_pose"]["joint_angles"]
            mano_thetas[t, h] = d["mano_pose"]["thetas"]
            mano_wrist[t, h] = d["mano_pose"]["wrist_xform"]
            hand_vis[t, h] = d["visibilities_modeled"].get(stream, 0.0)
            if stream in d["boxes_amodal"]:
                hand_box[t, h] = rotate_box(d["boxes_amodal"][stream], img_w, img_h)
        if t % KEEP_EVERY == 0:
            upright_frame(f[f"jpeg_{stream}"]).save(frame_dir / f"{t:03d}.jpg", quality=90)

    np.savez_compressed(
        out_dir / "kin.npz", t_ns=t_ns, obj_ids=np.array(obj_ids), obj_pose=obj_pose, obj_vis=obj_vis,
        obj_box=obj_box, wrist_pose=wrist_pose, joint_angles=joint_angles, mano_thetas=mano_thetas,
        mano_wrist=mano_wrist, hand_vis=hand_vis, hand_box=hand_box, cam_pose=cam_pose,
    )
    np.savez_compressed(out_dir / "masks.npz", frames=np.array(kept), masks=np.packbits(masks, axis=-1))

    meta = {
        "clip": tar_path.stem,
        "sequence_id": info0["sequence_id"],
        "participant_id": info0["participant_id"],
        "device": info0["device"],
        "stream": stream,
        "n_frames": n,
        "fps": FPS,
        "kept_frames": kept,
        "frame_size": FRAME_SIZE,
        "mask_size": MASK_SIZE,
        "source_size": [img_w, img_h],
        "upright_size": upright_square(img_w, img_h)[0],
        "objects": {str(k): names[k] for k in obj_ids},
        "camera_calibration": calib,
        "hand_shape": hand_shape,
    }
    tmp = out_dir / "meta.json.tmp"
    tmp.write_text(json.dumps(meta))
    tmp.rename(out_dir / "meta.json")
    return meta


def load_masks(clip_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Kept frame indices and bool masks of shape (frames, objects, MASK_SIZE, MASK_SIZE)."""
    z = np.load(clip_dir / "masks.npz")
    return z["frames"], np.unpackbits(z["masks"], axis=-1)[..., :MASK_SIZE].astype(bool)
