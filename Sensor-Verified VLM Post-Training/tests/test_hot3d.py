import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sensor_vlm import hot3d

SAMPLE_TAR = Path(__file__).resolve().parents[1] / "data/raw/clip-001849.tar"
QUEST3_TAR = Path(__file__).resolve().parents[1] / "data/raw/clip-000838-quest3.tar"


def encode_rle(mask: np.ndarray) -> dict:
    """Reference encoder matching HOT3D's clip_util.encode_binary_mask_rle."""
    pixels = np.concatenate([[0], mask.flatten().astype(np.uint8), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return {"height": mask.shape[0], "width": mask.shape[1], "rle": runs.tolist()}


def test_rle_roundtrip_is_row_major():
    rng = np.random.default_rng(0)
    mask = rng.random((11, 7)) > 0.6
    mask[0, :] = True  # a run touching the first pixel
    assert np.array_equal(hot3d.decode_rle(encode_rle(mask)), mask)


@pytest.mark.parametrize("w,h", [(22, 22), (20, 16)])
def test_image_mask_box_and_point_rotate_together(w, h):
    raw = np.zeros((h, w), dtype=bool)
    raw[2:5, 13:18] = True  # rows 2-4, cols 13-17: a blob near the top-right
    box = [13, 2, 18, 5]  # x1, y1, x2, y2
    side = max(w, h)

    img = Image.fromarray(raw.astype(np.uint8) * 255)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    up_img = np.asarray(hot3d.upright_frame(buf.getvalue(), size=side).convert("L")) > 127
    up_mask = hot3d.upright_pooled_mask(raw, out=side)
    assert np.array_equal(up_img, up_mask)

    x1, y1, x2, y2 = hot3d.rotate_box(box, w=w, h=h)
    rows, cols = np.nonzero(up_mask)
    assert (cols.min(), rows.min(), cols.max() + 1, rows.max() + 1) == (x1, y1, x2, y2)

    px, py = hot3d.rotate_point(15.5, 3.5, w=w, h=h)  # blob center
    assert up_mask[int(py), int(px)]


def test_pooled_mask_keeps_small_objects():
    m = np.zeros((1408, 1408), dtype=bool)
    m[700, 3] = True
    pooled = hot3d.upright_pooled_mask(m)
    assert pooled.shape == (128, 128) and pooled.sum() == 1


@pytest.mark.skipif(not SAMPLE_TAR.exists(), reason="sample clip not downloaded")
def test_process_sample_clip(tmp_path):
    meta = hot3d.process_clip(SAMPLE_TAR, tmp_path)
    kin = np.load(tmp_path / "kin.npz")
    kept, masks = hot3d.load_masks(tmp_path)

    assert meta["n_frames"] == 150 and len(kept) == 50
    assert len(list((tmp_path / "frames").glob("*.jpg"))) == 50
    assert masks.shape == (50, len(meta["objects"]), 128, 128)
    assert kin["obj_pose"].shape == (150, len(meta["objects"]), 7)

    # the keyboard is the only object that moves in this clip
    speed = np.linalg.norm(np.diff(kin["obj_pose"][:, :, 4:], axis=0), axis=-1) * hot3d.FPS
    moving = {meta["objects"][str(o)] for o, s in zip(kin["obj_ids"], speed.max(axis=0)) if s > 0.1}
    assert moving == {"keyboard"}

    # every visible object's mask lies inside its (upright) amodal box
    t, j = 0, list(meta["objects"].values()).index("keyboard")
    x1, y1, x2, y2 = kin["obj_box"][t, j] * 128 / 1408
    rows, cols = np.nonzero(masks[0, j])
    assert cols.min() >= x1 - 1 and cols.max() <= x2 + 1 and rows.min() >= y1 - 1 and rows.max() <= y2 + 1


@pytest.mark.skipif(not QUEST3_TAR.exists(), reason="sample Quest 3 clip not downloaded")
def test_process_quest3_clip(tmp_path):
    meta = hot3d.process_clip(QUEST3_TAR, tmp_path)
    kin = np.load(tmp_path / "kin.npz")
    kept, masks = hot3d.load_masks(tmp_path)
    assert meta["device"] == "Quest3" and meta["stream"] == "1201-1" and meta["upright_size"] == 1280
    from PIL import Image as _Image
    assert _Image.open(tmp_path / "frames/000.jpg").size == (256, 256)
    side = meta["upright_size"]
    for i, t in enumerate(kept[:10]):
        for j in range(masks.shape[1]):
            if not masks[i, j].any():
                continue
            x1, y1, x2, y2 = kin["obj_box"][t, j] * 128 / side
            rows, cols = np.nonzero(masks[i, j])
            assert cols.min() >= x1 - 1 and cols.max() <= x2 + 1 and rows.min() >= y1 - 1 and rows.max() <= y2 + 1
