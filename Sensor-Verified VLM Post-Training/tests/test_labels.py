from pathlib import Path

import numpy as np
import pytest

from sensor_vlm import labels

DATA = Path(__file__).resolve().parents[1] / "data/hot3d"


def test_clean_runs_fills_interior_gaps_and_drops_blips():
    m = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0], dtype=bool)
    out = labels.clean_runs(m, fill_gap=2, min_run=2)
    assert out.tolist() == [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


def test_clean_runs_keeps_edge_gaps():
    m = np.array([0, 1, 1, 1, 0], dtype=bool)
    assert labels.clean_runs(m, fill_gap=5, min_run=1).tolist() == m.tolist()


def test_dilate():
    m = np.zeros(7, dtype=bool)
    m[3] = True
    assert labels.dilate(m, 2).tolist() == [0, 1, 1, 1, 1, 1, 0]


def test_quat_to_rot_matches_axis_angle():
    a = 0.7
    q = np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)])  # rotation about z
    R = labels.quat_to_rot(q)
    expected = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    assert np.allclose(R, expected)


@pytest.mark.skipif(not (DATA / "clips/clip-001849/meta.json").exists(), reason="clip not processed")
@pytest.mark.parametrize("method", ["contact", "wrist"])
def test_keyboard_clip_labels(method):
    clip = DATA / "clips/clip-001849"
    out = labels.label_clip(clip, labels.SurfaceIndex(DATA / "object_models"), method=method)
    names = labels.object_names(clip)
    kb = names.index("keyboard")
    assert set(np.flatnonzero(out["moving"].any(0))) == {kb}
    held = out["held"][:, :, kb]
    assert held.any() and not np.delete(out["held"], kb, axis=2).any()
    if method == "wrist":
        assert not (held.any(1) & ~labels.dilate(out["moving"][:, kb], labels.PAD)).any()
    else:
        assert not (held & ~out["touch"][:, :, kb]).any()
        for h in range(2):
            for start, end in labels.runs(held[:, h]):
                assert out["moving"][start:end, kb].sum() >= labels.MOTION_OVERLAP
