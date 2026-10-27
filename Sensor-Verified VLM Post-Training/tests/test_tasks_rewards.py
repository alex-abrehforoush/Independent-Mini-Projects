import json
import random
from pathlib import Path

import numpy as np
import pytest

from sensor_vlm import labels, rewards, tasks
from sensor_vlm.hot3d import KEEP_EVERY, MASK_SIZE

DATA = Path(__file__).resolve().parents[1] / "data/hot3d"


def test_windows_stay_on_stored_frames():
    ws = tasks.window_frames(150)
    assert all(len(w) == 4 and w[-1] < 150 and all(t % KEEP_EVERY == 0 for t in w) for w in ws)
    assert {w[1] - w[0] for w in ws} == set(tasks.SPACINGS)


def test_answer_for_picks_most_held_object():
    held = np.zeros((150, 2, 3), dtype=bool)
    held[30:90, 0, 2] = True  # left holds object 2
    held[60:70, 1, 1] = True  # right briefly holds object 1
    ans, o, slots = tasks.answer_for([0, 36, 72, 108], held, ["a", "b", "c"])
    assert ans["left"] == [0, 1, 1, 0] and ans["right"] == [0, 0, 0, 0]
    assert ans["object"] == "c" and o == 2 and slots == [1, 2]


def test_answer_for_none():
    ans, o, slots = tasks.answer_for([0, 12, 24, 36], np.zeros((150, 2, 2), dtype=bool), ["a", "b"])
    assert ans["object"] == "none" and o is None and slots == []


@pytest.mark.parametrize("quadrant", range(4))
def test_mask_point_roundtrip(quadrant):
    m = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
    m[40:60, 90:100] = True
    p = tasks.mask_point(m, quadrant)
    enc = tasks.encode_mask(m)
    assert np.array_equal(tasks.decode_mask(enc), m)
    assert rewards.point_on_mask(p, quadrant, enc)
    assert not rewards.point_on_mask(p, (quadrant + 1) % 4, enc)


def _sample(obj="mug", point_mask=True):
    m = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
    m[10:30, 10:30] = True
    return {
        "answer": {"left": [0, 1, 1, 0], "right": [0, 0, 0, 1], "object": obj,
                   "point": tasks.mask_point(m, 1) if obj != "none" else None},
        "point_frame": 1 if obj != "none" else None,
        "point_mask": tasks.encode_mask(m) if (obj != "none" and point_mask) else None,
    }


def test_score_perfect_and_fenced():
    s = _sample()
    r = rewards.score("```json\n" + json.dumps(s["answer"]) + "\n```", s)
    assert r["total"] == pytest.approx(1.0)


def test_score_partial_and_invalid():
    s = _sample()
    wrong = {"left": [0, 1, 1, 1], "right": [0, 0, 0, 1], "object": "bowl", "point": [10, 990]}
    r = rewards.score(json.dumps(wrong), s)
    assert r["hands"] == pytest.approx(7 / 8) and r["object"] == 0 and r["point"] == 0
    assert rewards.score("the left hand holds a mug", s)["total"] == 0
    bad_bits = dict(wrong, left=[0, 2, 1, 1])
    assert rewards.score(json.dumps(bad_bits), s)["format"] == 0


def test_score_none_and_unscorable_point():
    s = _sample("none")
    assert rewards.score(json.dumps(s["answer"]), s)["total"] == pytest.approx(1.0)
    s = _sample(point_mask=False)
    r = rewards.score(json.dumps(dict(s["answer"], point=None)), s)
    assert np.isnan(r["point"]) and r["total"] == pytest.approx(1.0)


@pytest.mark.skipif(not (DATA / "clips/clip-001868/meta.json").exists(), reason="clip not processed")
def test_clip_samples_score_perfectly_against_themselves():
    clip = DATA / "clips/clip-001868"
    out = labels.label_clip(clip, labels.SurfaceIndex(DATA / "object_models"))
    samples = tasks.build_clip_samples(clip, out["held"], labels.object_names(clip), random.Random(0))
    assert samples
    for s in samples:
        assert rewards.score(json.dumps(s["answer"]), s)["total"] == pytest.approx(1.0)
        assert all(name in s["prompt"] for name in s["objects"])
    tile = tasks.render_tile(clip, samples[0]["frames"])
    assert tile.size == (tasks.TILE, tasks.TILE)


def test_fields_scored_independently():
    s = _sample()
    d = dict(s["answer"], left=[3, 3, 3, 3])
    r = rewards.score(json.dumps(d), s)
    assert r["format"] == 0 and r["object"] == 1 and r["point"] == 1 and r["hands"] == pytest.approx(4 / 8)
