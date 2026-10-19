import random

import numpy as np

from malayer.serve.experiment import MSPRT, assign_arm
from malayer.serve.guardrails import check_audio
from malayer.serve.monitor import psi

SR = 22050


def _speech_like(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2  # syllable-rate modulation, never silent
    return (0.3 * env * np.sin(2 * np.pi * 180 * t) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def test_guardrails_pass_on_plausible_audio():
    r = check_audio(_speech_like(2.0), SR, n_phones=28)
    assert r.passed, r.violations


def test_guardrails_catch_dropout_clipping_and_truncation():
    a = np.concatenate([_speech_like(1.0), np.zeros(int(2.0 * SR), np.float32), _speech_like(1.0, 1)])
    assert "dropout" in check_audio(a, SR, n_phones=50).violations
    assert "clipping" in check_audio(np.clip(_speech_like(2.0) * 10, -1, 1), SR, n_phones=28).violations
    assert "too_fast_or_truncated" in check_audio(_speech_like(0.3), SR, n_phones=40).violations
    assert check_audio(np.array([np.nan], np.float32), SR, 1).violations == ["non_finite_or_empty"]


def test_assign_arm_is_deterministic_and_balanced():
    arms = {"control": 0.5, "treatment": 0.5}
    assert assign_arm("user-42", arms) == assign_arm("user-42", arms)
    share = np.mean([assign_arm(f"u{i}", arms) == "treatment" for i in range(4000)])
    assert 0.47 < share < 0.53


def test_msprt_controls_false_positives_under_continuous_peeking():
    rng = random.Random(0)
    false_pos = 0
    for _ in range(200):  # A/A tests, p-value checked after every observation
        t = MSPRT(tau=0.1, min_n=20)
        for _ in range(400):
            t.add("a", float(rng.random() < 0.9))
            t.add("b", float(rng.random() < 0.9))
        false_pos += t.p_value < 0.05
    assert false_pos / 200 <= 0.06


def test_msprt_detects_real_effect():
    rng = random.Random(1)
    t = MSPRT(tau=0.1, min_n=20)
    for _ in range(1500):
        t.add("a", float(rng.random() < 0.80))
        t.add("b", float(rng.random() < 0.88))
    assert t.p_value < 0.05


def test_psi_flags_shift_only():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 500)) < 0.1
    assert psi(ref, rng.normal(1.0, 1, 500)) > 0.2


def test_psi_detects_shift_in_zero_inflated_feature():
    ref = np.zeros(2000)  # e.g. unknown-character ratio on clean Persian text
    cur = np.concatenate([np.zeros(400), np.full(100, 0.3)])  # 20% of traffic now has Latin script
    assert psi(ref, cur) > 0.2
    assert psi(ref, np.zeros(500)) < 0.01
