"""Per-utterance guardrails: cheap checks run synchronously on every response.

They target the failure modes that reach users in production TTS:
NaN/garbage output, clipping, dropouts (long silences inside speech),
truncated or run-away audio (speaking rate far outside the voice's range), and
front-end coverage problems (characters the G2P cannot read).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GuardrailConfig:
    clip_threshold: float = 0.99
    max_clip_ratio: float = 0.005
    silence_dbfs: float = -45.0
    max_internal_silence_s: float = 1.2
    # seconds of audio per phoneme symbol; Mana voice sits around 0.06-0.09 at length_scale=1
    min_sec_per_phone: float = 0.03
    max_sec_per_phone: float = 0.25
    max_unknown_char_ratio: float = 0.02


@dataclass
class GuardrailReport:
    passed: bool
    violations: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def _frame_db(audio: np.ndarray, sr: int, frame_s: float = 0.02) -> np.ndarray:
    n = max(int(sr * frame_s), 1)
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms)


def check_audio(audio: np.ndarray, sr: int, n_phones: int, unknown_char_ratio: float = 0.0,
                cfg: GuardrailConfig = GuardrailConfig()) -> GuardrailReport:
    v: list[str] = []
    m: dict[str, float] = {}
    if audio.size == 0 or not np.isfinite(audio).all():
        return GuardrailReport(False, ["non_finite_or_empty"], {})
    dur = len(audio) / sr
    m["duration_s"] = dur
    m["clip_ratio"] = float((np.abs(audio) >= cfg.clip_threshold).mean())
    if m["clip_ratio"] > cfg.max_clip_ratio:
        v.append("clipping")

    db = _frame_db(audio, sr)
    voiced = np.flatnonzero(db > cfg.silence_dbfs)
    if voiced.size == 0:
        return GuardrailReport(False, ["all_silence"], m)
    inner = db[voiced[0]: voiced[-1] + 1] <= cfg.silence_dbfs
    longest, run = 0, 0
    for s in inner:
        run = run + 1 if s else 0
        longest = max(longest, run)
    m["max_internal_silence_s"] = longest * 0.02
    if m["max_internal_silence_s"] > cfg.max_internal_silence_s:
        v.append("dropout")

    if n_phones:
        m["sec_per_phone"] = dur / n_phones
        if m["sec_per_phone"] < cfg.min_sec_per_phone:
            v.append("too_fast_or_truncated")
        elif m["sec_per_phone"] > cfg.max_sec_per_phone:
            v.append("too_slow_or_runaway")

    m["unknown_char_ratio"] = unknown_char_ratio
    if unknown_char_ratio > cfg.max_unknown_char_ratio:
        v.append("unreadable_characters")
    return GuardrailReport(not v, v, m)
