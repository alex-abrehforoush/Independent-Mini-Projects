"""Input / model drift monitoring for the front-end.

Each request yields a few cheap features. A rolling window of recent values is
compared against a reference distribution (from the dev set) with the
Population Stability Index; PSI > 0.2 is the conventional "significant shift"
alert. Typical causes this catches for Persian TTS: a client starts sending
Finglish or Arabic-script text (unknown chars up), informal/colloquial text
(model confidence down), or very long inputs (length shift).
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURES = ("n_chars", "unknown_char_ratio", "ezafe_rate", "mean_confidence", "min_confidence")


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with quantile bins from the reference.

    Outer bins are open-ended and values equal to an edge fall in the lower bin,
    so zero-inflated features (e.g. unknown-character ratio, almost always 0 in
    the reference) still get a "0" bin and a "> 0" bin.
    """
    if len(ref) < bins or len(cur) < 10:
        return 0.0
    inner = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))[1:-1])
    if inner.size == 0:
        inner = np.array([np.median(ref)])
    k = inner.size + 1
    r = np.bincount(np.searchsorted(inner, ref, side="left"), minlength=k) / len(ref)
    c = np.bincount(np.searchsorted(inner, cur, side="left"), minlength=k) / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(((c - r) * np.log(c / r)).sum())


@dataclass
class DriftMonitor:
    reference: dict[str, np.ndarray]
    window: int = 500
    threshold: float = 0.2

    def __post_init__(self):
        self.recent = {f: collections.deque(maxlen=self.window) for f in FEATURES}

    @classmethod
    def from_file(cls, path: str | Path, **kw) -> "DriftMonitor":
        d = json.loads(Path(path).read_text())
        return cls({k: np.asarray(v) for k, v in d.items()}, **kw)

    def observe(self, feats: dict[str, float]) -> None:
        for f in FEATURES:
            if f in feats:
                self.recent[f].append(feats[f])

    def report(self) -> dict[str, dict]:
        out = {}
        for f in FEATURES:
            cur = np.asarray(self.recent[f])
            val = psi(self.reference[f], cur) if f in self.reference else 0.0
            out[f] = {"psi": round(val, 4), "alert": val > self.threshold, "n": len(cur)}
        return out
