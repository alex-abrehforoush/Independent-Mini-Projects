"""Online experimentation: deterministic arm assignment + always-valid sequential tests.

Arms are assigned by hashing the session id, so a user keeps the same variant
across requests and assignment needs no state.

`MSPRT` implements the mixture sequential probability ratio test for a
difference in means (Johari, Pekelis & Walsh, "Always Valid Inference", 2017).
Its p-value stays valid no matter how often it is checked, so a dashboard can
be watched continuously and the experiment stopped as soon as p < alpha —
unlike a fixed-horizon t-test, where peeking inflates false positives.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field


def assign_arm(session_id: str, arms: dict[str, float], salt: str = "exp1") -> str:
    """Deterministic weighted assignment from a session id."""
    h = int(hashlib.sha256(f"{salt}:{session_id}".encode()).hexdigest()[:12], 16) / 16 ** 12
    acc = 0.0
    total = sum(arms.values())
    for name, w in arms.items():
        acc += w / total
        if h < acc:
            return name
    return name


@dataclass
class _Stream:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, x: float) -> None:  # Welford
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def var(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0


@dataclass
class MSPRT:
    """mSPRT on a difference of means with a N(0, tau^2) mixing prior.

    Two-arm by default (arm "b" minus arm "a"). With `baseline` set it is a
    one-sample test of arm "b"'s mean against that constant (e.g. a listener
    preference rate against 0.5).
    """

    tau: float = 0.05
    min_n: int = 30
    baseline: float | None = None
    a: _Stream = field(default_factory=_Stream)
    b: _Stream = field(default_factory=_Stream)
    p_value: float = 1.0

    def add(self, arm: str, x: float) -> None:
        (self.a if arm == "a" else self.b).add(x)
        ready = self.b.n >= self.min_n and (self.baseline is not None or self.a.n >= self.min_n)
        if ready:
            self.p_value = min(self.p_value, 1.0 / max(self.likelihood_ratio(), 1e-300))

    def likelihood_ratio(self) -> float:
        if self.baseline is None:
            V = self.a.var / self.a.n + self.b.var / self.b.n
            diff = self.b.mean - self.a.mean
        else:
            V = self.b.var / self.b.n
            diff = self.b.mean - self.baseline
        if V <= 0:
            return 1.0
        t2 = self.tau ** 2
        return math.sqrt(V / (V + t2)) * math.exp(t2 * diff ** 2 / (2 * V * (V + t2)))

    def summary(self) -> dict:
        ref = self.baseline if self.baseline is not None else self.a.mean
        return {"n_a": self.a.n, "n_b": self.b.n, "mean_a": round(ref, 4), "mean_b": round(self.b.mean, 4),
                "diff": round(self.b.mean - ref, 4), "always_valid_p": round(self.p_value, 5)}
