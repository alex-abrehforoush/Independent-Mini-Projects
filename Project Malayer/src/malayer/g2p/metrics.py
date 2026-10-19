"""G2P evaluation metrics.

* PER       character-level edit distance between surface phoneme strings
            (ezafe markers and spaces removed) / reference length.
* Ezafe     word-level precision / recall / F1 of the ezafe marker, computed on
            sentences whose hypothesis and reference have the same word count.
* Homograph accuracy: the homograph word's bare pronunciation matches the
            sentence reference (or a listed alternative pronunciation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from malayer.phonemes import EZAFE, split_ezafe, strip_markers


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def surface(r2: str) -> str:
    return strip_markers(r2).replace(" ", "")


@dataclass
class G2PScores:
    edits: int = 0
    ref_len: int = 0
    sent_correct: int = 0
    n_sent: int = 0
    ez_tp: int = 0
    ez_fp: int = 0
    ez_fn: int = 0
    ez_sents: int = 0
    hom_correct: int = 0
    hom_total: int = 0
    hom_errors: list = field(default_factory=list)

    def add(self, hyp: str, ref: str, graphemes: str | None = None,
            hom: str | None = None, hom_ref: str | None = None) -> None:
        h, r = surface(hyp), surface(ref)
        self.edits += edit_distance(h, r)
        self.ref_len += len(r)
        self.n_sent += 1
        self.sent_correct += h == r
        hw, rw = hyp.split(), ref.split()
        if len(hw) == len(rw):
            self.ez_sents += 1
            for a, b in zip(hw, rw):
                pa, pb = a.endswith(EZAFE), b.endswith(EZAFE)
                self.ez_tp += pa and pb
                self.ez_fp += pa and not pb
                self.ez_fn += pb and not pa
        if hom and graphemes is not None:
            gold, _ = homograph_pronunciation(graphemes, ref, hom)
            accept = {gold} | set((hom_ref or "").split())
            accept.discard(None)
            if not accept:
                return
            self.hom_total += 1
            pred, exact = homograph_pronunciation(graphemes, hyp, hom)
            ok = pred is not None and (
                pred in accept or (not exact and any(pred.startswith(a) for a in accept)))
            self.hom_correct += ok
            if not ok:
                self.hom_errors.append((graphemes, hom, hom_ref, pred))

    def summary(self) -> dict:
        p = self.ez_tp / max(self.ez_tp + self.ez_fp, 1)
        r = self.ez_tp / max(self.ez_tp + self.ez_fn, 1)
        return {
            "PER": round(100 * self.edits / max(self.ref_len, 1), 2),
            "sentence_acc": round(100 * self.sent_correct / max(self.n_sent, 1), 2),
            "ezafe_P": round(100 * p, 2), "ezafe_R": round(100 * r, 2),
            "ezafe_F1": round(100 * 2 * p * r / max(p + r, 1e-9), 2),
            "ezafe_coverage": round(self.ez_sents / max(self.n_sent, 1), 3),
            "homograph_acc": round(100 * self.hom_correct / self.hom_total, 2) if self.hom_total else None,
            "n": self.n_sent,
        }


def homograph_pronunciation(graphemes: str, hyp: str, hom: str) -> tuple[str | None, bool]:
    """Bare pronunciation a phoneme string gives the homograph word.

    Grapheme and hypothesis words are matched by index when counts agree;
    otherwise None (counted as an error — a system that changes word
    segmentation around the homograph gets no credit). The flag says whether the
    grapheme word *is* the homograph (vs. an inflected form containing it).
    """
    punct = ".,!?؟،؛:«»()\"'-"
    gw = [w for w in graphemes.replace("\u200c", "").split() if w.strip(punct)]
    hw = hyp.split()
    target = hom.replace("\u200c", "")
    if len(gw) != len(hw):
        return None, False
    for g, h in zip(gw, hw):
        if g.strip(punct) == target:
            return split_ezafe(h)[0], True
    for g, h in zip(gw, hw):  # inflected occurrence (prefix/suffix attached)
        if target in g:
            return split_ezafe(h)[0], False
    return None, False
