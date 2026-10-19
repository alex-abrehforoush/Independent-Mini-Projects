"""Monotonic grapheme->phoneme alignment for building tagging labels.

Every normalized grapheme character is assigned exactly one *label*: the
phoneme string it emits (possibly empty), e.g. ک->"ke", ت->"t", ا->"a", ب->"b".
Word-final characters may additionally carry an ezafe flag. Concatenating the
labels (and realizing ezafe) reconstructs the sentence's R2 phonemes exactly.

Because outputs are tied 1:1 to input characters, a tagger trained on these
labels cannot insert, drop or repeat words — the failure mode that makes
seq2seq G2P risky in a production TTS front-end.

The aligner is a weighted DP over (grapheme index, phoneme index) whose
transitions come from a small hand-written table of letter realizations.
Sentences it cannot align are rejected, which doubles as a data-quality filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

from persian_tts.phonemes import join_ezafe, split_ezafe
from persian_tts.text.normalize import PUNCT, TANWIN_FATH, ZWNJ

EZ = "\x01"  # single-char stand-in for an ezafe marker inside the DP

_CONSONANTS = {
    "ب": "b", "پ": "p", "ت": "t", "ث": "s", "ج": "j", "چ": "c", "ح": "h", "خ": "x",
    "د": "d", "ذ": "z", "ر": "r", "ز": "z", "ژ": ";", "س": "s", "ش": "$", "ص": "s",
    "ض": "z", "ط": "t", "ظ": "z", "غ": "q", "ف": "f", "ق": "q", "ک": "k", "گ": "g",
    "ل": "l", "م": "m", "ن": "n",
}

# letter -> [(core realization, cost)]. Costs only break ties between
# alignments that reconstruct the same phonemes; lower = more canonical.
_SPECIAL: dict[str, list[tuple[str, float]]] = {
    "ا": [("a", 0), ("@", 0), ("/", 1), ("e", 1), ("o", 1), ("@a", 1), ("/n", 1), ("", 2)],
    "آ": [("@a", 0), ("a", 0)],
    "أ": [("@", 0), ("a", 1)],
    "ؤ": [("@", 0), ("o@", 1), ("u", 1)],
    "ئ": [("@", 0), ("y", 1), ("e@", 1), ("i", 1)],
    "ء": [("@", 0), ("", 1)],
    "ع": [("@", 0), ("", 1)],
    "ه": [("h", 0), ("e", 1), ("", 2)],
    "و": [("v", 0), ("u", 0), ("o", 1), ("ow", 1), ("@o", 1), ("", 2)],
    "ی": [("y", 0), ("i", 0), ("ey", 1), ("iy", 1), ("a", 1), ("", 2)],
    "ة": [("t", 0), ("e", 1), ("", 1)],
    TANWIN_FATH: [("n", 0), ("/n", 0), ("", 1)],
}
# Letters after which a hidden short vowel may follow.
_TAKES_SHORT_VOWEL = set(_CONSONANTS) | {"ا", "أ", "ؤ", "ئ", "ء", "ع", "ه", "و", "ی", "ة"}
_SHORT_VOWELS = [("", 0.0), ("/", 0.1), ("e", 0.1), ("o", 0.1)]
# Romanized digraphs that leak into some annotations ("beshe", "rogh/n");
# matched in the reference but stored as the canonical single symbol.
_DIGRAPHS = {"ش": ("sh", "$"), "غ": ("gh", "q"), "ق": ("gh", "q"), "ژ": ("zh", ";"),
             "چ": ("ch", "c"), "خ": ("kh", "x")}


class Cand(NamedTuple):
    emit: str        # what must appear in the reference phoneme string
    label: str       # what we store as the training label
    cost: float
    ezafe: bool      # True if this realization *is* an ezafe (written ezafe ی)


@lru_cache(maxsize=None)
def candidates(ch: str, word_final: bool = False) -> tuple[Cand, ...]:
    """All realizations a grapheme char may have, given whether it ends a word."""
    if ch in (" ", ZWNJ):
        return (Cand(" ", " ", 0.0, False), Cand("", "", 0.5, False))
    if ch in PUNCT:
        out = [Cand("", "", 0.0, False), Cand(" ", "", 0.0, False)]
        if ch in "؟?":  # annotators sometimes transcribe the question mark as a glottal stop
            out.append(Cand("@", "", 1.0, False))
        return tuple(out)
    if ch in _CONSONANTS:
        c = _CONSONANTS[ch]
        cores = [(c, c, 0.0), (c + c, c + c, 0.5)]
        if ch in _DIGRAPHS:
            emit, lab = _DIGRAPHS[ch]
            cores.append((emit, lab, 0.2))
    elif ch in _SPECIAL:
        cores = [(c, c, cost) for c, cost in _SPECIAL[ch]]
    else:
        return ()
    vowels = _SHORT_VOWELS if ch in _TAKES_SHORT_VOWEL else [("", 0.0)]
    out = []
    for emit, lab, cc in cores:
        for v, vc in vowels:
            if ch == "ی" and word_final and emit == "y" and v == "e":
                continue  # word-final "ye" is the written ezafe, handled below
            out.append(Cand(emit + v, lab + v, cc + vc, False))
    if ch == "ی" and word_final:
        # Written ezafe ("کتاب‌های خوب" -> "ketabhaye xub"), often left unmarked in the
        # reference; we store it as an ezafe flag so ezafe labels stay consistent.
        out.append(Cand("ye", "", 0.0, True))
    return tuple(out)


def _encode_phonemes(r2: str) -> str:
    """Replace each word's ezafe suffix with the EZ sentinel: 'xaneye1' -> 'xane\x01'."""
    words = []
    for w in r2.split():
        bare, ez = split_ezafe(w)
        words.append(bare + (EZ if ez else ""))
    return " ".join(words)


@dataclass
class Alignment:
    graphemes: str
    labels: list[str]   # one phoneme string per grapheme char
    ezafe: list[bool]   # ezafe flag per grapheme char (only ever True on a letter)
    cost: float

    def to_r2(self) -> str:
        return decode(self.graphemes, self.labels, self.ezafe)


def decode(graphemes: str, labels: list[str], ezafe: list[bool]) -> str:
    """Rebuild an R2 string from per-char labels and ezafe flags.

    Every written space is a word boundary, even where the reference joins two
    written words into one phonological word ("ماهی ها" -> "mahiha"): the surface
    phonemes are identical, and keeping the input's segmentation lets word-level
    metrics line hypotheses up with the text.
    """
    out: list[str] = []
    cur = ""
    for ch, lab, ez in zip(graphemes, labels, ezafe):
        if lab == " " or ch == " ":
            if cur:
                out.append(cur)
            cur = ""
            continue
        cur += lab
        if ez and cur:
            cur = join_ezafe(cur, True)
    if cur:
        out.append(cur)
    return " ".join(w for w in out if w)


def align(graphemes: str, r2: str, band: int = 16) -> Alignment | None:
    """Min-cost monotonic alignment of normalized graphemes to R2 phonemes."""
    ph = " ".join(_encode_phonemes(r2).split())
    n, m = len(graphemes), len(ph)
    cands = []
    for i, ch in enumerate(graphemes):
        final = i + 1 == n or graphemes[i + 1] == " " or graphemes[i + 1] in PUNCT
        c = candidates(ch, final)
        if not c:
            return None
        cands.append(c)
    ratio = m / max(n, 1)
    INF = float("inf")
    # best[i] maps phoneme position j -> (cost, back-pointer j, label, ezafe)
    best: list[dict[int, tuple[float, int, str, bool]]] = [dict() for _ in range(n + 1)]
    best[0][0] = (0.0, -1, "", False)
    for i in range(n):
        if not best[i]:
            return None
        center = i * ratio
        is_letter = graphemes[i] not in PUNCT and graphemes[i] not in (" ", ZWNJ)
        for j, (cost, *_rest) in best[i].items():
            if abs(j - center) > band:
                continue
            for cand in cands[i]:
                k = j + len(cand.emit)
                if ph[j:k] != cand.emit:
                    continue
                for ez in (False, True) if is_letter else (False,):
                    kk = k
                    if ez:  # consume an explicit ezafe marker after this letter
                        if kk < m and ph[kk] == EZ:
                            kk += 1
                        else:
                            continue
                    elif cand.ezafe and kk < m and ph[kk] == EZ:
                        kk += 1  # written ezafe that is *also* marked: consume the marker
                    nc = cost + cand.cost
                    if nc < best[i + 1].get(kk, (INF,))[0]:
                        best[i + 1][kk] = (nc, j, cand.label, ez or cand.ezafe)
    if m not in best[n]:
        return None
    labels, ezafe = [""] * n, [False] * n
    j = m
    for i in range(n, 0, -1):
        _, pj, lab, ez = best[i][j]
        labels[i - 1], ezafe[i - 1] = lab, ez
        j = pj
    return Alignment(graphemes, labels, ezafe, best[n][m][0])
