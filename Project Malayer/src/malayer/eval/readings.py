"""Per-written-word readings (R2) from different sources, aligned to grapheme tokens.

Forced-choice tests need one reading per *written word* so that the readings
from two systems can be swapped word by word.
"""

from __future__ import annotations

from malayer.g2p.baselines import EspeakG2P, ipa_to_r2
from malayer.phonemes import join_ezafe

_PUNCT = ".,!?؟،؛:«»()\"'"


def _cores(g: str) -> list[str]:
    return [t.strip(_PUNCT) for t in g.split(" ") if t.strip(_PUNCT)]


def gold_word_readings(item: dict) -> list[str]:
    """Reference reading per written word, from an aligned dataset item."""
    g, labels, ezafe = item["g"], item["labels"], item["ezafe"]
    out, start = [], 0
    for j, ch in enumerate(g + " "):
        if ch == " ":
            if g[start:j].strip(_PUNCT):
                out.append(join_ezafe("".join(labels[start:j]), any(ezafe[start:j])))
            start = j + 1
    return out


def tagged_word_readings(g: str, tagged: list[tuple[str, bool]]) -> list[str]:
    """Readings from Phonemizer.tag_words output."""
    return [join_ezafe(bare, ez) for tok, (bare, ez) in zip(g.split(" "), tagged) if tok.strip(_PUNCT)]


def espeak_word_readings(g: str, esp: EspeakG2P) -> list[str]:
    """espeak's reading per written word (sentence context; per-word fallback)."""
    cores = _cores(g)
    words = ipa_to_r2(esp.ipa(g)).split()
    if len(words) == len(cores):
        return words
    return [esp(c) for c in cores]
