"""Baseline G2P systems, all emitting R2 so they are scored identically.

* EspeakG2P   - espeak-ng 'fa' rules (what the stock Piper voice uses).
* LexiconG2P  - context-free dictionary: each written word gets its most
                frequent pronunciation in our training alignments; no ezafe;
                OOV words fall back to espeak. This isolates what *context* buys.
"""

from __future__ import annotations

import collections
import re

from piper.phonemize_espeak import EspeakPhonemizer

from persian_tts.phonemes import VOWELS_R2

_IPA_TO_R2 = [("tʃ", "c"), ("dʒ", "j"), ("q1", "q"), ("ʃ", "$"), ("ʒ", ";"), ("ɑ", "a"), ("a", "/"),
              ("j", "y"), ("ʔ", "@"), ("ɡ", "g"), ("χ", "x"), ("ɾ", "r")]
_DROP = re.compile(r"[ˈˌː.,!?;:«»\"()]")


def ipa_to_r2(ipa: str) -> str:
    """espeak IPA -> R2. Adds the word-initial glottal stop that R2 writes and espeak omits."""
    words = []
    for w in _DROP.sub("", ipa).split():
        # left-to-right scan; multi-char symbols (tʃ, dʒ, q1) are listed first
        out, i = [], 0
        while i < len(w):
            for src, dst in _IPA_TO_R2:
                if w.startswith(src, i):
                    out.append(dst)
                    i += len(src)
                    break
            else:
                out.append(w[i])
                i += 1
        r2 = "".join(out)
        if r2 and r2[0] in VOWELS_R2:
            r2 = "@" + r2
        words.append(r2)
    return " ".join(words)


class EspeakG2P:
    name = "espeak-ng"

    def __init__(self):
        self._ph = EspeakPhonemizer()

    def ipa(self, text: str) -> str:
        return " ".join("".join(s) for s in self._ph.phonemize("fa", text))

    def __call__(self, text: str) -> str:
        return ipa_to_r2(self.ipa(text))


class LexiconG2P:
    name = "lexicon (context-free)"

    def __init__(self, train_items: list[dict], fallback: EspeakG2P | None = None):
        counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for it in train_items:
            for gw, pw in _word_pairs(it["g"], it["labels"]):
                counts[gw][pw] += 1
        self.lex = {w: c.most_common(1)[0][0] for w, c in counts.items()}
        self.fallback = fallback or EspeakG2P()
        self.oov = 0

    def __call__(self, text: str) -> str:
        out = []
        for w in text.split():
            key = w.strip(".,!?؟،؛:«»()\"'")
            if not key:
                continue
            if key in self.lex:
                out.append(self.lex[key])
            else:
                self.oov += 1
                out.append(self.fallback(key))
        return " ".join(p for p in out if p)


def _word_pairs(g: str, labels: list[str]):
    """(written word, bare pronunciation) pairs from an aligned sentence."""
    gw, pw = "", ""
    for c, l in zip(g + " ", labels + [" "]):
        if c == " ":
            key = gw.strip(".,!?؟،؛:«»()\"'")
            if key and pw:
                yield key, pw
            gw, pw = "", ""
        else:
            gw += c
            pw += l
