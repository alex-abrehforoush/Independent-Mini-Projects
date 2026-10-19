"""Phoneme representations and conversions.

Internal representation ("R2") is the GE2PE / HomoRich "Mapped Phoneme" format,
one ASCII char per phoneme:

    /  -> a  (short a, IPA a/æ)      a -> ɒ  (long A, IPA ɑ)
    e o i u                           @ -> ʔ  (glottal stop)
    $ -> ʃ   c -> tʃ   j -> dʒ   ; -> ʒ   x q y(=j) and the usual consonants

Ezafe is marked with a trailing "1" after its vowel ("ketabe1", "xaneye1").
Inside the tagger we store words *without* the ezafe suffix plus a boolean
flag, because the surface form of ezafe is fully determined by the last phoneme
of the bare word (vowel -> "ye", consonant -> "e").

SentenceBench uses "R1" (A=ɒ, a=a, ?=ʔ, S=ʃ, C=tʃ, Z=ʒ, hyphenated "-e"/"-ye").
"""

from __future__ import annotations

VOWELS_R2 = set("/aeoiu")
EZAFE = "1"

_R1_TO_R2 = {"a": "/", "A": "a", "?": "@", "S": "$", "C": "c", "Z": ";"}

# R2 -> IPA (espeak-ng 'fa' conventions, which is what the Piper voice was trained on).
R2_TO_IPA = {
    "/": "a", "a": "ɑ", "e": "e", "o": "o", "i": "i", "u": "u",
    "@": "ʔ", "$": "ʃ", "c": "tʃ", "j": "dʒ", ";": "ʒ", "y": "j", "g": "ɡ",
    "b": "b", "p": "p", "t": "t", "d": "d", "k": "k", "q": "q", "f": "f", "v": "v",
    "s": "s", "z": "z", "x": "x", "h": "h", "m": "m", "n": "n", "l": "l", "r": "r",
}


def r1_to_r2(r1: str) -> str:
    """Convert a SentenceBench (R1) phoneme string to R2.

    Hyphenated ezafe ("ketAb-e", "xAne-ye") becomes the explicit marker; the
    enclitic conjunction "-o" becomes its own word (it is the grapheme "و");
    any other hyphen (prefix "mi-", plural "-hA") is a morpheme join and is removed.
    """
    words = []
    for w in r1.split():
        w = w.strip(".,!،؟")  # sentence punctuation; ASCII '?' is the R1 glottal stop
        tail = ""
        if w.endswith("-o"):  # enclitic conjunction: its own word, like the grapheme "و"
            w, tail = w[:-2], "o"
        if w.endswith("-ye") or w.endswith("-e"):
            w = w.rsplit("-", 1)[0] + w.rsplit("-", 1)[1] + EZAFE
        w = w.replace("-", "")
        words.append("".join(_R1_TO_R2.get(c, c) for c in w))
        words.append(tail)
    return " ".join(w for w in words if w)


def split_ezafe(word: str) -> tuple[str, bool]:
    """'xaneye1' -> ('xane', True); 'ketabe1' -> ('ketab', True); 'ketab' -> ('ketab', False)."""
    if not word.endswith(EZAFE):
        return word, False
    w = word[:-1]
    if w.endswith("ye") and len(w) >= 3 and w[-3] in VOWELS_R2:
        return w[:-2], True
    if w.endswith("e"):
        return w[:-1], True
    return w, True  # malformed marker; keep the word, keep the flag


def join_ezafe(bare: str, ezafe: bool) -> str:
    """Inverse of `split_ezafe`."""
    if not ezafe or not bare:
        return bare
    return bare + ("ye" if bare[-1] in VOWELS_R2 else "e") + EZAFE


def strip_markers(r2: str) -> str:
    """Surface phonemes only (drop the ezafe marker) - used for PER."""
    return r2.replace(EZAFE, "")


def r2_to_ipa(r2: str) -> str:
    """R2 -> IPA characters understood by the Piper voice's phoneme_id_map."""
    return "".join(R2_TO_IPA.get(c, c) for c in strip_markers(r2))


def canonical_r2(r2: str) -> str:
    """Canonical R2: ezafe realized by rule ("q/vie1" -> "q/viye1"), single spaces."""
    return " ".join(join_ezafe(*split_ezafe(w)) for w in r2.split())
