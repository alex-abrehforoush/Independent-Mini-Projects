"""Persian text normalization for the G2P front-end.

The goal is a canonical character stream the tagger can rely on: one code point
per letter shape, explicit ZWNJ, no decorative marks, collapsed whitespace.
Normalization is deliberately *length-aware*: `normalize` returns plain text,
and every transformation is character-local so tagger labels stay aligned with
what the user typed after the (cheap, deterministic) mapping below.
"""

from __future__ import annotations

import re
import unicodedata

ZWNJ = "‌"
TANWIN_FATH = "ً"  # ً  kept: it is pronounced (/an/) in Persian loanwords

# Letter-shape unification (Arabic code points -> Persian code points).
_CHAR_MAP = {
    "ي": "ی",  # Arabic yeh
    "ى": "ی",  # alef maksura
    "ك": "ک",  # Arabic kaf
    "ۀ": "ه",  # heh with yeh above (ezafe-marked heh) -> plain heh + ZWNJ ی handled below
    "إ": "ا",  # alef with hamza below
    "ٱ": "ا",  # alef wasla
    "ە": "ه",  # ae
    "‍": "",  # ZWJ
    "‎": "",  # LRM
    "‏": "",  # RLM
    " ": " ",  # nbsp
    "ـ": "",  # tatweel
    "ٔ": "",  # hamza above (combining) - pronounced form handled via letter context
}

# Short-vowel diacritics, sukun, shadda, superscript alef: rarely present in
# running text and the training data does not contain them, so we drop them.
_DROP_MARKS = {"ٌ", "ٍ", "َ", "ُ", "ِ", "ّ", "ْ", "ٰ"}

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_DIGIT_MAP = {d: str(i) for i, d in enumerate(PERSIAN_DIGITS)} | {
    d: str(i) for i, d in enumerate(ARABIC_DIGITS)
}

PERSIAN_LETTERS = set("آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیءأؤئة")
PUNCT = set(".,،؛;:!?؟«»\"'()[]{}-–—…/")


def normalize(text: str) -> str:
    """Canonicalize Persian text (letters, ZWNJ, whitespace, digits to ASCII)."""
    text = unicodedata.normalize("NFC", text)
    # Written ezafe on silent heh ("ۀ") becomes heh + ZWNJ + yeh, which is the
    # spelling the tagger is trained on ("خانه‌ی").
    text = text.replace("ۀ", "ه" + ZWNJ + "ی").replace("هٔ", "ه" + ZWNJ + "ی")
    out = []
    for ch in text:
        if ch in _DROP_MARKS:
            continue
        ch = _CHAR_MAP.get(ch, ch)
        ch = _DIGIT_MAP.get(ch, ch)
        out.append(ch)
    text = "".join(out)
    # ZWNJ adjacent to a space (or at a word edge) is meaningless; repeats collapse.
    text = re.sub(f"{ZWNJ}+", ZWNJ, text)
    text = re.sub(f"\\s*{ZWNJ}\\s+|\\s+{ZWNJ}\\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip().strip(ZWNJ)
    return text


def is_persian_only(text: str) -> bool:
    """True if every char is a Persian letter, ZWNJ, space, tanwin or punctuation."""
    return all(ch in PERSIAN_LETTERS or ch in PUNCT or ch in (" ", ZWNJ, TANWIN_FATH) for ch in text)
