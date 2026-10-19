"""Expand digits into Persian number words ("1402" -> "هزار و چهارصد و دو")."""

from __future__ import annotations

import re

_UNITS = ["صفر", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"]
_TEENS = ["ده", "یازده", "دوازده", "سیزده", "چهارده", "پانزده", "شانزده", "هفده", "هجده", "نوزده"]
_TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
_HUNDREDS = ["", "صد", "دویست", "سیصد", "چهارصد", "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد"]
_SCALES = ["", "هزار", "میلیون", "میلیارد", "تریلیون"]
_AND = " و "


def _below_thousand(n: int) -> str:
    parts = []
    h, rest = divmod(n, 100)
    if h:
        parts.append(_HUNDREDS[h])
    if 10 <= rest < 20:
        parts.append(_TEENS[rest - 10])
    else:
        t, u = divmod(rest, 10)
        if t:
            parts.append(_TENS[t])
        if u:
            parts.append(_UNITS[u])
    return _AND.join(parts)


def int_to_words(n: int) -> str:
    if n == 0:
        return _UNITS[0]
    if n < 0:
        return "منفی " + int_to_words(-n)
    groups = []
    scale = 0
    while n:
        n, g = divmod(n, 1000)
        if g:
            words = _below_thousand(g)
            if scale:
                # "هزار" rather than "یک هزار"
                words = _SCALES[scale] if (g == 1 and scale == 1) else f"{words} {_SCALES[scale]}"
            groups.append(words)
        scale += 1
        if scale >= len(_SCALES) and n:
            raise ValueError("number too large")
    return _AND.join(reversed(groups))


def expand_numbers(text: str) -> str:
    """Replace ASCII-digit numbers (after normalization) with words; decimals use 'ممیز'."""
    def repl(m: re.Match) -> str:
        whole, frac = m.group(1).replace(",", ""), m.group(2)
        out = int_to_words(int(whole))
        if frac:
            out += " ممیز " + int_to_words(int(frac))
        return f" {out} "

    text = re.sub(r"(\d[\d,]*)(?:[./٫](\d+))?", repl, text)
    return re.sub(r"\s+", " ", text).strip()
