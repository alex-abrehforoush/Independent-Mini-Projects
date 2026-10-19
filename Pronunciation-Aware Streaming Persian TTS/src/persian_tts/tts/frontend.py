"""TTS front-end: Persian text -> IPA phonemes for the Piper voice.

The Mana Piper voice was trained on espeak-ng 'fa' output, so it expects
espeak's conventions: stress marks (ˈ ˌ), length marks (ː), the "q1" symbol,
no word-initial glottal stop. Our tagger fixes espeak's *segmental* errors
(missing short vowels, ezafe, homographs). To stay in-distribution we keep
espeak's suprasegmental marks: per word, we align our phonemes to espeak's and
transfer its stress/length marks onto our symbols ("minimal-edit" phonemes).
"""

from __future__ import annotations

from dataclasses import dataclass

from piper.phonemize_espeak import EspeakPhonemizer

from persian_tts.g2p.infer import Phonemizer
from persian_tts.phonemes import join_ezafe
from persian_tts.text.normalize import is_persian_only, normalize
from persian_tts.text.numbers import expand_numbers

_R2_TO_IPA_SYM = {
    "/": "a", "a": "ɑ", "e": "e", "o": "o", "i": "i", "u": "u", "@": "ʔ", "$": "ʃ",
    "c": "tʃ", "j": "dʒ", ";": "ʒ", "y": "j", "g": "ɡ", "q": "q1",
}
_IPA_VOWELS = {"a", "ɑ", "e", "o", "i", "u"}
_MULTI = ("tʃ", "dʒ", "q1")
_STRESS = ("ˈ", "ˌ")
_PUNCT_MAP = {"،": ",", "؟": "?", "؛": ";", ".": ".", "!": "!", ",": ",", ":": ":", "?": "?"}


def tokenize_ipa(ipa: str) -> list[tuple[str, str, bool]]:
    """espeak IPA word -> [(stress_mark_before, symbol, long)]."""
    out: list[tuple[str, str, bool]] = []
    i, stress = 0, ""
    while i < len(ipa):
        ch = ipa[i]
        if ch in _STRESS:
            stress = ch
            i += 1
            continue
        if ch == "ː":
            if out:
                s, sym, _ = out[-1]
                out[-1] = (s, sym, True)
            i += 1
            continue
        sym = next((m for m in _MULTI if ipa.startswith(m, i)), ch)
        out.append((stress, sym, False))
        stress = ""
        i += len(sym)
    return out


def r2_word_to_symbols(bare: str, ezafe: bool) -> list[str]:
    word = join_ezafe(bare, ezafe).rstrip("1")
    syms = [_R2_TO_IPA_SYM.get(c, c) for c in word]
    if syms and syms[0] == "ʔ" and len(syms) > 1:  # espeak never writes word-initial ʔ
        syms = syms[1:]
    return syms


def _sub_cost(a: str, b: str) -> float:
    """Substitution cost: identical 0, vowel<->vowel or consonant<->consonant 0.5, else 1.5."""
    if a == b:
        return 0.0
    return 0.5 if (a in _IPA_VOWELS) == (b in _IPA_VOWELS) else 1.5


def transfer_marks(ours: list[str], espeak: list[tuple[str, str, bool]]) -> str:
    """Edit-distance align our symbols to espeak's; copy stress/length marks across."""
    n, m = len(ours), len(espeak)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = _sub_cost(ours[i - 1], espeak[j - 1][1])
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + sub)
    # backtrace -> for each of our symbols, the espeak token it aligns to (or None)
    i, j, link = n, m, [None] * n
    pending_stress = [""] * (n + 1)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub = _sub_cost(ours[i - 1], espeak[j - 1][1])
            if d[i][j] == d[i - 1][j - 1] + sub:
                link[i - 1] = j - 1
                i, j = i - 1, j - 1
                continue
        if j > 0 and (i == 0 or d[i][j] == d[i][j - 1] + 1):
            if espeak[j - 1][0]:  # stress on a deleted espeak symbol -> move to our next symbol
                pending_stress[i] = pending_stress[i] or espeak[j - 1][0]
            j -= 1
        else:
            i -= 1
    out = []
    for k, sym in enumerate(ours):
        stress, long_ = pending_stress[k], False
        if link[k] is not None:
            s, esym, l = espeak[link[k]]
            stress = s or stress
            long_ = l and sym in _IPA_VOWELS and esym in _IPA_VOWELS
        out.append(stress + sym + ("ː" if long_ else ""))
    return "".join(out)


@dataclass
class WordPhones:
    text: str
    r2: str
    ipa: str


class PersianFrontend:
    """Text -> IPA string for Piper. `phonemizer=None` gives the stock espeak front-end."""

    def __init__(self, phonemizer: Phonemizer | None = None, lookahead: int | None = None):
        self.phonemizer, self.lookahead = phonemizer, lookahead
        self._espeak = EspeakPhonemizer()

    def espeak_word(self, word: str) -> str:
        return "".join("".join(s) for s in self._espeak.phonemize("fa", word)).strip(" .")

    def prepare(self, text: str) -> str:
        """Normalization shared by offline and streaming paths."""
        return normalize(expand_numbers(normalize(text)))

    def words(self, text: str) -> list[WordPhones]:
        g = self.prepare(text)
        tokens = g.split(" ")
        if self.phonemizer is None:
            tagged = [(None, False)] * len(tokens)
        else:
            tagged = self.phonemizer.tag_words(g, lookahead=self.lookahead)
        return self.words_from_tagged(tokens, tagged)

    def words_from_tagged(self, tokens: list[str], tagged: list[tuple[str | None, bool]]) -> list[WordPhones]:
        """IPA per token given the tagger's (bare R2, ezafe) per token (None = use espeak as-is)."""
        punct = "".join(_PUNCT_MAP)
        cores = [t.strip(punct + "«»()\"'") for t in tokens]
        # espeak in sentence context (it de-stresses function words); per word if counts differ
        sent = [w.strip(punct) for w in self.espeak_word(" ".join(tokens)).split()]
        sent = [w for w in sent if w]
        n_cores = sum(1 for c in cores if c)
        esp_iter = iter(sent) if len(sent) == n_cores else None
        out = []
        for tok, core, (bare, ez) in zip(tokens, cores, tagged):
            trail = "".join(_PUNCT_MAP[c] for c in tok[len(tok.rstrip(punct)):] if c in _PUNCT_MAP)
            if not core:
                if out and trail:
                    out[-1].ipa += trail
                continue
            esp = next(esp_iter) if esp_iter else self.espeak_word(core)
            if bare is None or not bare or not is_persian_only(core):
                # Not Persian script (Latin brand names, Finglish, ...): the tagger has no
                # labels for it, so read it with espeak rather than emitting silence.
                ipa, r2 = esp, ""
            else:
                ipa = transfer_marks(r2_word_to_symbols(bare, ez), tokenize_ipa(esp))
                r2 = join_ezafe(bare, ez)
            out.append(WordPhones(tok, r2, ipa + trail))
        return out

    def ipa(self, text: str) -> str:
        return " ".join(w.ipa for w in self.words(text))
