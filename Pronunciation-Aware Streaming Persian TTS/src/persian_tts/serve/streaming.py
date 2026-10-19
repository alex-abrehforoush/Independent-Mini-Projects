"""Incremental text -> audio for LLM-driven voice agents.

Text arrives in arbitrary fragments (LLM tokens). A word is *complete* once
whitespace follows it. Word w is *committed* (its phonemes frozen) as soon as
`lookahead` further words are complete, using the tagger on the visible prefix
without <eos>. Committed words accumulate into a phrase that is synthesized at
punctuation (or when it grows too long), trading a little latency for natural
phrase-level prosody. The first phrase is capped shorter so audio starts early.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from persian_tts.g2p.infer import Phonemizer
from persian_tts.text.normalize import normalize
from persian_tts.text.numbers import expand_numbers
from persian_tts.tts.synth import Synthesizer

_BREAK = tuple(".!?؟،؛:,")


@dataclass
class AudioChunk:
    audio: np.ndarray
    sample_rate: int
    text: str
    ipa: str
    t_emit: float  # seconds since session start


@dataclass
class StreamingSession:
    synth: Synthesizer
    phonemizer: Phonemizer | None  # None -> espeak front-end (control arm)
    lookahead: int = 1
    min_phrase_words: int = 3
    max_phrase_words: int = 12
    first_phrase_words: int = 5  # shorter first phrase -> earlier first audio
    _pending: str = ""
    _words: list[str] = field(default_factory=list)      # complete, normalized words
    _committed: list[tuple[str, str | None, bool]] = field(default_factory=list)
    _phrase_start: int = 0
    t0: float = field(default_factory=time.perf_counter)

    def feed(self, fragment: str) -> list[AudioChunk]:
        self._pending += fragment
        if not any(ch.isspace() for ch in self._pending):
            return []
        # split on any whitespace; the last token stays pending until whitespace follows it
        parts = self._pending.split()
        complete = parts if self._pending[-1].isspace() else parts[:-1]
        self._pending = "" if self._pending[-1].isspace() else parts[-1]
        for w in complete:
            self._words.extend(normalize(expand_numbers(normalize(w))).split())
        n_ready = len(self._words) - self.lookahead
        return self._commit_until(n_ready, eos=False)

    def finish(self) -> list[AudioChunk]:
        if self._pending.strip():
            self._words.extend(normalize(expand_numbers(normalize(self._pending))).split())
        self._pending = ""
        return self._commit_until(len(self._words), eos=True, flush=True)

    def _commit_until(self, n_ready: int, eos: bool, flush: bool = False) -> list[AudioChunk]:
        start = len(self._committed)
        for w in range(start, n_ready):
            if self.phonemizer is None:
                self._committed.append((self._words[w], None, False))
                continue
            # exactly `lookahead` words of right context, even if more have already arrived,
            # so live behaviour equals the evaluated streaming simulation
            end = min(w + self.lookahead + 1, len(self._words))
            tagged = self.phonemizer.tag_words(" ".join(self._words[:end]), lookahead=None,
                                               eos=eos and end == len(self._words))
            self._committed.append((self._words[w], *tagged[w]))
        chunks = []
        while True:
            phrase = self._committed[self._phrase_start:]
            if not phrase:
                break
            cut = None
            limit = self.first_phrase_words if self._phrase_start == 0 else self.max_phrase_words
            for i, (tok, _, _) in enumerate(phrase):
                if (tok.endswith(_BREAK) and i + 1 >= self.min_phrase_words) or i + 1 >= limit:
                    cut = i + 1
                    break
            if cut is None:
                if flush:
                    cut = len(phrase)
                else:
                    break
            chunks.append(self._synthesize(phrase[:cut]))
            self._phrase_start += cut
        return chunks

    def _synthesize(self, phrase: list[tuple[str, str | None, bool]]) -> AudioChunk:
        tokens = [t for t, _, _ in phrase]
        words = self.synth.frontend.words_from_tagged(tokens, [(b, e) for _, b, e in phrase])
        ipa = " ".join(w.ipa for w in words)
        audio = self.synth.ipa_to_audio(ipa)
        return AudioChunk(audio, self.synth.sample_rate, " ".join(tokens), ipa, time.perf_counter() - self.t0)
