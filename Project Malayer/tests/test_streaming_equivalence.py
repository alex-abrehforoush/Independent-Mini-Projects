"""The live StreamingSession must commit exactly what the evaluation's streaming simulation predicts.

Otherwise the lookahead trade-off reported in the README would not describe the deployed system.
Skipped until a trained model exists.
"""

import random
from pathlib import Path

import pytest

RUN = Path("runs/g2p_base")
pytestmark = pytest.mark.skipif(not (RUN / "model.pt").exists(), reason="no trained model")


class _NullSynth:
    """Captures committed words instead of synthesizing audio."""

    sample_rate = 22050

    def __init__(self):
        self.frontend = self
        self.committed = []

    def words_from_tagged(self, tokens, tagged):
        self.committed += [(t, b, e) for t, (b, e) in zip(tokens, tagged)]
        return []

    def ipa_to_audio(self, ipa):
        import numpy as np

        return np.zeros(1, dtype="float32")


@pytest.mark.parametrize("k", [0, 1, 2])
def test_session_matches_simulation(k):
    from malayer.g2p.infer import Phonemizer
    from malayer.phonemes import join_ezafe
    from malayer.serve.streaming import StreamingSession

    ph = Phonemizer.from_dir(RUN)
    rng = random.Random(k)
    for text in ["دوران انتظار امام زمان سخت است.", "کتاب خوب من روی میز بود و او آن را برد."]:
        synth = _NullSynth()
        s = StreamingSession(synth, ph, lookahead=k)
        i = 0
        while i < len(text):  # LLM-like fragments
            n = rng.randint(1, 4)
            s.feed(text[i: i + n])
            i += n
        s.finish()
        live = [join_ezafe(b, e) for _, b, e in synth.committed]
        sim = [join_ezafe(b, e) for b, e in ph.tag_words(text, lookahead=k)]
        assert live == sim


def test_non_persian_tokens_are_not_silent():
    """Regression: Latin-script tokens produced empty phonemes (silent audio), caught by the
    'all_silence' guardrail in the shifted-traffic simulation."""
    from malayer.g2p.infer import Phonemizer
    from malayer.tts.frontend import PersianFrontend

    fe = PersianFrontend(Phonemizer.from_dir(RUN))
    for text in ["salam, chetori?", "من با iPhone کار می‌کنم"]:
        assert all(w.ipa.strip(" ,.?") for w in fe.words(text)), fe.words(text)


def test_short_phrases_are_audible():
    """Regression: unnormalized VITS output made 2-word phrases ~-40 dBFS ('all_silence')."""
    from malayer.serve.guardrails import check_audio
    from malayer.tts.frontend import PersianFrontend
    from malayer.tts.synth import Synthesizer

    s = Synthesizer(PersianFrontend())
    for text in ["بشن .", "کمتر می‌شه ."]:
        r = s.synthesize(text)
        assert "all_silence" not in check_audio(r.audio, r.sample_rate, 6).violations
