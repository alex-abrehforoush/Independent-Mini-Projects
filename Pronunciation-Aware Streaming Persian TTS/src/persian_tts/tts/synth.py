"""Speech synthesis: IPA from our front-end -> Piper (VITS) ONNX voice."""

from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from piper import PiperVoice
from piper.config import SynthesisConfig

from persian_tts.tts.frontend import PersianFrontend

DEFAULT_VOICE = Path("models/mana-piper/fa_IR-mana-medium.onnx")


@dataclass
class SynthResult:
    audio: np.ndarray  # float32 in [-1, 1]
    sample_rate: int
    ipa: str
    frontend_ms: float
    acoustic_ms: float

    @property
    def rtf(self) -> float:
        """Real-time factor: compute time / audio duration (lower is faster)."""
        return (self.frontend_ms + self.acoustic_ms) / 1000 / max(len(self.audio) / self.sample_rate, 1e-6)


class Synthesizer:
    def __init__(self, frontend: PersianFrontend, voice_path: Path = DEFAULT_VOICE, use_cuda: bool = False):
        self.frontend = frontend
        self.voice = PiperVoice.load(str(voice_path), use_cuda=use_cuda)
        self.sample_rate = self.voice.config.sample_rate
        self._known = set(self.voice.config.phoneme_id_map)

    def ipa_to_audio(self, ipa: str, length_scale: float | None = None, peak: float = 0.9) -> np.ndarray:
        phonemes = [p for p in unicodedata.normalize("NFD", ipa) if p in self._known]
        ids = self.voice.phonemes_to_ids(phonemes)
        cfg = SynthesisConfig(length_scale=length_scale)
        audio = self.voice.phoneme_ids_to_audio(ids, cfg).astype(np.float32)
        # The raw VITS output is unnormalized: short phrases came out ~25 dB quieter than
        # sentences (caught by the all_silence guardrail). Peak-normalize like Piper's
        # synthesize(), leaving headroom below the clipping guardrail.
        m = float(np.abs(audio).max()) if audio.size else 0.0
        return audio * (peak / m) if m > 1e-8 else np.zeros_like(audio)

    def synthesize(self, text: str, length_scale: float | None = None) -> SynthResult:
        t0 = time.perf_counter()
        ipa = self.frontend.ipa(text)
        t1 = time.perf_counter()
        audio = self.ipa_to_audio(ipa, length_scale)
        t2 = time.perf_counter()
        return SynthResult(audio, self.sample_rate, ipa, (t1 - t0) * 1000, (t2 - t1) * 1000)
