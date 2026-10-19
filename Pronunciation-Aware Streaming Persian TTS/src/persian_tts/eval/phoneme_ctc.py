"""Audio-level pronunciation measurement with a zero-shot phoneme recognizer.

We use facebook/wav2vec2-xlsr-53-espeak-cv-ft (392 multilingual IPA tokens)
and *project* its posteriors onto our Persian R2 inventory with a soft matrix
P(r2 class | ipa token) learned from human recordings (Mana-TTS clips whose
phonemes are known). This turns it into a Persian R2 CTC model, which gives us:

* greedy R2 transcription  -> phoneme error rate vs. an intended reading
* CTC log-likelihood of any candidate R2 sequence -> *minimal-pair forced
  choice*: did the audio say word w as reading A or reading B?

Forced choice compares two likelihoods on the same audio, so it is far more
robust than free transcription — it only needs the recognizer to rank, not to
transcribe perfectly. We validate it on human speech before trusting it on TTS.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

from persian_tts.phonemes import strip_markers

R2_CLASSES = list("/aeoiu@$cj;ybpdtkqgfvszxhmnlr")  # 29 phonemes; class 0 = blank
BLANK = 0

# Seed IPA tokens per R2 phoneme; only used to bootstrap the confusion alignment.
SEED = {
    "/": ["a", "æ"], "a": ["ɑ", "ɑː", "ɒ", "ɔ"], "e": ["e", "ɛ"], "o": ["o", "oː"],
    "i": ["i", "iː"], "u": ["u", "uː"], "@": ["ʔ"], "$": ["ʃ"], "c": ["tʃ"], "j": ["dʒ"],
    ";": ["ʒ"], "y": ["j"], "g": ["ɡ"], "q": ["q", "ɢ", "ʁ", "ɣ"], "x": ["x", "χ"], "r": ["r", "ɾ"],
}


def r2_symbols(r2: str) -> list[str]:
    return [c for c in strip_markers(r2) if c in R2_CLASSES]


_V = set("/aeoiu")


def scoring_symbols(r2: str) -> list[str]:
    """Symbols used for CTC scoring: drops what the recognizer cannot reliably hear.

    Glottal stops ('@', mostly word-initial and weak in fluent speech) and the
    glide 'y' between two vowels ("ertebatiye" ~ "ertebatie", "aya" ~ "aia") are
    removed, so readings that differ only there are not treated as contested.
    """
    syms = [c for c in r2_symbols(r2.replace(" ", "")) if c != "@"]
    return [c for i, c in enumerate(syms)
            if not (c == "y" and 0 < i < len(syms) - 1 and syms[i - 1] in _V and syms[i + 1] in _V)]


def same_for_scoring(a: str, b: str) -> bool:
    return scoring_symbols(a) == scoring_symbols(b)


class PhonemeCTC:
    def __init__(self, model_dir: str | Path = "models/w2v2-phoneme", device: str = "cpu",
                 projection: str | Path | None = "configs/recognizer_projection.json"):
        from transformers import Wav2Vec2ForCTC

        self.model = Wav2Vec2ForCTC.from_pretrained(str(model_dir)).to(device).eval()
        self.device = device
        vocab = json.loads((Path(model_dir) / "vocab.json").read_text())
        self.tokens = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
        self.blank_id = vocab["<pad>"]
        self.class_index = {c: i + 1 for i, c in enumerate(R2_CLASSES)}
        self.proj: torch.Tensor | None = None
        if projection and Path(projection).exists():
            self.set_projection(json.loads(Path(projection).read_text()))

    # ---- acoustic posteriors --------------------------------------------------
    @torch.inference_mode()
    def token_logprobs(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """[T, 392] log-posteriors over IPA tokens for mono float audio."""
        if sr != 16000:
            g = np.gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // g, sr // g)
        audio = audio.astype(np.float32)
        audio = (audio - audio.mean()) / (audio.std() + 1e-7)
        x = torch.from_numpy(audio)[None].to(self.device)
        return self.model(x).logits[0].float().log_softmax(-1).cpu()

    def greedy_tokens(self, lp: torch.Tensor) -> list[str]:
        out, prev = [], None
        for i in lp.argmax(-1).tolist():
            if i != prev and i != self.blank_id:
                out.append(self.tokens[i])
            prev = i
        return out

    # ---- projection onto R2 ---------------------------------------------------
    def set_projection(self, table: dict[str, dict[str, float]]) -> None:
        """table: ipa token -> {r2 class or 'other': prob}. Missing tokens go to 'other'."""
        C = len(R2_CLASSES) + 2  # blank, 29 phonemes, other
        M = torch.zeros(len(self.tokens), C)
        for t_id, tok in enumerate(self.tokens):
            if t_id == self.blank_id:
                M[t_id, BLANK] = 1.0
                continue
            dist = table.get(tok, {"other": 1.0})
            for cls, p in dist.items():
                M[t_id, C - 1 if cls == "other" else self.class_index[cls]] += p
        self.proj = M

    def class_logprobs(self, lp: torch.Tensor) -> torch.Tensor:
        """[T, 31] log P(class) = log sum_token P(token) P(class|token)."""
        return torch.log(lp.exp() @ self.proj + 1e-8)

    def greedy_r2(self, clp: torch.Tensor) -> str:
        out, prev = [], None
        other = clp.shape[1] - 1
        for i in clp.argmax(-1).tolist():
            if i != prev and i not in (BLANK, other):
                out.append(R2_CLASSES[i - 1])
            prev = i
        return "".join(out)

    def ctc_logprob(self, clp: torch.Tensor, r2: str) -> float:
        """log P(r2 sequence | audio) under CTC, on `scoring_symbols` (spaces, ezafe markers,
        glottal stops and inter-vocalic glides ignored)."""
        target = torch.tensor([self.class_index[s] for s in scoring_symbols(r2)], dtype=torch.long)
        if len(target) == 0 or len(target) > clp.shape[0]:
            return float("-inf")
        loss = F.ctc_loss(clp[:, None, :], target[None], torch.tensor([clp.shape[0]]),
                          torch.tensor([len(target)]), blank=BLANK, reduction="sum", zero_infinity=False)
        return -loss.item()

    def forced_choice(self, clp: torch.Tensor, words: list[str], idx: int, alt: str) -> float:
        """Log-likelihood ratio  log P(sentence) - log P(sentence with words[idx]=alt).

        > 0 means the audio is more consistent with the given reading of word idx.
        """
        alt_words = list(words)
        alt_words[idx] = alt
        return self.ctc_logprob(clp, " ".join(words)) - self.ctc_logprob(clp, " ".join(alt_words))
