"""Inference: offline and streaming (bounded-lookahead) phonemization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from persian_tts.g2p.align import decode
from persian_tts.g2p.model import G2PTagger, Vocab
from persian_tts.text.normalize import normalize


@dataclass
class WordPrediction:
    graphemes: str
    labels: list[str]
    ezafe: list[bool]

    @property
    def r2(self) -> str:
        return decode(self.graphemes, self.labels, self.ezafe)


class Phonemizer:
    """Wraps the tagger. `lookahead=None` means offline (whole sentence visible)."""

    def __init__(self, model: G2PTagger, vocab: Vocab, device: str = "cpu"):
        self.model, self.vocab, self.device = model.to(device).eval(), vocab, device

    @classmethod
    def from_dir(cls, run_dir: str | Path, device: str = "cpu") -> "Phonemizer":
        run_dir = Path(run_dir)
        return cls(G2PTagger.load(run_dir / "model.pt"), Vocab.load(run_dir / "vocab.json"), device)

    @torch.inference_mode()
    def _tag_batch(self, texts: list[str], eos: list[bool]) -> list[tuple[list[str], list[bool]]]:
        enc = [self.vocab.encode(t, e) for t, e in zip(texts, eos)]
        T = max(len(e) for e in enc)
        ids = torch.zeros(len(enc), T, dtype=torch.long)
        for i, e in enumerate(enc):
            ids[i, : len(e)] = torch.tensor(e)
        lab_logits, ez_logits = self.model(ids.to(self.device))
        lab = lab_logits.argmax(-1).cpu()
        ez = ez_logits.argmax(-1).cpu().bool()
        out = []
        for i, t in enumerate(texts):
            labels = [self.vocab.labels[j] for j in lab[i, : len(t)].tolist()]
            ezafe = [bool(x) and c not in " ‌" for x, c in zip(ez[i, : len(t)].tolist(), t)]
            out.append((labels, ezafe))
        return out

    def phonemize(self, text: str, normalized: bool = False) -> str:
        """Offline: whole sentence, returns R2 (ezafe marked with '1')."""
        g = text if normalized else normalize(text)
        labels, ezafe = self._tag_batch([g], [True])[0]
        return decode(g, labels, ezafe)

    def phonemize_batch(self, texts: list[str]) -> list[str]:
        res = self._tag_batch(texts, [True] * len(texts))
        return [decode(t, l, e) for t, (l, e) in zip(texts, res)]

    def _tag_chars(self, g: str, lookahead: int | None, eos: bool = True) -> tuple[list[str], list[bool]]:
        """Per-char labels/ezafe for normalized text, offline or with bounded lookahead.

        Streaming: word w is committed once words up to w+lookahead have arrived;
        each commit re-tags the visible prefix (no <eos> unless the sentence is
        complete) and keeps only the newly committed word, exactly as the
        streaming server does. Committed words are never revised.
        """
        if lookahead is None:
            return self._tag_batch([g], [eos])[0]
        spans, start = [], 0
        for j, c in enumerate(g + " "):
            if c == " ":
                spans.append((start, j))
                start = j + 1
        n = len(spans)
        prefixes, eos = [], []
        for w in range(n):
            last = min(w + lookahead, n - 1)
            prefixes.append(g[: spans[last][1]])
            eos.append(last == n - 1)
        labels, ezafe = [" "] * len(g), [False] * len(g)
        for (s, e), (labs, ezs) in zip(spans, self._tag_batch(prefixes, eos)):
            labels[s:e], ezafe[s:e] = labs[s:e], ezs[s:e]
        return labels, ezafe

    def phonemize_streaming(self, text: str, lookahead: int, normalized: bool = False) -> str:
        """R2 output when the text arrives word by word and each word may see `lookahead` more."""
        g = text if normalized else normalize(text)
        return decode(g, *self._tag_chars(g, lookahead))

    def tag_words(self, g: str, lookahead: int | None = None, eos: bool = True) -> list[tuple[str, bool]]:
        """(bare R2, ezafe) for every space-separated token of normalized text `g`.

        `eos=False` tags a stream prefix (more text may follow); only meaningful offline.
        """
        labels, ezafe = self._tag_chars(g, lookahead, eos)
        out, start = [], 0
        for j, c in enumerate(g + " "):
            if c == " ":
                out.append(("".join(labels[start:j]), any(ezafe[start:j])))
                start = j + 1
        return out

    @torch.inference_mode()
    def analyze(self, g: str) -> dict[str, float]:
        """Cheap monitoring features for one normalized sentence."""
        ids = torch.tensor([self.vocab.encode(g, eos=True)], device=self.device)
        lab_logits, ez_logits = self.model(ids)
        conf = lab_logits[0, : len(g)].float().softmax(-1).max(-1).values
        letters = [i for i, c in enumerate(g) if c not in " \u200c"]
        n_words = max(len(g.split()), 1)
        ez = ez_logits[0, : len(g)].argmax(-1).bool().tolist()
        return {
            "n_chars": float(len(g)),
            "unknown_char_ratio": sum(c not in self.vocab.char2id for c in g) / max(len(g), 1),
            "ezafe_rate": sum(ez[i] for i in letters) / n_words,
            "mean_confidence": float(conf[letters].mean()) if letters else 1.0,
            "min_confidence": float(conf[letters].min()) if letters else 1.0,
        }
