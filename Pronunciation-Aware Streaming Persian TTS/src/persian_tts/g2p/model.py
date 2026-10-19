"""Character-level transformer tagger for Persian vowel / ezafe restoration.

For each input character the model predicts
  * a phoneme *label* (what this letter emits, incl. any hidden short vowel), and
  * an *ezafe* bit (does the word end here with an ezafe).

Decoding is constrained: a character may only take labels that were observed
for it in training (e.g. "ب" -> {b, b/, be, bo, bb, ...}), so the model cannot
emit phonemes unrelated to the text — a cheap, hard guardrail.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

PAD, UNK, EOS = 0, 1, 2
SPECIALS = ["<pad>", "<unk>", "<eos>"]


@dataclass
class TaggerConfig:
    n_chars: int
    n_labels: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 320


class Vocab:
    """Character + label vocabularies and the per-character allowed-label mask."""

    def __init__(self, chars: list[str], labels: list[str], allowed: dict[str, list[str]]):
        self.chars = SPECIALS + chars
        self.labels = labels
        self.char2id = {c: i for i, c in enumerate(self.chars)}
        self.label2id = {l: i for i, l in enumerate(labels)}
        self.allowed = allowed
        mask = torch.zeros(len(self.chars), len(labels), dtype=torch.bool)
        for c, labs in allowed.items():
            if c in self.char2id:
                for l in labs:
                    if l in self.label2id:
                        mask[self.char2id[c], self.label2id[l]] = True
        mask[UNK, self.label2id[""]] = True
        mask[EOS, self.label2id[""]] = True
        mask[PAD, self.label2id[""]] = True
        self.mask = mask  # [n_chars, n_labels]

    def encode(self, text: str, eos: bool) -> list[int]:
        ids = [self.char2id.get(c, UNK) for c in text]
        return ids + [EOS] if eos else ids

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"chars": self.chars[len(SPECIALS):], "labels": self.labels,
                                    "allowed": self.allowed}, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        d = json.loads(Path(path).read_text())
        return cls(d["chars"], d["labels"], d["allowed"])


class G2PTagger(nn.Module):
    def __init__(self, cfg: TaggerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.n_chars, cfg.d_model, padding_idx=PAD)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.label_head = nn.Linear(cfg.d_model, cfg.n_labels)
        self.ezafe_head = nn.Linear(cfg.d_model, 2)
        self.register_buffer("allowed", torch.ones(cfg.n_chars, cfg.n_labels, dtype=torch.bool), persistent=True)
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.normal_(self.embed.weight, std=1 / math.sqrt(cfg.d_model))

    def set_allowed(self, mask: torch.Tensor) -> None:
        self.allowed.copy_(mask)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ids [B, T] -> (label logits [B, T, L] with disallowed labels masked, ezafe logits [B, T, 2])."""
        T = ids.shape[1]
        pos = torch.arange(T, device=ids.device)
        h = self.embed(ids) + self.pos(pos)[None]
        h = self.encoder(h, src_key_padding_mask=ids.eq(PAD))
        h = self.norm(h)
        label_logits = self.label_head(h)
        label_logits = label_logits.masked_fill(~self.allowed[ids], torch.finfo(label_logits.dtype).min)
        return label_logits, self.ezafe_head(h)

    def save(self, path: Path) -> None:
        torch.save({"cfg": asdict(self.cfg), "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, map_location: str = "cpu") -> "G2PTagger":
        ck = torch.load(path, map_location=map_location)
        m = cls(TaggerConfig(**ck["cfg"]))
        m.load_state_dict(ck["state"])
        return m.eval()
