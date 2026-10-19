"""Training data for the tagger, with random-prefix truncation for streaming.

A sample is either the complete sentence (followed by <eos>) or a prefix cut at
a random word boundary (no <eos>). In a prefix, the last word has 0 words of
lookahead, the one before it 1, and so on, so a single model learns every
lookahead regime; the <eos> token tells it whether the right context is final.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from persian_tts.g2p.model import PAD, Vocab

IGNORE = -100


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


class TaggingDataset(Dataset):
    def __init__(self, items: list[dict], vocab: Vocab, p_complete: float = 0.5,
                 max_len: int = 300, train: bool = True, seed: int = 0):
        self.items = [it for it in items if len(it["g"]) < max_len]
        self.vocab, self.p_complete, self.train = vocab, p_complete, train
        random.seed(seed)  # DataLoader re-seeds `random` per worker

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[list[int], list[int], list[int]]:
        it = self.items[i]
        g, labels, ezafe = it["g"], it["labels"], it["ezafe"]
        complete = (not self.train) or random.random() < self.p_complete
        if not complete:
            spaces = [j for j, c in enumerate(g) if c == " "]
            if spaces:
                cut = random.choice(spaces)  # keep words [0..cut), drop the rest
                g, labels, ezafe = g[:cut], labels[:cut], ezafe[:cut]
            else:
                complete = True
        ids = self.vocab.encode(g, eos=complete)
        lab = [self.vocab.label2id.get(l, IGNORE) for l in labels]
        ez = [int(e) if c not in " ‌" else IGNORE for c, e in zip(g, ezafe)]
        if complete:
            lab.append(IGNORE)
            ez.append(IGNORE)
        return ids, lab, ez


def collate(batch: list[tuple[list[int], list[int], list[int]]]):
    T = max(len(b[0]) for b in batch)
    ids = torch.full((len(batch), T), PAD, dtype=torch.long)
    lab = torch.full((len(batch), T), IGNORE, dtype=torch.long)
    ez = torch.full((len(batch), T), IGNORE, dtype=torch.long)
    for i, (a, b, c) in enumerate(batch):
        ids[i, : len(a)] = torch.tensor(a)
        lab[i, : len(b)] = torch.tensor(b)
        ez[i, : len(c)] = torch.tensor(c)
    return ids, lab, ez


def build_vocab(train: list[dict], labels: list[str], min_count: int = 1) -> Vocab:
    """Characters seen in training + the labels each character was aligned to."""
    from collections import Counter, defaultdict

    seen: dict[str, Counter] = defaultdict(Counter)
    for it in train:
        for c, l in zip(it["g"], it["labels"]):
            seen[c][l] += 1
    allowed = {c: [l for l, n in cnt.items() if n >= min_count or l == ""] for c, cnt in seen.items()}
    chars = sorted(seen)
    return Vocab(chars, labels, allowed)
