"""Train the streaming G2P tagger.

    python scripts/02_train_g2p.py --out runs/g2p_base
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from persian_tts.g2p.data import IGNORE, TaggingDataset, build_vocab, collate, read_jsonl
from persian_tts.g2p.infer import Phonemizer
from persian_tts.g2p.metrics import G2PScores
from persian_tts.g2p.model import G2PTagger, TaggerConfig


class BucketSampler(Sampler):
    """Shuffled batches of similar-length sentences, capped by a padded-token budget.

    A fixed sentence count per batch OOMs a 4 GB GPU when a bucket of ~300-char
    sentences comes up; bounding (batch size x max length) bounds memory.
    """

    def __init__(self, lengths: list[int], max_tokens: int, max_batch: int, seed: int = 0):
        self.lengths, self.max_tokens, self.max_batch = lengths, max_tokens, max_batch
        self.rng = random.Random(seed)
        self._n = len(self._make_batches())

    def _make_batches(self) -> list[list[int]]:
        idx = list(range(len(self.lengths)))
        self.rng.shuffle(idx)
        pool = self.max_batch * 50
        batches = []
        for i in range(0, len(idx), pool):
            chunk = sorted(idx[i: i + pool], key=lambda j: self.lengths[j])
            cur, cur_max = [], 0
            for j in chunk:
                m = max(cur_max, self.lengths[j] + 1)
                if cur and (m * (len(cur) + 1) > self.max_tokens or len(cur) >= self.max_batch):
                    batches.append(cur)
                    cur, m = [], self.lengths[j] + 1
                cur.append(j)
                cur_max = m
            if cur:
                batches.append(cur)
        self.rng.shuffle(batches)
        return batches

    def __iter__(self):
        yield from self._make_batches()

    def __len__(self) -> int:
        return self._n


def evaluate(phonemizer: Phonemizer, items: list[dict], lookahead: int | None = None) -> dict:
    scores = G2PScores()
    if lookahead is None:
        hyps = []
        for i in range(0, len(items), 256):
            hyps += phonemizer.phonemize_batch([it["g"] for it in items[i: i + 256]])
    else:
        hyps = [phonemizer.phonemize_streaming(it["g"], lookahead, normalized=True) for it in items]
    for it, h in zip(items, hyps):
        scores.add(h, it["ref"], it["g"], it.get("hom"), it.get("hom_ref"))
    return scores.summary()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/g2p")
    ap.add_argument("--out", default="runs/g2p_base")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=128, help="max sentences per batch")
    ap.add_argument("--max-tokens", type=int, default=8192, help="max padded characters per batch")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--p-complete", type=float, default=0.5,
                    help="fraction of samples that are full sentences (rest are random prefixes)")
    ap.add_argument("--ezafe-weight", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N training sentences")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast (4x *slower* on a GTX 1650 / TU117 - benchmark before enabling)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_items = read_jsonl(Path(args.data) / "train.jsonl")
    if args.limit:
        train_items = train_items[: args.limit]
    dev_items = read_jsonl(Path(args.data) / "dev.jsonl")
    sb_items = read_jsonl(Path(args.data) / "sentencebench.jsonl")
    labels = json.loads((Path(args.data) / "labels.json").read_text())["labels"]
    vocab = build_vocab(train_items, labels)
    vocab.save(out / "vocab.json")

    cfg = TaggerConfig(n_chars=len(vocab.chars), n_labels=len(labels), d_model=args.d_model,
                       n_layers=args.layers, n_heads=args.d_model // 64, d_ff=4 * args.d_model)
    model = G2PTagger(cfg)
    model.set_allowed(vocab.mask)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M  labels={len(labels)} chars={len(vocab.chars)}")

    ds = TaggingDataset(train_items, vocab, p_complete=args.p_complete, seed=args.seed)
    loader = DataLoader(ds, batch_sampler=BucketSampler([len(it["g"]) for it in ds.items], args.max_tokens, args.batch_size, args.seed),
                        collate_fn=collate, num_workers=4, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    total = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0))))
    use_amp = args.amp and device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    log = open(out / "train_log.jsonl", "a")
    best, step, t0 = float("inf"), 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        for ids, lab, ez in loader:
            ids, lab, ez = ids.to(device), lab.to(device), ez.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                lab_logits, ez_logits = model(ids)
            loss_lab = F.cross_entropy(lab_logits.float().flatten(0, 1), lab.flatten(), ignore_index=IGNORE)
            loss_ez = F.cross_entropy(ez_logits.float().flatten(0, 1), ez.flatten(), ignore_index=IGNORE)
            loss = loss_lab + args.ezafe_weight * loss_ez
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 200 == 0:
                rec = {"step": step, "epoch": epoch, "loss_label": round(loss_lab.item(), 4),
                       "loss_ezafe": round(loss_ez.item(), 4), "lr": sched.get_last_lr()[0],
                       "elapsed_min": round((time.time() - t0) / 60, 1)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()

        model.eval()
        ph = Phonemizer(model, vocab, device)
        dev = evaluate(ph, dev_items)
        dev_k0 = evaluate(ph, dev_items[:1000], lookahead=0)
        sb = evaluate(ph, sb_items)
        rec = {"epoch": epoch, "step": step, "dev": dev, "dev_stream_k0": dev_k0, "sentencebench": sb}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if dev["PER"] < best:  # model selection on dev only; SentenceBench is reported, never selected on
            best = dev["PER"]
            model.save(out / "model.pt")
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    print("best dev PER", best)


if __name__ == "__main__":
    main()
