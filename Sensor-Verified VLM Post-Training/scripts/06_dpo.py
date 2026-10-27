"""DPO from an SFT adapter, on preference pairs ranked by the verifiable reward.

Phase 1 (skipped if <out>/pairs.jsonl exists): for --pair-prompts training
prompts, sample --group completions from the SFT policy, score them, and keep
(best, worst) when their total rewards differ by at least --min-gap. The
reference log-probs (the SFT policy itself) are computed here, so training
needs no second model.

Phase 2: DPO on those pairs, starting from the same SFT adapter. Comparing
with GRPO at a similar sample budget asks whether offline preferences built
from the reward recover what online RL gets from it.

    python scripts/06_dpo.py --adapter runs/sft_2b/adapter --out runs/dpo_2b
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from sensor_vlm.evaluate import evaluate
from sensor_vlm.model import load_split
from sensor_vlm.rl import dpo_loss, rollout
from sensor_vlm.train import JsonlLog, collate, completion_logprobs, cosine_lr, encode_prompt, trainable_model


def sequence_logprobs(model, processor, prompt: dict, completions: list[str]) -> torch.Tensor:
    batch, mask = collate(processor, [prompt] * len(completions), completions)
    mask = mask.to(0)
    return (completion_logprobs(model, batch, mask.shape[1]) * mask).sum(1)


def make_pairs(model, processor, train: list[dict], args) -> list[dict]:
    order = list(range(len(train)))
    random.Random(args.seed + 1).shuffle(order)
    chosen = [train[i] for i in order[: args.pair_prompts]]
    pairs, t0 = [], time.perf_counter()
    for k in range(0, len(chosen), args.gen_batch):
        samples = chosen[k:k + args.gen_batch]
        groups, scores = rollout(model, processor, samples, args.tasks, args.group, args.temperature, args.max_new)
        for s, g, sc in zip(samples, groups, scores):
            totals = [x["total"] for x in sc]
            best, worst = int(np.argmax(totals)), int(np.argmin(totals))
            if totals[best] - totals[worst] < args.min_gap:
                continue
            with torch.no_grad():
                model.eval()
                ref = sequence_logprobs(model, processor, encode_prompt(processor, s, args.tasks), [g[best], g[worst]])
            pairs.append({"id": s["id"], "chosen": g[best], "rejected": g[worst],
                          "reward_chosen": totals[best], "reward_rejected": totals[worst],
                          "ref_chosen": ref[0].item(), "ref_rejected": ref[1].item()})
        print(f"pairs {len(pairs)} from {k + len(samples)} prompts ({time.perf_counter() - t0:.0f}s)", flush=True)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--adapter", required=True, help="SFT adapter: pair sampler, reference and starting point")
    ap.add_argument("--tasks", type=Path, default=Path("data/tasks"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pair-prompts", type=int, default=600)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--gen-batch", type=int, default=4, help="prompts per generate call")
    ap.add_argument("--min-gap", type=float, default=0.25)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8, help="pairs per optimizer step")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--eval-n", type=int, default=None)
    ap.add_argument("--fp16", action="store_true", help="fp16 vision + 4-bit compute with loss scaling (T4)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=1))
    log = JsonlLog(args.out / "log.jsonl")

    model, processor = trainable_model(args.model, args.adapter, fp16=args.fp16)
    train = load_split(args.tasks, "train")
    by_id = {s["id"]: s for s in train}

    pairs_file = args.out / "pairs.jsonl"
    if pairs_file.exists():
        pairs = [json.loads(line) for line in open(pairs_file)]
    else:
        pairs = make_pairs(model, processor, train, args)
        with open(pairs_file, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
    print(f"{len(pairs)} pairs, mean reward chosen {np.mean([p['reward_chosen'] for p in pairs]):.3f} "
          f"vs rejected {np.mean([p['reward_rejected'] for p in pairs]):.3f}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    rng = random.Random(args.seed)
    order = [i for _ in range(args.epochs) for i in rng.sample(range(len(pairs)), len(pairs))]
    total = len(order) // args.accum
    model.train()
    for step in range(total):
        t0 = time.perf_counter()
        lr = cosine_lr(step, total, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        stats = []
        for i in order[step * args.accum:(step + 1) * args.accum]:
            p = pairs[i]
            prompt = encode_prompt(processor, by_id[p["id"]], args.tasks)
            pol = sequence_logprobs(model, processor, prompt, [p["chosen"], p["rejected"]])
            ref = torch.tensor([p["ref_chosen"], p["ref_rejected"]], device=pol.device)
            loss, st = dpo_loss(pol[:1], pol[1:], ref[:1], ref[1:], args.beta)
            scaler.scale(loss / args.accum).backward()
            stats.append({**st, "loss": loss.item()})
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        row = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
        log(step=step + 1, lr=lr, grad_norm=grad_norm, sec=time.perf_counter() - t0, **row)
        if (step + 1) % 10 == 0 or step == 0:
            print(f"step {step + 1}/{total} loss {row['loss']:.4f} margin {row['margin']:.3f} "
                  f"acc {row['accuracy']:.2f} {time.perf_counter() - t0:.0f}s", flush=True)

    model.save_pretrained(args.out / "adapter")
    val = load_split(args.tasks, "val", args.eval_n)
    metrics = evaluate(model, processor, val, args.tasks, out_dir=args.out, split="val")
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
