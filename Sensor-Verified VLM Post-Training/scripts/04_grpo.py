"""GRPO on the verifiable sensor-derived rewards, starting from an SFT adapter.

Each step samples --group completions for each of --prompts training prompts,
scores them with `sensor_vlm.rewards`, and takes one policy-gradient step on the
group-relative advantages (one update per rollout batch, so the PPO ratio is
1 at the update and clipping only matters with --updates > 1). Groups whose
completions all score the same carry no signal and are skipped. With
--beta > 0 a k3 KL penalty to the base model (adapter disabled) is added.

Writes to --out: log.jsonl (rewards, informative-group fraction, completion
length, clip fraction, KL, lr, timing), samples.jsonl (a few rollouts per step),
checkpoint-<step>/ + trainer_state.pt (resumable), adapter/, val metrics.

    python scripts/04_grpo.py --adapter runs/sft_2b/adapter --out runs/grpo_2b
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from sensor_vlm.evaluate import evaluate
from sensor_vlm.model import load_split
from sensor_vlm.rl import group_advantages, policy_loss, rollout
from sensor_vlm.train import JsonlLog, collate, completion_logprobs, cosine_lr, encode_prompt, trainable_model

COMPONENTS = ["total", "format", "hands", "object", "point"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--adapter", default=None, help="start from this LoRA (SFT); a fresh LoRA if omitted")
    ap.add_argument("--tasks", type=Path, default=Path("data/tasks"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prompts", type=int, default=4, help="prompts per step")
    ap.add_argument("--group", type=int, default=4, help="completions per prompt")
    ap.add_argument("--micro", type=int, default=2, help="completions per forward/backward")
    ap.add_argument("--updates", type=int, default=1, help="gradient steps per rollout batch")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--adv-scale", choices=["std", "none"], default="std")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=0, help="val eval every N steps (0: only at the end)")
    ap.add_argument("--eval-n", type=int, default=None)
    ap.add_argument("--fp16", action="store_true", help="fp16 vision + 4-bit compute with loss scaling (T4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=1))
    log, sample_log = JsonlLog(args.out / "log.jsonl"), JsonlLog(args.out / "samples.jsonl")

    model, processor = trainable_model(args.model, args.adapter, fp16=args.fp16)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    train = load_split(args.tasks, "train")
    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)
    val = load_split(args.tasks, "val", args.eval_n)

    start = 0
    if args.resume and (args.out / "trainer_state.pt").exists():
        state = torch.load(args.out / "trainer_state.pt", weights_only=False)
        start = state["step"]
        set_peft_model_state_dict(model, load_file(args.out / f"checkpoint-{start}" / "adapter_model.safetensors"))
        opt.load_state_dict(state["opt"])
        log.truncate_after(start)

    for step in range(start, args.steps):
        t0 = time.perf_counter()
        lr = cosine_lr(step, args.steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = [order[(step * args.prompts + i) % len(order)] for i in range(args.prompts)]
        samples = [train[i] for i in idx]

        groups, scores = rollout(model, processor, samples, args.tasks, args.group, args.temperature, args.max_new)
        comps = [c for g in groups for c in g]
        r = torch.tensor([[sc["total"] for sc in g] for g in scores])
        adv, informative = group_advantages(r, args.adv_scale)
        t_gen = time.perf_counter() - t0

        # policy update(s) on the informative groups
        model.train()
        work = [(i, j) for i in range(args.prompts) if informative[i] for j in range(args.group)]
        prompts = {i: encode_prompt(processor, samples[i], args.tasks) for i in {i for i, _ in work}}
        old = {}
        stats = {"clip_frac": [], "kl": []}
        for u in range(args.updates):
            for k in range(0, len(work), args.micro):
                chunk = work[k:k + args.micro]
                batch, mask = collate(processor, [prompts[i] for i, _ in chunk], [groups[i][j] for i, j in chunk])
                mask = mask.to(0)
                logp = completion_logprobs(model, batch, mask.shape[1])
                if u == 0:
                    old[k] = logp.detach()
                ref = None
                if args.beta > 0:
                    with torch.no_grad(), model.disable_adapter():
                        ref = completion_logprobs(model, batch, mask.shape[1])
                a = torch.stack([adv[i, j] for i, j in chunk]).to(0)
                loss, st = policy_loss(logp, old[k], a, mask, args.clip, ref, args.beta)
                scaler.scale(loss / len(work)).backward()
                for key in stats:
                    if key in st:
                        stats[key].append(st[key])
            if work:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
            opt.zero_grad(set_to_none=True)

        tok = processor.tokenizer
        row = {f"reward_{c}": float(np.nanmean([sc[c] for g in scores for sc in g])) for c in COMPONENTS}
        row.update(
            step=step + 1, lr=lr, reward_std=float(r.std(1, unbiased=False).mean()),
            informative=float(informative.float().mean()), n_updated=len(work),
            completion_tokens=float(np.mean([len(tok(c).input_ids) for c in comps])),
            clip_frac=float(np.mean(stats["clip_frac"])) if stats["clip_frac"] else 0.0,
            kl=float(np.mean(stats["kl"])) if stats["kl"] else 0.0,
            sec=time.perf_counter() - t0, sec_rollout=t_gen,
        )
        log(**row)
        sample_log(step=step + 1, id=samples[0]["id"], target=samples[0]["answer"],
                   completions=groups[0], rewards=[sc["total"] for sc in scores[0]])
        print(f"step {step + 1}/{args.steps} reward {row['reward_total']:.3f} (hands {row['reward_hands']:.2f} "
              f"obj {row['reward_object']:.2f} pt {row['reward_point']:.2f}) informative {row['informative']:.2f} "
              f"len {row['completion_tokens']:.0f} {row['sec']:.0f}s", flush=True)

        if (step + 1) % args.save_every == 0:
            model.save_pretrained(args.out / f"checkpoint-{step + 1}")
            torch.save({"step": step + 1, "opt": opt.state_dict()}, args.out / "trainer_state.pt")
        if args.eval_every and (step + 1) % args.eval_every == 0 and step + 1 < args.steps:
            m = evaluate(model, processor, val, args.tasks)
            log(step=step + 1, **{f"val_{k}": v for k, v in m.items()})
            print(f"  val {json.dumps(m)}", flush=True)

    model.save_pretrained(args.out / "adapter")
    metrics = evaluate(model, processor, val, args.tasks, out_dir=args.out, split="val")
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
