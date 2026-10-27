"""Supervised fine-tuning (LoRA on a 4-bit Qwen3-VL) on the sensor-derived answers.

The loss is the mean token cross-entropy over the answer (JSON + <|im_end|>);
prompt and image tokens are context only. Writes to --out:

    log.jsonl          step, loss, lr, grad norm, seconds per step
    adapter/           final LoRA adapter (checkpoint-<step>/ along the way)
    trainer_state.pt   step + optimizer state of the latest checkpoint
    val_metrics.json   reward on the val split after training

    python scripts/03_sft.py --out runs/sft_2b
    python scripts/03_sft.py --out runs/sft_2b --resume    # continue after an interruption
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from sensor_vlm.evaluate import evaluate
from sensor_vlm.model import load_split
from sensor_vlm.train import JsonlLog, answer_text, collate, completion_logprobs, cosine_lr, encode_prompt, trainable_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--tasks", type=Path, default=Path("data/tasks"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2, help="micro-batch")
    ap.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer step")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=None, help="val samples for the final eval (default all)")
    ap.add_argument("--fp16", action="store_true", help="fp16 vision + 4-bit compute with loss scaling (T4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in --out")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=1))
    log = JsonlLog(args.out / "log.jsonl")

    model, processor = trainable_model(args.model, lora_r=args.lora_r, fp16=args.fp16)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    train = load_split(args.tasks, "train")
    order = []
    rng = random.Random(args.seed)
    while len(order) < args.epochs * len(train):
        epoch = list(range(len(train)))
        rng.shuffle(epoch)
        order += epoch
    order = order[: int(args.epochs * len(train))]
    per_step = args.batch * args.accum
    total = len(order) // per_step if args.max_steps is None else min(args.max_steps, len(order) // per_step)
    start = 0
    if args.resume and (args.out / "trainer_state.pt").exists():
        state = torch.load(args.out / "trainer_state.pt", weights_only=False)
        start = state["step"]
        set_peft_model_state_dict(model, load_file(args.out / f"checkpoint-{start}" / "adapter_model.safetensors"))
        opt.load_state_dict(state["opt"])
        log.truncate_after(start)
    print(f"{len(train)} train samples, {total} steps of {per_step}, starting at {start}", flush=True)

    model.train()
    for step in range(start, total):
        t0 = time.perf_counter()
        lr = cosine_lr(step, total, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_sum, tokens = 0.0, 0
        for m in range(args.accum):
            idx = order[(step * args.accum + m) * args.batch:(step * args.accum + m + 1) * args.batch]
            samples = [train[i] for i in idx]
            prompts = [encode_prompt(processor, s, args.tasks) for s in samples]
            batch, mask = collate(processor, prompts, [answer_text(s) for s in samples])
            mask = mask.to(0)
            logp = completion_logprobs(model, batch, mask.shape[1])
            nll = -(logp * mask).sum()
            # normalize by the tokens of the whole optimizer step, estimated per micro-batch
            scaler.scale(nll / (mask.sum() * args.accum)).backward()
            loss_sum += nll.item()
            tokens += int(mask.sum().item())
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        log(step=step + 1, loss=loss_sum / tokens, lr=lr, grad_norm=grad_norm, sec=time.perf_counter() - t0)
        if (step + 1) % 10 == 0 or step == 0:
            print(f"step {step + 1}/{total} loss {loss_sum / tokens:.4f} lr {lr:.2e} gnorm {grad_norm:.2f} "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
        if (step + 1) % args.save_every == 0:
            model.save_pretrained(args.out / f"checkpoint-{step + 1}")
            torch.save({"step": step + 1, "opt": opt.state_dict()}, args.out / "trainer_state.pt")

    model.save_pretrained(args.out / "adapter")
    val = load_split(args.tasks, "val", args.eval_n)
    metrics = evaluate(model, processor, val, args.tasks, out_dir=args.out, split="val")
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
