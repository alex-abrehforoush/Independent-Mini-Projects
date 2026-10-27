"""Evaluate a model (optionally with a LoRA adapter) on a task split.

Writes per-sample predictions and rewards to <out>/<split>_predictions.jsonl
and the mean of each reward component to <out>/<split>_metrics.json.

    python scripts/02_evaluate.py --out runs/zeroshot_2b --split val
    python scripts/02_evaluate.py --adapter runs/sft_2b/adapter --out runs/sft_2b
    python scripts/02_evaluate.py --adapter runs/sft_2b/adapter --out runs/sft_2b --split train \
        --ids-file kaggle/train_probe_ids.txt --name train_probe
"""

import argparse
import json
from pathlib import Path

from sensor_vlm.evaluate import evaluate
from sensor_vlm.model import load_model, load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tasks", type=Path, default=Path("data/tasks"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids-file", type=Path, default=None, help="evaluate only these sample ids (one per line)")
    ap.add_argument("--name", default=None, help="output file prefix (default: the split)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--fp16", action="store_true", help="fp16 vision + 4-bit compute (T4)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model, processor = load_model(args.model, args.adapter, vision_fp32=not args.fp16, compute_fp32=not args.fp16)
    samples = load_split(args.tasks, args.split)
    if args.ids_file:
        keep = {line.strip() for line in open(args.ids_file) if line.strip() and not line.startswith("#")}
        samples = [s for s in samples if s["id"] in keep]
    samples = samples[:args.limit] if args.limit else samples
    name = args.name or args.split
    metrics = evaluate(model, processor, samples, args.tasks, args.batch, args.out, name)
    metrics.update(model=args.model, adapter=args.adapter)
    (args.out / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
