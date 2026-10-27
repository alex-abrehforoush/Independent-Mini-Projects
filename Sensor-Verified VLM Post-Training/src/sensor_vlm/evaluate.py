"""Scoring a model on a task split with the verifiable rewards."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sensor_vlm import rewards
from sensor_vlm.model import generate

KEYS = ["total", "format", "hands", "object", "point"]


def evaluate(model, processor, samples: list[dict], tasks_dir: Path, batch: int = 4,
             out_dir: Path | None = None, split: str = "val") -> dict:
    """Greedy-decode every sample, score it; optionally write predictions and metrics."""
    was_training = model.training
    model.eval()
    rows, t0 = [], time.perf_counter()
    for i in tqdm(range(0, len(samples), batch), unit="batch", desc=f"eval {split}", leave=False):
        chunk = samples[i:i + batch]
        for s, completion in zip(chunk, generate(model, processor, chunk, tasks_dir)):
            rows.append({"id": s["id"], "completion": completion, "reward": rewards.score(completion, s)})
    metrics = {k: float(np.nanmean([r["reward"][k] for r in rows])) for k in KEYS}
    metrics.update(n=len(rows), seconds_per_sample=(time.perf_counter() - t0) / len(rows))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{split}_predictions.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        (out_dir / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=1))
    model.train(was_training)
    return metrics


def constant_baselines(samples: list[dict]) -> dict[str, float]:
    """Mean reward of the best simple constant answers."""
    objs = Counter(s["answer"]["object"] for s in samples if s["answer"]["object"] != "none")
    common = objs.most_common(1)[0][0] if objs else "none"
    cands = {
        "all hands busy, most common object, center": {"left": [1] * 4, "right": [1] * 4, "object": common, "point": [500, 500]},
        "right hand only, most common object": {"left": [0] * 4, "right": [1] * 4, "object": common, "point": [750, 750]},
        "nothing held": {"left": [0] * 4, "right": [0] * 4, "object": "none", "point": None},
    }
    return {k: float(np.mean([rewards.score(json.dumps(a), s)["total"] for s in samples])) for k, a in cands.items()}
