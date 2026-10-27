"""Kaggle runner (GPU T4 x2): the 4B/8B runs, with the same scripts as the local 2B runs.

Setup: attach the private datasets made by scripts/05_kaggle_bundle.py
(sensor-vlm-code, sensor-vlm-tasks), Accelerator = GPU T4 x2, Internet = on, then
"Save Version -> Save & Run All". Outputs land in /kaggle/working/runs.

PLANS maps a plan name to (gpu, script arguments) stages. Stages on the same
GPU run in order; the two GPUs run in parallel (GPU quota is billed per
session, so both should stay busy). To continue an interrupted run, attach the
previous version's output and set RESUME_FROM to its runs/ folder: its runs
are copied in and --resume picks up from the last checkpoint.

A stage can name an earlier run's adapter as @<run> (e.g. @sft_8b); it resolves
to <run>/adapter inside an attached dataset (sensor-vlm-adapters).
"""

import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

PLAN = "probe"
RUNS = "/kaggle/working/runs"
RESUME_FROM = None  # e.g. "/kaggle/input/sensor-vlm-t4-runner/runs"
M4, M8 = "Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"


def smoke(gpu: int, model: str, tag: str) -> list[tuple[int, str]]:
    """Timing and numerics check: a few SFT and GRPO steps plus a small eval."""
    return [
        (gpu, f"scripts/03_sft.py --model {model} --out {RUNS}/smoke_sft_{tag} --fp16 --batch 4 --accum 1 "
              f"--max-steps 3 --warmup 1 --eval-n 8"),
        (gpu, f"scripts/04_grpo.py --model {model} --out {RUNS}/smoke_grpo_{tag} --fp16 --steps 2 --prompts 2 "
              f"--group 4 --micro 4 --eval-n 4"),
    ]


def sft(gpu: int, model: str, tag: str) -> list[tuple[int, str]]:
    """One SFT epoch (effective batch 8, as the local 2B run), then the val eval."""
    return [(gpu, f"scripts/03_sft.py --model {model} --out {RUNS}/sft_{tag} --fp16 --batch 4 --accum 2 "
                  f"--save-every 100 --resume")]


def zeroshot(gpu: int, model: str, tag: str, split: str = "val") -> list[tuple[int, str]]:
    return [(gpu, f"scripts/02_evaluate.py --model {model} --out {RUNS}/zeroshot_{tag} --split {split} --fp16 --batch 8")]


def grpo(gpu: int, model: str, tag: str, steps: int = 100) -> list[tuple[int, str]]:
    return [(gpu, f"scripts/04_grpo.py --model {model} --adapter @sft_{tag} --out {RUNS}/grpo_{tag} --fp16 "
                  f"--steps {steps} --prompts 4 --group 4 --micro 4 --save-every 25 --resume")]


def dpo(gpu: int, model: str, tag: str) -> list[tuple[int, str]]:
    return [(gpu, f"scripts/06_dpo.py --model {model} --adapter @sft_{tag} --out {RUNS}/dpo_{tag} --fp16 "
                  f"--pair-prompts 500 --gen-batch 4")]


def probe(gpu: int, model: str, run: str) -> list[tuple[int, str]]:
    """Greedy eval of an earlier run on train-participant prompts that no GRPO/DPO run used."""
    return [(gpu, f"scripts/02_evaluate.py --model {model} --adapter @{run} --out {RUNS}/{run} --split train "
                  f"--ids-file kaggle/train_probe_ids.txt --name train_probe --fp16 --batch 8")]


PLANS = {
    # 8.6 min on T4 x2: 4B SFT 1.3 s/sample, 8B 1.8 s/sample; GRPO ~11 / ~15 s per prompt (G=4)
    "smoke": smoke(0, M4, "4b") + smoke(1, M8, "8b"),
    # ~2.6 h: GPU0 8B SFT + eval; GPU1 4B SFT + eval, then both zero-shot baselines
    "sft": sft(0, M8, "8b") + sft(1, M4, "4b") + zeroshot(1, M4, "4b") + zeroshot(1, M8, "8b"),
    # ~2.6 h: GPU0 8B GRPO + eval; GPU1 4B GRPO + eval, then 4B DPO (pairs, training, eval)
    "grpo": grpo(0, M8, "8b") + grpo(1, M4, "4b") + dpo(1, M4, "4b"),
    # ~20 min: did GRPO improve greedy answers on the training participants, or only sharpen sampling?
    "probe": probe(0, M4, "sft_4b") + probe(1, M4, "grpo_4b") + probe(0, M8, "sft_8b") + probe(1, M8, "grpo_8b"),
}

WORK = Path("/kaggle/working/sensor_vlm")


def find_input(marker: str) -> Path:
    """Folder under /kaggle/input holding `marker` (mount layouts differ between Kaggle versions)."""
    hits = sorted(Path("/kaggle/input").rglob(marker), key=lambda p: len(p.parts))
    if not hits:
        raise FileNotFoundError(f"{marker} not found under /kaggle/input")
    return hits[0].parent


def resolve_adapters(args: str) -> str:
    """Replace @<run> with the path of <run>/adapter in an attached dataset."""
    out = []
    for token in args.split():
        if token.startswith("@"):
            hits = [p.parent for p in Path("/kaggle/input").rglob("adapter_config.json")
                    if p.parent.name == "adapter" and p.parent.parent.name == token[1:]]
            if not hits:
                raise FileNotFoundError(f"adapter for {token[1:]} not found under /kaggle/input")
            token = str(hits[0])
        out.append(token)
    return " ".join(out)


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    code, tasks = find_input("pyproject.toml"), find_input("train.jsonl")
    print(f"code: {code}\ntasks: {tasks}", flush=True)
    sh("pip install -q transformers==5.20.2 peft==0.21.2 bitsandbytes==0.50.2 accelerate==1.15.0")
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(code, WORK, ignore=shutil.ignore_patterns("dataset-metadata.json"))
    sh(f"pip install -q --no-deps -e {WORK}")
    (WORK / "data").mkdir(exist_ok=True)
    (WORK / "data" / "tasks").symlink_to(tasks)
    if RESUME_FROM:
        shutil.copytree(RESUME_FROM, RUNS, dirs_exist_ok=True)
    Path(RUNS).mkdir(parents=True, exist_ok=True)

    per_gpu = defaultdict(list)
    for gpu, args in PLANS[PLAN]:
        per_gpu[gpu].append(f"{sys.executable} {resolve_adapters(args)}")
    procs = []
    for gpu, cmds in per_gpu.items():
        log = open(f"{RUNS}/gpu{gpu}.log", "a")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_ALLOC_CONF="expandable_segments:True")
        procs.append(subprocess.Popen(" && ".join(cmds), shell=True, cwd=WORK, env=env, stdout=log, stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in procs):
        time.sleep(120)
        for gpu in per_gpu:
            tail = Path(f"{RUNS}/gpu{gpu}.log").read_text()[-600:].replace("\r", "\n").splitlines()
            print(f"[gpu{gpu}] {tail[-1] if tail else ''}", flush=True)
    print("exit codes:", [p.returncode for p in procs], flush=True)
    for gpu in per_gpu:
        print(f"===== gpu{gpu}.log (tail) =====\n" + Path(f"{RUNS}/gpu{gpu}.log").read_text()[-4000:], flush=True)


if __name__ == "__main__":
    main()
