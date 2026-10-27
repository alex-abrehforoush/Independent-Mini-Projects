"""Package code and task data as private Kaggle datasets, plus the T4 runner kernel.

    python scripts/05_kaggle_bundle.py --user <kaggle-username>

creates dist/kaggle/{sensor-vlm-tasks, sensor-vlm-code, kernel}. Upload with the
Kaggle CLI (or the web UI, uploading the same folders):

    kaggle datasets create -p dist/kaggle/sensor-vlm-tasks --dir-mode zip
    kaggle datasets create -p dist/kaggle/sensor-vlm-code --dir-mode zip
    kaggle kernels push -p dist/kaggle/kernel

After code changes: `kaggle datasets version -p dist/kaggle/sensor-vlm-code -m "..." --dir-mode zip`.
`--adapter-runs runs/x/sft_4b ...` also packages those runs' adapter/ folders
(plus their metrics) as sensor-vlm-adapters and attaches it to the kernel, for
stages that start from an earlier run (@sft_4b in kaggle/run_t4.py).
Task files are hard-linked, not copied. The data derives from HOT3D, whose
hand annotations are CC BY-NC-SA, so the datasets stay private and NC-SA.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def link_tree(src: Path, dst: Path) -> None:
    for path in src.rglob("*"):
        target = dst / path.relative_to(src)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="Kaggle username")
    ap.add_argument("--out", type=Path, default=ROOT / "dist/kaggle")
    ap.add_argument("--adapter-runs", type=Path, nargs="*", default=[])
    args = ap.parse_args()

    tasks = args.out / "sensor-vlm-tasks"
    link_tree(ROOT / "data/tasks", tasks / "tasks")
    (tasks / "dataset-metadata.json").write_text(json.dumps({
        "title": "sensor-vlm-tasks", "id": f"{args.user}/sensor-vlm-tasks", "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
    }, indent=1))

    code = args.out / "sensor-vlm-code"
    if code.exists():
        shutil.rmtree(code)
    ignore = shutil.ignore_patterns("__pycache__", "*.egg-info")
    for part in ("src", "scripts", "kaggle"):
        shutil.copytree(ROOT / part, code / part, ignore=ignore)
    shutil.copy(ROOT / "pyproject.toml", code)
    (code / "dataset-metadata.json").write_text(json.dumps({
        "title": "sensor-vlm-code", "id": f"{args.user}/sensor-vlm-code", "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
    }, indent=1))

    sources = [f"{args.user}/sensor-vlm-code", f"{args.user}/sensor-vlm-tasks"]
    if args.adapter_runs:
        adapters = args.out / "sensor-vlm-adapters"
        if adapters.exists():
            shutil.rmtree(adapters)
        for run in args.adapter_runs:
            shutil.copytree(run / "adapter", adapters / run.name / "adapter")
            for f in run.glob("*_metrics.json"):
                shutil.copy(f, adapters / run.name)
        (adapters / "dataset-metadata.json").write_text(json.dumps({
            "title": "sensor-vlm-adapters", "id": f"{args.user}/sensor-vlm-adapters",
            "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
        }, indent=1))
        sources.append(f"{args.user}/sensor-vlm-adapters")

    kernel = args.out / "kernel"
    kernel.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "kaggle/run_t4.py", kernel)
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.user}/sensor-vlm-t4-runner", "title": "sensor-vlm-t4-runner", "code_file": "run_t4.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": sources,
    }, indent=1))
    print(f"bundle in {args.out}")


if __name__ == "__main__":
    main()
