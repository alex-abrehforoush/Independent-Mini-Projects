"""Collect every run's metrics into a results table and figures for the README.

Runs are found by directory name, <method>_<size> (zeroshot / sft / grpo / dpo;
2b / 4b / 8b), anywhere under --runs (local runs and downloaded Kaggle
outputs alike); when a name appears twice the newest metrics win.

    docs/results_<split>.md               metrics table (the charts' table view), paired
                                          differences vs SFT with bootstrap CIs, and how often
                                          each model calls a hand busy
    docs/figures/<split>_total.png        total reward by method and model size
    docs/figures/<split>_components.png   hands / object / point, small multiples
    docs/figures/sft_loss.png             SFT training loss
    docs/figures/grpo_reward.png          mean rollout reward during GRPO

    python scripts/07_report.py --split val
    python scripts/07_report.py --split train_probe --base-split train   # the samples the runs predicted
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

from sensor_vlm import rewards  # noqa: E402
from sensor_vlm.evaluate import constant_baselines  # noqa: E402
from sensor_vlm.model import load_split  # noqa: E402

METHODS = ["zeroshot", "sft", "grpo", "dpo"]
METHOD_LABEL = {"zeroshot": "Zero-shot", "sft": "SFT", "grpo": "SFT + GRPO", "dpo": "SFT + DPO"}
SIZES = ["2b", "4b", "8b"]
# reference palette, categorical slots 1-3 (all pairs validated, light surface); color follows model size
COLOR = {"2b": "#2a78d6", "4b": "#eb6834", "8b": "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COMPONENTS = ["hands", "object", "point"]
RUN_NAME = re.compile(r"^(zeroshot|sft|grpo|dpo)_(2b|4b|8b)$")
DPI = 160
BAR_PX = 24  # max column thickness in the saved image


def find_runs(root: Path, split: str) -> dict[tuple[str, str], dict]:
    found = {}
    for f in sorted(root.rglob(f"{split}_metrics.json"), key=lambda p: p.stat().st_mtime):
        m = RUN_NAME.match(f.parent.name)
        if m:
            found[(m.group(1), m.group(2))] = {**json.loads(f.read_text()), "dir": f.parent}
    return found


def predictions(run: dict, split: str) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, open(run["dir"] / f"{split}_predictions.jsonl"))}


def paired_delta(a: dict, b: dict, key: str, n_boot: int = 5000) -> tuple[float, float, float]:
    """Mean per-sample difference a - b and its 95% bootstrap interval (same samples, so paired)."""
    d = np.array([a[i]["reward"][key] - b[i]["reward"][key] for i in a if i in b], float)
    d = d[~np.isnan(d)]
    boots = np.random.default_rng(0).choice(d, (n_boot, len(d))).mean(1)
    return float(d.mean()), *np.percentile(boots, [2.5, 97.5])


def busy_rate(answers: list[dict | None], side: str) -> float:
    """Fraction of frames in which `side` is said to hold something (well-formed answers only)."""
    return float(np.mean([np.mean(a[side]) for a in answers if a and rewards._bits(a.get(side))]))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def data_per_px(ax) -> tuple[float, float]:
    bbox = ax.get_window_extent()
    return np.diff(ax.get_xlim())[0] / bbox.width, np.diff(ax.get_ylim())[0] / bbox.height


def column(ax, x, width, height, color, radius_px=4):
    """One path: square at the baseline, corners rounded by radius_px at the data end."""
    xpp, ypp = data_per_px(ax)
    rx, ry = min(radius_px * xpp, width / 2), min(radius_px * ypp, height)
    x0, x1 = x - width / 2, x + width / 2
    t = np.linspace(0, np.pi / 2, 8)
    left = [(x0 + rx - rx * np.cos(a), height - ry + ry * np.sin(a)) for a in t]
    right = [(x1 - rx + rx * np.sin(a), height - ry + ry * np.cos(a)) for a in t]
    verts = [(x0, 0)] + left + right + [(x1, 0), (x0, 0)]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none"))


def grouped(ax, runs, key, sizes, methods, ymax=1.0, labels=True, gap_px=16):
    ax.set_xlim(-0.5, len(methods) - 0.5)
    ax.set_ylim(0, ymax)
    width = min(0.8 / len(sizes), BAR_PX * data_per_px(ax)[0])
    gap = gap_px * data_per_px(ax)[0]  # wide enough for the value labels to clear their neighbors
    for i, method in enumerate(methods):
        present = [s for s in sizes if (method, s) in runs]
        for j, size in enumerate(present):
            x = i + (j - (len(present) - 1) / 2) * (width + gap)
            v = runs[(method, size)][key]
            column(ax, x, width, v, COLOR[size])
            if labels:
                ax.text(x, v + ymax * 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(methods)), [METHOD_LABEL[m] for m in methods])


def legend(fig, sizes, y=0.0, baseline=None):
    handles = [Rectangle((0, 0), 1, 1, facecolor=COLOR[s]) for s in sizes]
    names = [f"Qwen3-VL-{s.upper()}" for s in sizes]
    if baseline is not None:
        handles.append(Line2D([0], [0], color=INK2, linewidth=1))
        names.append(f"best constant answer ({baseline:.2f})")
    fig.legend(handles, names, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, y))


def smooth(y, k):
    if len(y) < k:
        return np.asarray(y)
    return np.convolve(y, np.ones(k) / k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--tasks", type=Path, default=Path("data/tasks"))
    ap.add_argument("--split", default="val", help="prefix of the runs' metrics/predictions files")
    ap.add_argument("--base-split", default=None, help="task split the samples come from (default: --split)")
    ap.add_argument("--out", type=Path, default=Path("docs"))
    args = ap.parse_args()
    (args.out / "figures").mkdir(parents=True, exist_ok=True)

    runs = find_runs(args.runs, args.split)
    sizes = [s for s in SIZES if any(k[1] == s for k in runs)]
    methods = [m for m in METHODS if any(k[0] == m for k in runs)]
    samples = load_split(args.tasks, args.base_split or args.split)
    if args.base_split:  # a subset: keep the samples the runs were scored on
        seen = {i for r in runs.values() for i in predictions(r, args.split)}
        samples = [s for s in samples if s["id"] in seen]
    baseline = max(constant_baselines(samples).values())

    # table
    lines = [f"| Model | Method | n | Total | Hands | Object | Point |", "|---|---|---|---|---|---|---|",
             f"| - | best constant answer | | {baseline:.3f} | | | |"]
    for size in sizes:
        for method in methods:
            if (method, size) in runs:
                r = runs[(method, size)]
                lines.append(f"| Qwen3-VL-{size.upper()} | {METHOD_LABEL[method]} | {r['n']} | **{r['total']:.3f}** | "
                             f"{r['hands']:.3f} | {r['object']:.3f} | {r['point']:.3f} |")

    # post-training vs its SFT starting point, on the same samples
    pairs = [(m, s) for s in sizes for m in ("grpo", "dpo") if (m, s) in runs and ("sft", s) in runs]
    if pairs:
        lines += ["", f"Change from SFT on the same {args.split} samples (mean, 95% paired bootstrap interval):", "",
                  "| Model | Method | Total | Hands | Point |", "|---|---|---|---|---|"]
        for m, s in pairs:
            a, b = predictions(runs[(m, s)], args.split), predictions(runs[("sft", s)], args.split)
            cells = [paired_delta(a, b, k) for k in ("total", "hands", "point")]
            lines.append(f"| Qwen3-VL-{s.upper()} | {METHOD_LABEL[m]} | "
                         + " | ".join(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for d, lo, hi in cells) + " |")

    # how often each hand is called busy, against the labels
    gold = [s["answer"] for s in samples]
    lines += ["", "Fraction of frames in which a hand is said to be busy:", "",
              "| | Left | Right |", "|---|---|---|",
              f"| labels | {busy_rate(gold, 'left'):.2f} | {busy_rate(gold, 'right'):.2f} |"]
    for size in sizes:
        for method in methods:
            if (method, size) in runs:
                ans = [rewards.parse(r["completion"]) for r in predictions(runs[(method, size)], args.split).values()]
                lines.append(f"| Qwen3-VL-{size.upper()} {METHOD_LABEL[method]} | "
                             f"{busy_rate(ans, 'left'):.2f} | {busy_rate(ans, 'right'):.2f} |")
    (args.out / f"results_{args.split}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # total reward
    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor=SURFACE, dpi=DPI)
    style(ax)
    ax.set_ylim(0, 1)
    fig.subplots_adjust(bottom=0.2, top=0.88, left=0.07, right=0.98)
    ax.axhline(baseline, color=INK2, linewidth=1, zorder=3)
    grouped(ax, runs, "total", sizes, methods)
    ax.set_title(f"Total reward on {args.split} (held-out participant)", loc="left", fontsize=11, color=INK)
    legend(fig, sizes, baseline=baseline)
    fig.savefig(args.out / "figures" / f"{args.split}_total.png", dpi=DPI, facecolor=SURFACE)

    # components
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), facecolor=SURFACE, sharey=True, dpi=DPI)
    fig.subplots_adjust(bottom=0.22, top=0.85, left=0.05, right=0.99, wspace=0.08)
    for ax, comp in zip(axes, COMPONENTS):
        style(ax)
        grouped(ax, runs, comp, sizes, methods, labels=False)
        ax.set_title(comp.capitalize(), loc="left", fontsize=10, color=INK)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"Reward components on {args.split}", x=0.05, ha="left", fontsize=11, color=INK)
    legend(fig, sizes)
    fig.savefig(args.out / "figures" / f"{args.split}_components.png", dpi=DPI, facecolor=SURFACE)

    # training curves
    for method, key, ylabel, k, name in (("sft", "loss", "answer-token cross-entropy", 20, "sft_loss"),
                                         ("grpo", "reward_total", "mean rollout reward", 10, "grpo_reward")):
        logs = {}
        for size in SIZES:
            f = next((d / "log.jsonl" for (m, s), r in runs.items() if m == method and s == size
                      for d in [r["dir"]] if (d / "log.jsonl").exists()), None)
            if f:
                rows = [json.loads(line) for line in open(f)]
                rows = [r for r in rows if key in r]
                logs[size] = ([r["step"] for r in rows], [r[key] for r in rows])
        if not logs:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 3.4), facecolor=SURFACE, dpi=DPI)
        style(ax)
        fig.subplots_adjust(bottom=0.22, top=0.86, left=0.1, right=0.98)
        for size, (steps, ys) in logs.items():
            sm = smooth(ys, k)
            ax.plot(steps[len(steps) - len(sm):], sm, color=COLOR[size], linewidth=2, solid_capstyle="round")
        ax.set_xlabel("optimizer step", fontsize=9, color=INK2)
        ax.set_ylabel(ylabel, fontsize=9, color=INK2)
        ax.set_title(f"{METHOD_LABEL[method]}: {ylabel} ({k}-step moving average)", loc="left", fontsize=11, color=INK)
        legend(fig, list(logs))
        fig.savefig(args.out / "figures" / f"{name}.png", dpi=DPI, facecolor=SURFACE)
    print(f"figures in {args.out / 'figures'}")


if __name__ == "__main__":
    main()
