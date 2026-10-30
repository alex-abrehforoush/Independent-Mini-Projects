"""Experiment 1 report: rate-distortion figure + markdown tables from runs/tok/*/eval_recon.json.

The figure and main table use the runs trained for --steps (the equal-budget comparison); the
ablation table lists the step-matched 12k runs (trajectory coding, context-free vs predictive keys)."""
import argparse
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# reference palette, light mode: slot 1 blue, 2 orange, 3 aqua (validated all-pairs for 3 series)
C_FR, C_KFA, C_KFU = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load(root, steps=None):
    runs = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "eval_recon.json"))):
        r = json.load(open(f))
        if steps is None or r["config"]["steps"] == steps:
            runs[os.path.basename(os.path.dirname(f))] = r
    return runs


def by_rate(point):
    """Sort key for (bits/s, point, run name) entries; equal-rate runs fall back to the name."""
    return point[0], point[2]


def pareto(points):
    """Lower-left frontier of (x, y) points."""
    pts = sorted(points)
    out, best = [], float("inf")
    for x, y in pts:
        if y < best:
            out.append((x, y))
            best = y
    return out


def style(ax, ylabel):
    ax.set_xscale("log")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlabel("bitrate (bits / s, log scale)", color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def series(runs):
    fr = {"fsq": [], "vq": [], "rvq": []}
    kfa, kfu = [], []
    for name, r in runs.items():
        cfg = r["config"]
        if cfg["kind"] == "fr":
            if cfg["quant"] in fr:
                p = r["points"][0]
                fr[cfg["quant"]].append((p["bits_per_s"], p, name))
        elif cfg["quant"] != "none":
            for p in r["points"]:
                if p.get("variant") != "quantized":
                    continue
                (kfa if p["placement"] == "adaptive" else kfu).append((p["bits_per_s"], p, name))
    return fr, kfa, kfu


def figure(runs, out, metrics_=(("mpjpe_cm", "mean joint error (cm)"),
                                 ("mpjpe_local_cm", "root-relative joint error (cm)"),
                                 ("skate_cm_s", "foot skate (cm/s)"),
                                 ("jitter_ratio", "jitter / ground-truth jitter"))):
    fr, kfa, kfu = series(runs)
    gt = next(iter(runs.values()))["gt"]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.0), dpi=130)
    axes = axes.flatten()
    for ax, (key, label) in zip(axes, metrics_):
        def val(p):
            return p["jitter"] / gt["jitter"] if key == "jitter_ratio" else p[key]
        for pts, col, name, mk in ((kfu, C_KFU, "keyframe, uniform keys", "s"), (kfa, C_KFA, "keyframe, adaptive keys", "o")):
            if not pts:
                continue
            xy = [(x, val(p)) for x, p, _ in pts if x < 3000]
            ax.scatter(*zip(*xy), s=10, color=col, alpha=0.35, lw=0)
            if key == "jitter_ratio":  # no frontier (closer to 1 is better): each model's own curve
                for run in sorted({n for _, _, n in pts}):
                    ax.plot(*zip(*sorted((x, val(p)) for x, p, n in pts if n == run and x < 3000)),
                            color=col, lw=1.5, marker=mk, ms=3)
            else:
                ax.plot(*zip(*pareto(xy)), color=col, lw=2, marker=mk, ms=4, label=name)
        fsq = sorted(fr["fsq"], key=by_rate)
        if fsq:
            ax.plot([x for x, _, _ in fsq], [val(p) for _, p, _ in fsq], color=C_FR, lw=2, marker="D", ms=5,
                    label="fixed-rate FSQ (1 token / 4 frames)")
        for q, mk, dy in (("vq", "v", -10), ("rvq", "^", 4)):  # same bitrate: one label above, one below
            for x, p, _ in fr[q]:
                ax.scatter([x], [val(p)], color=C_FR, marker=mk, s=60, edgecolor="white", lw=1.5, zorder=5)
                ax.annotate(q.upper(), (x, val(p)), textcoords="offset points", xytext=(6, dy), fontsize=7, color=INK2)
        if key == "mpjpe_cm":  # each keyframe model levels off at its highest key rate: label where it ends
            for run in sorted({n for _, _, n in kfa + kfu}):
                x, p = max((x, p) for x, p, n in kfa + kfu if n == run and x < 3000)
                bits = runs[run]["config"]["groups"] * math.log2(1000)
                ax.annotate(f"{bits:.0f} bits/key", (x, val(p)), textcoords="offset points", xytext=(0, 5),
                            ha="right", fontsize=7, color=INK2)
        if key == "skate_cm_s":
            ax.axhline(gt["skate_cm_s"], color=INK2, lw=1, ls="--")
            ax.annotate("ground truth", (0.01, gt["skate_cm_s"]), xycoords=ax.get_yaxis_transform(), textcoords="offset points",
                        xytext=(0, 3), fontsize=7, color=INK2)
        if key == "jitter_ratio":
            ax.axhline(1.0, color=INK2, lw=1, ls="--")
            ax.annotate("ground truth", (0.01, 1.0), xycoords=ax.get_yaxis_transform(), textcoords="offset points",
                        xytext=(0, 3), fontsize=7, color=INK2)
        style(ax, label)
    axes[0].legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def table(runs):
    fr, kfa, kfu = series(runs)
    rows = []
    for x, p, name in sorted(fr["fsq"] + fr["vq"] + fr["rvq"], key=by_rate):
        rows.append((name, "fixed", f"{x:.0f}", "7.5 tok/s", p))
    for pts, lab in ((kfu, "uniform"), (kfa, "adaptive")):
        for x, p, name in sorted(pts, key=by_rate):
            if x > 1500:
                continue
            rows.append((name, f"{lab} {p['param']}", f"{x:.0f}", f"{p['keys_per_s']:.1f} keys/s", p))
    return rows_to_md(rows)


def rows_to_md(rows):
    lines = ["| model | placement | bits/s | rate | MPJPE cm | root-relative cm | root cm | max err cm | foot skate cm/s | jitter |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for name, pl, b, rate, p in rows:
        lines.append(f"| {name} | {pl} | {b} | {rate} | {p['mpjpe_cm']:.2f} | {p['mpjpe_local_cm']:.2f} | "
                     f"{p['root_err_cm']:.2f} | {p['max_err_cm']:.1f} | {p['skate_cm_s']:.1f} | {p['jitter']:.0f} |")
    return "\n".join(lines)


def ablation_table(runs, params=(4, 8)):
    """Step-matched ablations: fixed-rate at its one point, keyframe models at uniform strides."""
    rows = []
    for name, r in sorted(runs.items()):
        for p in r["points"]:
            if r["config"]["kind"] == "fr":
                rows.append((name, "fixed", f"{p['bits_per_s']:.0f}", "7.5 tok/s", p))
            elif p["variant"] == "quantized" and p["placement"] == "uniform" and p["param"] in params:
                rows.append((name, f"uniform {p['param']}", f"{p['bits_per_s']:.0f}", f"{p['keys_per_s']:.1f} keys/s", p))
    return rows_to_md(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/tok")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--steps", type=int, default=48000)
    ap.add_argument("--ablation_steps", type=int, default=12000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load(a.runs, a.steps)
    figure(runs, os.path.join(a.out, "fig_rate_distortion.png"))
    t = table(runs)
    open(os.path.join(a.out, "table_tokenizers.md"), "w").write(t + "\n")
    print(t)
    t = ablation_table(load(a.runs, a.ablation_steps))
    open(os.path.join(a.out, "table_tokenizer_ablations.md"), "w").write(t + "\n")
    print(t)
