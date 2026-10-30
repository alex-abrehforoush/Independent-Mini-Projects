"""Experiment 2 report: in-betweening on the LAFAN1 benchmark (L2Q/L2P/NPSS at the standard gaps),
long gaps, physical plausibility and diversity, from runs/gen/*/eval_inbetween.json; token diagnostics
from runs/gen/*/token_analysis.json (scripts/22_token_analysis.py)."""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reference palette, light mode: slot 1 blue (fixed rate), 2 orange (keyframe), 3 aqua (uniform keys)
C_FR, C_KF, C_KFU, C_REF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
METHODS = (  # run name, label, colour, line style
    ("kf_g4p", "keyframe tokens, FSQ", C_KF, "-"),
    ("kf_cont", "keyframe tokens, continuous + flow", C_KF, "--"),
    ("fr_fsq_g4", "fixed-rate tokens, FSQ", C_FR, "-"),
    ("fr_cont", "fixed-rate tokens, continuous + flow", C_FR, "--"),
    ("kf_g4p_uniform4", "keyframe tokens, FSQ, uniform keys", C_KFU, ":"),
)
HEAD_ABLATION = (  # same backbone and tokenizer, one 1000-way softmax per FSQ group instead of per dimension
    ("fr_fsq_g4_joint", "fr_fsq_g4", "fixed-rate tokens, FSQ"),
    ("kf_g4p_joint", "kf_g4p", "keyframe tokens, FSQ"),
)
CONT_ABLATION = ("kf_cont_features", "keyframe tokens, continuous raw key features + flow")  # 136-d instead of 32-d latents
PUBLISHED = ("RMIB", "∆-I", "TST", "SILK")
GAPS = (5, 15, 30, 45)


def published(path):
    """Rows of the LAFAN1 table reprinted in SILK (2025): name -> {metric: [4 values or None]}."""
    out = {}
    for line in open(path):
        tok = line.split()
        if len(tok) < 14 or tok[0] in ("Task", "Length"):
            continue
        name = tok[0]
        vals = [None if v == "-" else float(v) for v in tok[-12:]]
        out[name] = {"L2P": vals[0:4], "L2Q": vals[4:8], "NPSS": vals[8:12]}
    return out


def fmt(v, metric):
    if v is None:
        return "–"
    return f"{v:.4f}" if metric == "NPSS" else f"{v:.2f}"


def benchmark_table(slerp, runs, pub):
    head = "| method | " + " | ".join(f"{m} {n}" for m in ("L2Q", "L2P", "NPSS") for n in GAPS) + " |"
    lines = [head, "|---|" + "---:|" * 12]
    lines.append("| SLERP (official script, this repo) | " + " | ".join(
        fmt(slerp[str(n)][m], m) for m in ("L2Q", "L2P", "NPSS") for n in GAPS) + " |")
    for name in PUBLISHED:
        if name in pub:
            lines.append(f"| {name} (published) | " + " | ".join(
                fmt(pub[name][m][i], m) for m in ("L2Q", "L2P", "NPSS") for i in range(4)) + " |")
    rows = [(label, name) for name, label, _, _ in METHODS] + [(CONT_ABLATION[1] + " (ablation)", CONT_ABLATION[0])]
    for label, name in rows:
        if name in runs:
            lines.append(f"| **{label}** | " + " | ".join(
                fmt(runs[name][str(n)][m], m) for m in ("L2Q", "L2P", "NPSS") for n in GAPS) + " |")
    return "\n".join(lines)


def physics_table(slerp, runs, gaps=(15, 45, 90)):
    lines = ["| method | gap | L2P | foot skate, own contacts cm/s (GT) | foot skate, GT contacts cm/s (GT) | jitter / GT | "
             "penetration cm | diversity APD cm | tokens or keys |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    rows = [("SLERP", slerp)] + [(label, runs[name]) for name, label, _, _ in METHODS if name in runs]
    for n in gaps:
        for label, r in rows:
            p = r[str(n)]
            apd = f"{p['apd_cm']:.1f}" if "apd_cm" in p else "0"
            tk = f"{p['tokens_or_keys']:.1f}" if "tokens_or_keys" in p else "–"
            lines.append(f"| {label} | {n} | {p['L2P']:.2f} | {p['skate_own_cm_s']:.1f} ({p['skate_own_gt_cm_s']:.1f}) | "
                         f"{p['skate_cm_s']:.1f} ({p['skate_gt_cm_s']:.1f}) | "
                         f"{p['jitter'] / p['jitter_gt']:.2f} | {p['penetration_cm']:.2f} | {apd} | {tk} |")
    return "\n".join(lines)


def head_table(runs, gaps=(5, 15, 45, 90)):
    lines = ["| tokens | code head | " + " | ".join(f"L2P {n}" for n in gaps) + " | foot skate cm/s @45 |",
             "|---|---|" + "---:|" * (len(gaps) + 1)]
    for joint, digits, label in HEAD_ABLATION:
        for name, head in ((joint, "joint softmax per group"), (digits, "softmax per dimension")):
            if name in runs:
                r = runs[name]
                lines.append(f"| {label} | {head} | " + " | ".join(f"{r[str(n)]['L2P']:.2f}" for n in gaps)
                             + f" | {r['45']['skate_cm_s']:.1f} |")
    return "\n".join(lines)


def token_table(analysis, gaps=(5, 15, 45, 90)):
    """scripts/22_token_analysis.py: what each generator has to predict, and what a mistake costs."""
    lines = ["| tokens | gap | emitted bits/s | generator cross-entropy bits/s | one-level code error: cm moved | "
             "frames moved > 1 cm | ground-truth codes cm | generator (greedy) cm | SLERP cm |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in gaps:
        for name, label, _, _ in METHODS:
            if name in analysis and str(n) in analysis[name]:
                p = analysis[name][str(n)]
                lines.append(f"| {label} | {n} | {p['nominal_bits_per_s']:.0f} | {p['ce_bits_per_s']:.0f} "
                             f"({p['ce_bits_per_s'] / p['nominal_bits_per_s']:.0%}) | {p['nudge_cm']:.2f} | "
                             f"{p['nudge_reach']:.0%} | {p['ceiling_mpjpe_cm']:.2f} | {p['greedy_mpjpe_cm']:.2f} | "
                             f"{p['slerp_mpjpe_cm']:.2f} |")
    return "\n".join(lines)


def figure(slerp, runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=130)
    gaps = sorted(int(n) for n in slerp)
    for ax, key, label in ((axes[0], "L2P", "L2P (normalised global positions)"),
                           (axes[1], "skate_own_cm_s", "foot skate on own contacts (cm/s)")):
        ax.plot(gaps, [slerp[str(n)][key] for n in gaps], color=C_REF, lw=2, marker="o", ms=4, label="SLERP")
        for name, lab, col, ls in METHODS:
            if name in runs:
                ax.plot(gaps, [runs[name][str(n)][key] for n in gaps], color=col, ls=ls, lw=2, marker="o", ms=4, label=lab)
        if key == "skate_own_cm_s":
            ax.plot(gaps, [slerp[str(n)]["skate_own_gt_cm_s"] for n in gaps], color=INK2, lw=1, ls="--")
            ax.annotate("ground truth", (gaps[0], slerp[str(gaps[0])]["skate_own_gt_cm_s"]), textcoords="offset points",
                        xytext=(2, 4), fontsize=7, color=INK2)
        ax.axvspan(45, max(gaps) + 5, color=GRID, alpha=0.5, lw=0)
        ax.text(0.98, 0.1, "beyond the benchmark", transform=ax.transAxes, fontsize=7, color=INK2, ha="right")
        ax.set_xlim(0, max(gaps) + 5)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.set_xlabel("transition length (frames at 30 fps)", color=INK2, fontsize=9)
        ax.set_ylabel(label, color=INK2, fontsize=9)
    axes[0].legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/gen")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--published", default="docs/published_lafan1_table_silk2025.txt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    slerp = json.load(open(os.path.join(a.runs, "baselines.json")))["slerp"]
    names = [m[0] for m in METHODS] + [h[0] for h in HEAD_ABLATION] + [CONT_ABLATION[0]]
    runs = {name: json.load(open(f)) for name in names
            if os.path.exists(f := os.path.join(a.runs, name, "eval_inbetween.json"))}
    main = {k: v for k, v in runs.items() if k in {m[0] for m in METHODS}}
    t1 = benchmark_table(slerp, {k: v for k, v in runs.items() if k in main or k == CONT_ABLATION[0]},
                         published(a.published))
    t2 = physics_table(slerp, main)
    t3 = head_table(runs)
    analysis = {name: json.load(open(f)) for name in main
                if os.path.exists(f := os.path.join(a.runs, name, "token_analysis.json"))}
    t4 = token_table(analysis)
    open(os.path.join(a.out, "table_inbetweening.md"), "w").write("\n\n".join((t1, t2, t3, t4)) + "\n")
    figure(slerp, main, os.path.join(a.out, "fig_inbetweening.png"))
    print("\n\n".join((t1, t2, t3, t4)))
