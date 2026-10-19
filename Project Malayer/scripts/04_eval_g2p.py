"""Evaluate G2P systems: accuracy, streaming lookahead trade-off, latency, guardrails.

    python scripts/04_eval_g2p.py --run runs/g2p_base

Writes runs/<run>/eval_g2p.json and docs/figures/lookahead_tradeoff.png
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from malayer.g2p.baselines import EspeakG2P, LexiconG2P
from malayer.g2p.data import read_jsonl
from malayer.g2p.infer import Phonemizer
from malayer.g2p.metrics import G2PScores

LOOKAHEADS = [0, 1, 2, 3]


def score(items: list[dict], hyps: list[str]) -> G2PScores:
    s = G2PScores()
    for it, h in zip(items, hyps):
        s.add(h, it["ref"], it["g"], it.get("hom"), it.get("hom_ref"))
    return s


def latency_ms(fn, texts: list[str], repeats: int = 1) -> dict:
    times = []
    for t in texts:
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn(t)
        times.append((time.perf_counter() - t0) * 1000 / repeats)
    times.sort()
    return {"p50": round(statistics.median(times), 2), "p95": round(times[int(0.95 * len(times))], 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/g2p_base")
    ap.add_argument("--test-sample", type=int, default=5000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    run = Path(args.run)

    sb = read_jsonl("data/g2p/sentencebench.jsonl")
    test = read_jsonl("data/g2p/test_homorich.jsonl")
    random.Random(0).shuffle(test)
    test = test[: args.test_sample]
    sets = {"sentencebench": sb, "homorich_test": test}

    ph = Phonemizer.from_dir(run, device=args.device)
    espeak = EspeakG2P()
    lexicon = LexiconG2P(read_jsonl("data/g2p/train.jsonl"), espeak)

    results: dict = {"systems": {}, "subsets": {}}
    for name, items in sets.items():
        texts = [it["g"] for it in items]
        sys_hyps = {
            "espeak-ng": [espeak(t) for t in texts],
            "lexicon (no context)": [lexicon(t) for t in texts],
            "tagger offline": [h for i in range(0, len(texts), 256) for h in ph.phonemize_batch(texts[i: i + 256])],
        }
        for k in LOOKAHEADS:
            sys_hyps[f"tagger stream k={k}"] = [ph.phonemize_streaming(t, k, normalized=True) for t in texts]
        for sname, hyps in sys_hyps.items():
            s = score(items, hyps)
            results["systems"].setdefault(sname, {})[name] = s.summary()
            print(f"{name:15s} {sname:24s} {s.summary()}")
        # SentenceBench per-source breakdown for the offline tagger and espeak
        if name == "sentencebench":
            for src in sorted({it["source"] for it in items}):
                idx = [i for i, it in enumerate(items) if it["source"] == src]
                results["subsets"][src] = {
                    sname: score([items[i] for i in idx], [sys_hyps[sname][i] for i in idx]).summary()
                    for sname in ("espeak-ng", "tagger offline", "tagger stream k=1")}
        # guardrail: the tagger never changes the number of written words
        if name == "homorich_test":
            mism = sum(len(ph.tag_words(t)) != len(t.split(" ")) for t in texts)
            esp_mism = sum(len(h.split()) != len([w for w in t.split(" ") if w.strip(".,!?؟،؛:«»()\"'")])
                           for t, h in zip(texts, sys_hyps["espeak-ng"]))
            results["guardrail_word_count_mismatch"] = {"tagger": mism, "espeak-ng": esp_mism, "n": len(texts)}

    # CPU latency, single thread, one sentence at a time (the streaming server's regime)
    torch.set_num_threads(1)
    ph_cpu = Phonemizer.from_dir(run, device="cpu")
    lat_texts = [it["g"] for it in sb[:200]]
    results["latency_ms_cpu_1thread"] = {
        "espeak-ng": latency_ms(espeak, lat_texts),
        "tagger offline (per sentence)": latency_ms(lambda t: ph_cpu.phonemize(t, normalized=True), lat_texts),
        "tagger stream k=1 (whole sentence, all commits)": latency_ms(
            lambda t: ph_cpu.phonemize_streaming(t, 1, normalized=True), lat_texts),
    }
    # what a streaming voice agent actually waits for: one commit = tag the visible prefix once
    prefixes = []
    for t in lat_texts:
        words = t.split(" ")
        prefixes += [" ".join(words[: i + 1]) for i in range(len(words))]
    results["latency_ms_cpu_1thread"]["tagger single commit (prefix re-tag)"] = latency_ms(
        lambda p: ph_cpu.tag_words(p, eos=False), prefixes[::5])
    results["n_params"] = sum(p.numel() for p in ph.model.parameters())
    (run / "eval_g2p.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results["latency_ms_cpu_1thread"], indent=2))

    # Figure: accuracy vs lookahead
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, metric, title in zip(axes, ["ezafe_F1", "homograph_acc", "PER"],
                                 ["Ezafe F1 (HomoRich test)", "Homograph acc. (SentenceBench)", "PER % (SentenceBench)"]):
        subset = "homorich_test" if metric == "ezafe_F1" else "sentencebench"
        ys = [results["systems"][f"tagger stream k={k}"][subset][metric] for k in LOOKAHEADS]
        off = results["systems"]["tagger offline"][subset][metric]
        esp = results["systems"]["espeak-ng"][subset][metric]
        ax.plot(LOOKAHEADS, ys, marker="o", color="#2a6fdb", label="tagger, streaming")
        ax.axhline(off, color="#2a6fdb", ls="--", lw=1, label="tagger, full sentence")
        ax.axhline(esp, color="#999999", ls=":", lw=1.2, label="espeak-ng")
        ax.set_xticks(LOOKAHEADS)
        ax.set_xlabel("lookahead (words)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "lookahead_tradeoff.png", dpi=150)
    print("saved", out / "lookahead_tradeoff.png")


if __name__ == "__main__":
    main()
