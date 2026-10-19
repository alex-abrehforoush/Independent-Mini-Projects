"""Fit and validate the IPA->R2 projection of the phoneme recognizer on human speech.

Uses Mana-TTS recordings whose transcripts appear in our aligned HomoRich data
(so their intended phonemes are known). Half the clips fit the projection, the
other half validate it:

  * greedy R2 PER on human speech
  * minimal-pair forced choice: for every word where the human reference and
    espeak's reading differ, does the audio prefer the reference? (lower bound:
    the reference annotations themselves are noisy)
  * synthetic minimal pairs: the same sentence synthesized with both readings;
    can the metric tell them apart? (ground truth known by construction)

    python scripts/03_calibrate_recognizer.py --device cpu
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from malayer.eval.phoneme_ctc import R2_CLASSES, SEED, PhonemeCTC, r2_symbols, same_for_scoring
from malayer.eval.readings import espeak_word_readings, gold_word_readings
from malayer.g2p.baselines import EspeakG2P
from malayer.g2p.data import read_jsonl
from malayer.g2p.metrics import edit_distance
from malayer.phonemes import split_ezafe
from malayer.text.normalize import normalize
from malayer.tts.frontend import PersianFrontend
from malayer.tts.synth import Synthesizer


def load_clips(shards: list[Path], refs: dict[str, dict], max_sec: float) -> list[dict]:
    clips = []
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=64, columns=["transcript", "audio", "samplerate", "duration"]):
            # audio is list<double>; slice the flat value buffer instead of converting to Python lists
            audio = batch.column("audio")
            offs, vals = audio.offsets.to_numpy(), audio.values.to_numpy()
            texts, srs, durs = (batch.column(c).to_pylist() for c in ("transcript", "samplerate", "duration"))
            for i, (t, sr, dur) in enumerate(zip(texts, srs, durs)):
                g = normalize(t or "")
                if g in refs and dur <= max_sec:
                    clips.append({"g": g, "item": refs[g], "audio": vals[offs[i]: offs[i + 1]].astype(np.float32),
                                  "sr": int(sr)})
    return clips


def align_tokens(tokens: list[str], ref: list[str]) -> list[tuple[str | None, str | None]]:
    """Edit-distance alignment of recognized IPA tokens to reference R2 symbols."""
    def match(t, r):
        return t in SEED.get(r, []) or t == r

    n, m = len(tokens), len(ref)
    d = np.zeros((n + 1, m + 1))
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (0 if match(tokens[i - 1], ref[j - 1]) else 1))
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (0 if match(tokens[i - 1], ref[j - 1]) else 1):
            pairs.append((tokens[i - 1], ref[j - 1])); i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            pairs.append((tokens[i - 1], None)); i -= 1
        else:
            pairs.append((None, ref[j - 1])); j -= 1
    return pairs[::-1]


def fit_projection(confusion: dict[str, collections.Counter], min_count: int = 5) -> dict[str, dict[str, float]]:
    table = {}
    for tok, cnt in confusion.items():
        total = sum(cnt.values())
        if total < min_count:
            continue
        table[tok] = {c: n / total for c, n in cnt.items() if n / total >= 0.02}
        s = sum(table[tok].values())
        table[tok] = {c: round(p / s, 4) for c, p in table[tok].items()}
    for r2, toks in SEED.items():  # rare seed tokens: trust the seed
        for t in toks:
            table.setdefault(t, {r2: 1.0})
    for c in R2_CLASSES:  # identical ASCII consonant tokens
        if c.isalpha():
            table.setdefault(c, {c: 1.0})
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-clips", type=int, default=600)
    ap.add_argument("--max-sec", type=float, default=12.0)
    ap.add_argument("--synthetic-pairs", type=int, default=240)
    ap.add_argument("--cache", default="data/cache/calib_logprobs.pt")
    ap.add_argument("--out", default="configs/recognizer_projection.json")
    ap.add_argument("--report", default="runs/recognizer_calibration.json")
    args = ap.parse_args()
    torch.set_num_threads(10)

    refs = {it["g"]: it for f in ("train", "dev") for it in read_jsonl(f"data/g2p/{f}.jsonl")}
    rec = PhonemeCTC(device=args.device, projection=None)
    cache = Path(args.cache)
    if cache.exists():
        clips = torch.load(cache)
        for c in clips:
            c["item"] = refs[c["g"]]
    else:
        clips = load_clips(sorted(Path("data/raw/mana").rglob("*.parquet")), refs, args.max_sec)
        random.Random(0).shuffle(clips)
        clips = clips[: args.max_clips]
        for c in tqdm(clips, desc="recognizer"):
            c["lp"] = rec.token_logprobs(c["audio"], c["sr"]).half()
            del c["audio"]
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save([{"g": c["g"], "lp": c["lp"]} for c in clips], cache)
    fit, val = clips[: len(clips) // 2], clips[len(clips) // 2:]
    print(f"clips: fit={len(fit)} val={len(val)}")

    # 1) confusion counts on the fit half -> soft projection
    confusion: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for c in fit:
        for tok, r2 in align_tokens(rec.greedy_tokens(c["lp"].float()), r2_symbols(c["item"]["ref"])):
            if tok is not None:
                confusion[tok][r2 if r2 is not None else "other"] += 1
    table = fit_projection(confusion)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(table, ensure_ascii=False, indent=1))
    rec.set_projection(table)

    # 2a) human speech, held-out half. Gold = HomoRich annotation of the Mana transcript,
    #     which is itself noisy, so this is a *lower bound* on metric reliability.
    esp = EspeakG2P()
    edits = ref_len = 0
    fc = collections.defaultdict(lambda: [0, 0])  # kind -> [correct, total]
    contested = []  # (sentence, gold readings, word index, alternative) for the synthetic test
    for c in tqdm(val, desc="validate/human"):
        clp = rec.class_logprobs(c["lp"].float())
        hyp = rec.greedy_r2(clp)
        ref = "".join(r2_symbols(c["item"]["ref"]))
        edits += edit_distance(hyp, ref)
        ref_len += len(ref)
        gold = gold_word_readings(c["item"])
        alt = espeak_word_readings(c["g"], esp)
        if len(gold) != len(alt):
            continue
        for k, (gw, aw) in enumerate(zip(gold, alt)):
            if not gw or not aw or same_for_scoring(gw, aw):
                continue
            kind = _kind(gw, aw)
            fc[kind][0] += rec.forced_choice(clp, gold, k, aw) > 0
            fc[kind][1] += 1
            contested.append((c["g"], gold, k, aw, kind))

    # 2b) synthetic minimal pairs with known ground truth: the same sentence is synthesized
    #     with reading A and with reading B of one word (phoneme injection into the voice).
    #     Discrimination = LLR(A vs B) is higher on A-audio than on B-audio (ABX-style).
    #     This measures metric + voice controllability together, free of annotation noise.
    synth = Synthesizer(PersianFrontend())
    random.Random(1).shuffle(contested)
    disc = collections.defaultdict(lambda: [0, 0, 0])  # kind -> [discriminated, both-correct, total]
    for g, gold, k, aw, kind in tqdm(contested[: args.synthetic_pairs], desc="validate/synthetic"):
        alt_reading = list(gold)
        alt_reading[k] = aw
        llr = []
        for reading in (gold, alt_reading):
            ipa = " ".join(w.ipa for w in synth.frontend.words_from_tagged(*_tagged(g, reading)))
            audio = synth.ipa_to_audio(ipa)
            clp = rec.class_logprobs(rec.token_logprobs(audio, synth.sample_rate))
            llr.append(rec.forced_choice(clp, gold, k, aw))
        disc[kind][0] += llr[0] > llr[1]
        disc[kind][1] += llr[0] > 0 > llr[1]
        disc[kind][2] += 1

    def pct(a, n):
        return round(100 * a / max(n, 1), 2)

    report = {
        "n_fit": len(fit), "n_val": len(val),
        "greedy_PER_human": pct(edits, ref_len),
        "human_forced_choice_vs_noisy_gold": {k: {"acc": pct(a, n), "n": n} for k, (a, n) in fc.items()},
        "human_forced_choice_all": pct(sum(a for a, _ in fc.values()), sum(n for _, n in fc.values())),
        "synthetic_minimal_pairs": {k: {"discrimination": pct(d, n), "both_correct": pct(b, n), "n": n}
                                    for k, (d, b, n) in disc.items()},
        "synthetic_discrimination_all": pct(sum(v[0] for v in disc.values()), sum(v[2] for v in disc.values())),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def _kind(gold: str, alt: str) -> str:
    gb, ge = split_ezafe(gold)
    ab, ae = split_ezafe(alt)
    return "ezafe" if ge != ae and same_for_scoring(gb, ab) else "vowel/other"


def _tagged(g: str, readings: list[str]) -> tuple[list[str], list[tuple[str | None, bool]]]:
    """Tokens of `g` and a (bare, ezafe) per token from per-written-word readings."""
    tokens = g.split(" ")
    it = iter(readings)
    tagged = [split_ezafe(next(it)) if t.strip(".,!?؟،؛:«»()\"'") else (None, False) for t in tokens]
    return tokens, tagged


if __name__ == "__main__":
    main()
