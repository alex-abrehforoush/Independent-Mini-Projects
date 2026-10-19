"""Build tagging datasets for the G2P model from HomoRich + SentenceBench.

Outputs (data/g2p/):
    train.jsonl, dev.jsonl        HomoRich recommended train split, de-duplicated and
                                  with every test / SentenceBench sentence removed
    test_homorich.jsonl           HomoRich recommended test split (homograph-focused)
    sentencebench.jsonl           400-sentence external benchmark (references in R2)
    labels.json                   label vocabulary (by frequency)
    prep_report.json              acceptance / rejection statistics

Each line: {"g": graphemes, "labels": [...], "ezafe": [...], "ref": canonical R2,
            "source": ..., "hom": homograph word or null, "hom_ref": its R2 or null}
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from malayer.g2p.align import align
from malayer.phonemes import canonical_r2, r1_to_r2
from malayer.text.normalize import is_persian_only, normalize

RAW = Path("data/raw")
OUT = Path("data/g2p")


def _process(row: tuple) -> dict | str:
    g, ph, source, hom, hom_ref = row
    if not isinstance(g, str) or not isinstance(ph, str):
        return "missing"
    g = normalize(g)
    if not g or not is_persian_only(g):
        return "non_persian_chars"
    a = align(g, ph)
    if a is None:
        return "unalignable"
    hom = normalize(hom) if isinstance(hom, str) else None
    return {
        "g": g, "labels": a.labels, "ezafe": a.ezafe, "ref": a.to_r2(),
        "source": source, "hom": hom,
        "hom_ref": hom_ref if isinstance(hom_ref, str) else None,
    }


def build(rows: list[tuple], name: str, report: dict, workers: int) -> list[dict]:
    with Pool(workers) as pool:
        results = pool.map(_process, rows, chunksize=2000)
    kept, reasons, by_source = [], collections.Counter(), collections.defaultdict(collections.Counter)
    for row, r in zip(rows, results):
        if isinstance(r, str):
            reasons[r] += 1
            by_source[row[2]][r] += 1
        else:
            kept.append(r)
            by_source[row[2]]["kept"] += 1
    report[name] = {"input": len(rows), "kept": len(kept), "rejected": dict(reasons),
                    "by_source": {k: dict(v) for k, v in by_source.items()}}
    print(name, report[name]["input"], "->", len(kept), dict(reasons))
    return kept


def write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-size", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    cols = ["Grapheme", "Mapped Phoneme", "Source", "Homograph Grapheme", "Mapped Homograph Phoneme"]
    split = RAW / "homorich/recommended_split"
    tr = pd.read_csv(split / "train_set.csv", usecols=cols, low_memory=False)
    te = pd.read_csv(split / "test_set.csv", usecols=cols, low_memory=False)
    sb = pd.read_csv(RAW / "sentencebench.csv").rename(columns={"homograph word": "hom"})

    # --- leakage control: nothing evaluated may be trained on -----------------
    held_out = {normalize(g) for g in te.Grapheme.dropna()} | {normalize(g) for g in sb.grapheme}
    tr["_norm"] = tr.Grapheme.fillna("").map(normalize)
    n0 = len(tr)
    leaked = tr._norm.isin(held_out)
    tr = tr[~leaked].drop_duplicates("_norm")
    report["leakage"] = {"train_rows": n0, "removed_overlap_with_test_or_sentencebench": int(leaked.sum()),
                         "removed_duplicates": int(n0 - leaked.sum() - len(tr))}
    print("leakage:", report["leakage"])

    te = te.assign(_norm=te.Grapheme.fillna("").map(normalize)).drop_duplicates("_norm")

    tr_items = build(list(tr[cols].itertuples(index=False, name=None)), "train_all", report, args.workers)
    te_items = build(list(te[cols].itertuples(index=False, name=None)), "test_homorich", report, args.workers)

    # SentenceBench: reference given in R1; convert to canonical R2. Never filtered
    # by the aligner (labels unused) so the benchmark stays complete.
    sb_items = []
    for r in sb.itertuples():
        hom_ref = r1_to_r2(r.pronunciation) if isinstance(r.pronunciation, str) else None
        sb_items.append({"g": normalize(r.grapheme), "ref": canonical_r2(r1_to_r2(r.phoneme)),
                         "source": f"sentencebench/{r.dataset}",
                         "hom": normalize(r.hom) if isinstance(r.hom, str) else None, "hom_ref": hom_ref})

    random.Random(args.seed).shuffle(tr_items)
    dev, train = tr_items[: args.dev_size], tr_items[args.dev_size:]

    label_counts = collections.Counter(l for it in train for l in it["labels"])
    labels = [l for l, _ in label_counts.most_common()]
    ez_rate = sum(sum(it["ezafe"]) for it in train) / sum(len(it["ref"].split()) for it in train)
    report["label_vocab"] = len(labels)
    report["ezafe_per_word_train"] = round(ez_rate, 4)

    write_jsonl(OUT / "train.jsonl", train)
    write_jsonl(OUT / "dev.jsonl", dev)
    write_jsonl(OUT / "test_homorich.jsonl", te_items)
    write_jsonl(OUT / "sentencebench.jsonl", sb_items)
    (OUT / "labels.json").write_text(json.dumps({"labels": labels, "counts": label_counts}, ensure_ascii=False))
    (OUT / "prep_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"train={len(train)} dev={len(dev)} test={len(te_items)} sb={len(sb_items)} labels={len(labels)}")


if __name__ == "__main__":
    main()
