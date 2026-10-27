"""Label processed HOT3D clips and build the tile-task dataset.

    data/hot3d/clips/<clip>/labels_<method>.npz    per-clip labels (cached)
    data/tasks/splits.json                         participant -> split
    data/tasks/tiles/<id>.jpg                      2x2 frame tiles
    data/tasks/{train,val,test}.jsonl              prompts, answers, reward targets

Splits are by participant, so val/test people never appear in training.
`--method wrist` builds the dataset from the baseline labels instead of
hand-mesh contact (see sensor_vlm.labels), for the label-quality comparison.
`--single-split NAME` puts every clip into one split (an extra test set, e.g.
Quest 3 clips from people absent from the Aria data) and leaves the
participant splits alone.
Also reports what constant answers score on val, the floor a model must beat.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sensor_vlm import labels, tasks
from sensor_vlm.evaluate import constant_baselines

def assign_splits(clips_per_participant: Counter, frac=(0.8, 0.1, 0.1), seed=0) -> dict[str, str]:
    """Shuffle participants, then fill test and val up to their share of clips; the rest is train."""
    people = sorted(clips_per_participant)
    random.Random(seed).shuffle(people)
    total = sum(clips_per_participant.values())
    split, filled = {}, Counter()
    for p in people:
        for name, f in (("test", frac[2]), ("val", frac[1])):
            if filled[name] + clips_per_participant[p] <= f * total * 1.25 and filled[name] < f * total:
                split[p] = name
                filled[name] += clips_per_participant[p]
                break
        else:
            split[p] = "train"
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/hot3d"))
    ap.add_argument("--models", type=Path, default=Path("data/hot3d/object_models"))
    ap.add_argument("--single-split", default=None)
    ap.add_argument("--out", type=Path, default=Path("data/tasks"))
    ap.add_argument("--method", choices=["contact", "wrist"], default="contact")
    ap.add_argument("--relabel", action="store_true")
    args = ap.parse_args()

    clips = sorted(p for p in (args.data / "clips").iterdir() if (p / "meta.json").exists())
    index = labels.SurfaceIndex(args.models)
    held = {}
    for c in tqdm(clips, desc="labels"):
        f = c / f"labels_{args.method}.npz"
        if args.relabel or not f.exists():
            np.savez_compressed(f, **labels.label_clip(c, index, args.method))
        held[c.name] = np.load(f)["held"]

    participants = {c.name: json.loads((c / "meta.json").read_text())["participant_id"] for c in clips}
    (args.out / "tiles").mkdir(parents=True, exist_ok=True)
    if args.single_split:
        split_of = {p: args.single_split for p in set(participants.values())}
        by_split = {args.single_split: []}
    else:
        split_of = assign_splits(Counter(participants.values()))
        (args.out / "splits.json").write_text(json.dumps({"labels": args.method, "participants": split_of}, indent=1))
        by_split = {"train": [], "val": [], "test": []}
    for c in tqdm(clips, desc="samples"):
        rng = random.Random(c.name)
        for s in tasks.build_clip_samples(c, held[c.name], labels.object_names(c), rng):
            tile = args.out / "tiles" / f"{s['id']}.jpg"
            if not tile.exists():
                tasks.render_tile(c, s["frames"]).save(tile, quality=90)
            s["image"] = f"tiles/{s['id']}.jpg"
            s["labels"] = args.method
            by_split[split_of[participants[c.name]]].append(s)

    for name, samples in by_split.items():
        with open(args.out / f"{name}.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

    print(f"clips {len(clips)}, participants {len(split_of)}")
    for name, samples in by_split.items():
        people = sorted({s["participant"] for s in samples})
        a = [s["answer"] for s in samples]
        print(f"{name:5s} samples {len(samples):5d} | people {len(people)} | object none {np.mean([x['object'] == 'none' for x in a]):.1%} "
              f"| point unscorable {np.mean([s['point_mask'] is None and x['object'] != 'none' for s, x in zip(samples, a)]):.1%} "
              f"| bits held {np.mean([x['left'] + x['right'] for x in a]):.1%}")
    report = args.single_split or "val"
    print(f"constant-answer rewards on {report}:")
    for k, v in constant_baselines(by_split[report]).items():
        print(f"  {v:.3f}  {k}")


if __name__ == "__main__":
    main()
