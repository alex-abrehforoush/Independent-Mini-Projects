"""Stream HOT3D-Clips (Aria) from Hugging Face into the compact processed layout.

Each clip tar (~100 MB) is downloaded, converted by `sensor_vlm.hot3d.process_clip`
(~1 MB kept) and deleted, so peak disk use stays at a few hundred MB.
Resumable: clips with a meta.json are skipped. Also fetches the object models,
clip split definitions and the dataset license.

    python scripts/00_fetch_hot3d.py                # all 1516 train_aria clips
    python scripts/00_fetch_hot3d.py --limit 20     # quick subset
    python scripts/00_fetch_hot3d.py --split train_quest3 --ids quest3_ids.json \
        --out data/hot3d_quest3 --skip-extras        # chosen clips only
"""

import argparse
import json
import shutil
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm

from sensor_vlm.hot3d import process_clip

REPO = "bop-benchmark/hot3d"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/"


def download(path: str, dest: Path, retries: int = 3) -> None:
    part = dest.with_suffix(".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(URL + path, timeout=60) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            part.rename(dest)
            return
        except Exception:
            part.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def fetch_clip(path: str, out: Path, tmp: Path) -> str:
    name = Path(path).stem
    clip_dir = out / "clips" / name
    if (clip_dir / "meta.json").exists():
        return "skipped"
    tar = tmp / f"{name}.tar"
    try:
        download(path, tar)
        if clip_dir.exists():
            shutil.rmtree(clip_dir)  # partial output from an interrupted run
        process_clip(tar, clip_dir)
        return "done"
    finally:
        tar.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train_aria")
    ap.add_argument("--out", type=Path, default=Path("data/hot3d"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", type=Path, default=None, help="JSON list of clip ids to fetch")
    ap.add_argument("--skip-extras", action="store_true", help="don't fetch object models / splits / license")
    args = ap.parse_args()

    tmp = args.out / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    if not args.skip_extras:
        snapshot_download(REPO, repo_type="dataset", local_dir=args.out,
                          allow_patterns=["object_models/*", "clip_splits.json", "hot3d_dataset_license_agreement.pdf"])

    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    clips = sorted(f for f in files if f.startswith(f"{args.split}/") and f.endswith(".tar"))
    if args.ids:
        wanted = {f"clip-{i:06d}" for i in json.loads(args.ids.read_text())}
        clips = [c for c in clips if Path(c).stem in wanted]
    clips = clips[: args.limit]
    counts = {"done": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(args.workers) as pool, open(args.out / "failed.txt", "a") as failed:
        futures = {pool.submit(fetch_clip, c, args.out, tmp): c for c in clips}
        bar = tqdm(as_completed(futures), total=len(futures), unit="clip")
        for fut in bar:
            try:
                counts[fut.result()] += 1
            except Exception:
                counts["failed"] += 1
                failed.write(f"{futures[fut]}\n{traceback.format_exc()}\n")
                failed.flush()
            bar.set_postfix(counts)
    print(counts)


if __name__ == "__main__":
    main()
