"""Download every external asset the project uses (≈3.3 GB total).

    data/raw/homorich/        HomoRich-G2P-Persian (CC0)          sentence-level phonemes
    data/raw/sentencebench.csv SentenceBench (GPL, eval only)      external G2P benchmark
    data/raw/mana/            2 shards of Mana-TTS (CC0)           human recordings for calibration
    models/mana-piper/        Mana Persian Piper voice (MIT)       acoustic model (VITS, ONNX)
    models/w2v2-phoneme/      wav2vec2-xlsr-53-espeak-cv-ft (Apache-2.0) phoneme recognizer
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def main() -> None:
    for f in ["data/train-01.parquet", "recommended_split/train_set.csv", "recommended_split/test_set.csv"]:
        hf_hub_download("MahtaFetrat/HomoRich-G2P-Persian", f, repo_type="dataset", local_dir="data/raw/homorich")
    sb = Path("data/raw/sentencebench.csv")
    if not sb.exists():
        urllib.request.urlretrieve(
            "https://huggingface.co/datasets/MahtaFetrat/SentenceBench/resolve/main/SentenceBench.csv", sb)
    for i in (1, 2):
        hf_hub_download("MahtaFetrat/Mana-TTS", f"dataset/dataset_part_00{i}.parquet", repo_type="dataset",
                        local_dir="data/raw/mana")
    for f in ["fa_IR-mana-medium.onnx", "fa_IR-mana-medium.onnx.json"]:
        hf_hub_download("MahtaFetrat/Mana-Persian-Piper", f, local_dir="models/mana-piper")
    snapshot_download("facebook/wav2vec2-xlsr-53-espeak-cv-ft", local_dir="models/w2v2-phoneme",
                      allow_patterns=["*.json", "pytorch_model.bin", "*.txt"])
    print("done")


if __name__ == "__main__":
    main()
