"""Analyze blind A/B listening-test votes collected by the service (/listen).

Reports preference for the treatment (tagger front-end) over control (espeak),
with a Wilson 95% interval and an exact two-sided sign test (ties excluded).

    python scripts/07_analyze_listening.py --db runs/listening.sqlite
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path

from scipy.stats import binomtest


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="runs/listening.sqlite")
    args = ap.parse_args()
    rows = sqlite3.connect(args.db).execute("SELECT rater, item, left_arm, choice FROM votes").fetchall()
    wins = losses = ties = 0
    for _, _, left, choice in rows:
        if choice == "same":
            ties += 1
            continue
        winner = left if choice == "left" else ("treatment" if left == "control" else "control")
        wins += winner == "treatment"
        losses += winner == "control"
    n = wins + losses
    lo, hi = wilson(wins, n)
    res = {
        "votes": len(rows), "raters": len({r[0] for r in rows}),
        "treatment_wins": wins, "control_wins": losses, "ties": ties,
        "treatment_preference": round(wins / n, 3) if n else None,
        "wilson95": [round(lo, 3), round(hi, 3)],
        "sign_test_p": binomtest(wins, n, 0.5).pvalue if n else None,
    }
    Path("runs/listening_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
