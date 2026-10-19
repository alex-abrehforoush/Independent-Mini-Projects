"""Replay synthetic traffic through the service to exercise A/B routing, guardrails,
drift detection and streaming latency. Runs the FastAPI app in-process.

Phases
  1. normal   : formal Persian sentences (HomoRich test)                -> no drift alerts expected
  2. shifted  : colloquial text, Finglish, mixed-script, digit-heavy      -> drift alerts expected
  3. streaming: 3-sentence replies arriving at ~60 chars/s (LLM-like) -> time-to-first-audio,
                compared with waiting for the whole reply and synthesizing it at once

    PTTS_RUN=runs/g2p_base python scripts/06_simulate_traffic.py
"""

from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from persian_tts.g2p.data import read_jsonl
from persian_tts.serve.app import app

CHARS_PER_SEC = 60

SHIFTED_EXTRA = [
    "salam, chetori? emruz miam pishet",
    "man ba iPhone 15 Pro kar mikonam",
    "سلام! امروز با Zoom جلسه داریم ساعت 10:30",
    "قیمت دلار امروز 58,400 تومان شد",
    "این API رو با Python 3.12 نوشتم",
    "ok mersi, bye",
]


def main() -> None:
    rng = random.Random(0)
    client = TestClient(app)
    normal = [it["g"] for it in read_jsonl("data/g2p/test_homorich.jsonl")]
    rng.shuffle(normal)
    informal = pd.read_parquet("data/raw/homorich/data/train-01.parquet")
    informal = informal[informal.Source == "GPTInformal"].Grapheme.dropna().tolist()
    rng.shuffle(informal)

    report: dict = {"phases": {}}

    def run_phase(name: str, texts: list[str]) -> None:
        arms, violations, lat = {"control": 0, "treatment": 0}, 0, []
        for i, t in enumerate(texts):
            t0 = time.perf_counter()
            r = client.post("/v1/tts", json={"text": t, "session_id": f"{name}-{i % 97}"})
            lat.append(time.perf_counter() - t0)
            arms[r.headers["x-arm"]] += 1
            violations += r.headers["x-guardrails"] != "pass"
        snap = client.get("/dashboard").json()
        report["phases"][name] = {"n": len(texts), "arms": arms, "guardrail_violations": violations,
                                  "violations_by_kind_cumulative": snap["guardrail_violations"],
                                  "latency_p50_s": round(statistics.median(lat), 3),
                                  "drift": snap["drift"], "experiments": snap["experiments"]}
        alerts = [f for f, d in snap["drift"].items() if d["alert"]]
        print(f"[{name}] n={len(texts)} arms={arms} violations={violations} drift alerts={alerts}")

    run_phase("normal", normal[:400])
    shifted = informal[:300] + SHIFTED_EXTRA * 15
    rng.shuffle(shifted)
    run_phase("shifted", shifted)

    # Streaming: multi-sentence replies arriving at LLM speed (~60 chars/s ≈ 15-20 tokens/s).
    # Baseline "wait-then-speak": the full reply must arrive before /v1/tts is called.
    sb = [it["g"] for it in read_jsonl("data/g2p/sentencebench.jsonl")]
    replies = [" ".join(rng.sample(sb, 3)) for _ in range(20)]
    stream = {}
    for arm in ("control", "treatment"):
        ttfa, wait_then_speak, arrivals = [], [], []
        for t in replies:
            with client.websocket_connect(f"/v1/stream?arm={arm}") as ws:
                t0 = time.perf_counter()
                first = None
                i = 0
                while i < len(t):
                    n = rng.randint(2, 4)
                    ws.send_text(json.dumps({"text": t[i: i + n]}))
                    i += n
                    time.sleep(n / CHARS_PER_SEC)
                arrival = time.perf_counter() - t0
                arrivals.append(arrival)
                ws.send_text(json.dumps({"end": True}))
                while True:
                    msg = ws.receive_json()
                    if msg["type"] == "audio" and first is None:
                        first = msg["t_emit"]
                    if msg["type"] == "done":
                        break
                ttfa.append(first)
            t1 = time.perf_counter()
            client.post("/v1/tts", json={"text": t, "arm": arm})
            wait_then_speak.append(arrival + time.perf_counter() - t1)
        stream[arm] = {"streaming_ttfa_p50_s": round(statistics.median(ttfa), 3),
                       "wait_then_speak_ttfa_p50_s": round(statistics.median(wait_then_speak), 3),
                       "reply_arrival_p50_s": round(statistics.median(arrivals), 2)}
        print(f"[stream] {arm}: {stream[arm]}")
    report["streaming"] = stream
    Path("runs/serving_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("wrote runs/serving_report.json")


if __name__ == "__main__":
    main()
