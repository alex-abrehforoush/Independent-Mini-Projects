"""Malayer TTS service: A/B-routed synthesis, streaming, guardrails, drift, listening test.

    MALAYER_RUN=runs/g2p_base uvicorn malayer.serve.app:app --port 8000

Endpoints
    POST /v1/tts            {"text", "session_id"?, "arm"?} -> audio/wav (+ X-Arm, X-Guardrails headers)
    WS   /v1/stream         send {"text": fragment} ... {"end": true}; receive audio chunks (PCM16, base64)
    GET  /metrics           Prometheus metrics (latency, RTF, guardrails, drift, experiment p-values)
    GET  /dashboard         JSON snapshot: drift report, guardrail rates, experiment status
    GET  /listen            blind A/B listening test;  POST /listen/vote
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

from malayer.g2p.data import read_jsonl
from malayer.g2p.infer import Phonemizer
from malayer.serve.experiment import MSPRT, assign_arm
from malayer.serve.guardrails import check_audio
from malayer.serve.monitor import FEATURES, DriftMonitor
from malayer.serve.streaming import StreamingSession
from malayer.tts.frontend import PersianFrontend
from malayer.tts.synth import Synthesizer

RUN = Path(os.environ.get("MALAYER_RUN", "runs/g2p_base"))
ARMS = json.loads(os.environ.get("MALAYER_ARMS", '{"control": 0.5, "treatment": 0.5}'))
LOOKAHEAD = int(os.environ.get("MALAYER_LOOKAHEAD", "1"))
DB_PATH = Path(os.environ.get("MALAYER_DB", "runs/listening.sqlite"))

REQS = Counter("tts_requests_total", "Requests", ["arm", "endpoint"])
LAT = Histogram("tts_latency_seconds", "Latency by stage", ["arm", "stage"],
                buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2))
RTF = Histogram("tts_rtf", "Real-time factor", ["arm"], buckets=(0.02, 0.05, 0.1, 0.2, 0.5, 1, 2))
GUARD = Counter("tts_guardrail_violations_total", "Guardrail violations", ["arm", "kind"])
DRIFT = Gauge("frontend_drift_psi", "PSI of input/model features vs. reference", ["feature"])
EXP_P = Gauge("experiment_always_valid_p", "mSPRT always-valid p-value", ["metric"])

app = FastAPI(title="Malayer Persian TTS")
_phonemizer = Phonemizer.from_dir(RUN)
SYNTH = {
    "control": Synthesizer(PersianFrontend()),
    "treatment": Synthesizer(PersianFrontend(_phonemizer, lookahead=None)),
}
EXPERIMENTS = {"guardrail_pass": MSPRT(tau=0.05), "listener_preference": MSPRT(tau=0.2, min_n=10, baseline=0.5)}
GUARD_COUNTS: dict[str, dict[str, int]] = {}  # arm -> violation kind -> count (dashboard copy)
_PENDING_PAIRS: dict[str, tuple[int, str]] = {}  # blind token -> (item, arm on the left)


def _drift_reference() -> DriftMonitor:
    path = RUN / "drift_reference.json"
    if not path.exists():
        dev = read_jsonl("data/g2p/dev.jsonl")[:2000]
        feats = [_phonemizer.analyze(it["g"]) for it in dev]
        path.write_text(json.dumps({f: [x[f] for x in feats] for f in FEATURES}))
    return DriftMonitor.from_file(path)


DRIFT_MON = _drift_reference()


def _observe(text: str, arm: str, endpoint: str) -> float:
    REQS.labels(arm, endpoint).inc()
    g = SYNTH["treatment"].frontend.prepare(text)
    feats = _phonemizer.analyze(g)
    DRIFT_MON.observe(feats)
    for f, r in DRIFT_MON.report().items():
        DRIFT.labels(f).set(r["psi"])
    return feats["unknown_char_ratio"]


def _count_violations(arm: str, violations: list[str]) -> None:
    counts = GUARD_COUNTS.setdefault(arm, {})
    for v in violations:
        GUARD.labels(arm, v).inc()
        counts[v] = counts.get(v, 0) + 1


def _wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, (np.clip(audio, -1, 1) * 32767).astype(np.int16), sr, format="WAV")
    return buf.getvalue()


class TTSRequest(BaseModel):
    text: str
    session_id: str | None = None
    arm: str | None = None  # force an arm (debugging / listening tests)


@app.post("/v1/tts")
def tts(req: TTSRequest) -> Response:
    arm = req.arm or assign_arm(req.session_id or str(random.random()), ARMS)
    unk = _observe(req.text, arm, "tts")
    t0 = time.perf_counter()
    res = SYNTH[arm].synthesize(req.text)
    LAT.labels(arm, "frontend").observe(res.frontend_ms / 1000)
    LAT.labels(arm, "acoustic").observe(res.acoustic_ms / 1000)
    LAT.labels(arm, "total").observe(time.perf_counter() - t0)
    RTF.labels(arm).observe(res.rtf)
    n_phones = sum(ch.isalpha() for ch in res.ipa)
    report = check_audio(res.audio, res.sample_rate, n_phones, unk)
    _count_violations(arm, report.violations)
    exp = EXPERIMENTS["guardrail_pass"]
    exp.add("a" if arm == "control" else "b", float(report.passed))
    EXP_P.labels("guardrail_pass").set(exp.p_value)
    headers = {"X-Arm": arm, "X-Guardrails": "pass" if report.passed else ",".join(report.violations),
               "X-RTF": f"{res.rtf:.3f}"}
    return Response(_wav_bytes(res.audio, res.sample_rate), media_type="audio/wav", headers=headers)


@app.websocket("/v1/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    params = ws.query_params
    arm = params.get("arm") or assign_arm(params.get("session_id", str(random.random())), ARMS)
    synth = SYNTH[arm]
    session = StreamingSession(synth, _phonemizer if arm == "treatment" else None, lookahead=LOOKAHEAD,
                               first_phrase_words=int(params.get("first_phrase_words", 5)))
    REQS.labels(arm, "stream").inc()
    first = True
    text_so_far = []
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if "text" in msg:
                text_so_far.append(msg["text"])
                chunks = session.feed(msg["text"])
            elif msg.get("end"):
                chunks = session.finish()
            else:
                continue
            for c in chunks:
                if first:
                    LAT.labels(arm, "ttfa").observe(c.t_emit)
                    first = False
                report = check_audio(c.audio, c.sample_rate, sum(ch.isalpha() for ch in c.ipa))
                _count_violations(arm, report.violations)
                pcm = (np.clip(c.audio, -1, 1) * 32767).astype(np.int16).tobytes()
                await ws.send_json({"type": "audio", "arm": arm, "text": c.text, "ipa": c.ipa,
                                    "sample_rate": c.sample_rate, "t_emit": round(c.t_emit, 4),
                                    "guardrails": report.violations, "pcm16": base64.b64encode(pcm).decode()})
            if msg.get("end"):
                _observe("".join(text_so_far), arm, "stream_text")
                await ws.send_json({"type": "done"})
                break
    except WebSocketDisconnect:
        pass


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/dashboard")
def dashboard() -> JSONResponse:
    return JSONResponse({"drift": DRIFT_MON.report(), "experiments": {k: v.summary() for k, v in EXPERIMENTS.items()},
                         "guardrail_violations": GUARD_COUNTS, "arms": ARMS, "run": str(RUN)})


# ---------------------------------------------------------------- listening test
def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS votes (ts REAL, rater TEXT, item INTEGER, left_arm TEXT, choice TEXT)")
    return con


def _listen_items(n: int = int(os.environ.get("MALAYER_LISTEN_ITEMS", "40"))) -> list[dict]:
    """SentenceBench homograph sentences where the two arms send *different* phonemes
    (identical pairs would only waste rater time), sampled reproducibly."""
    items = [it for it in read_jsonl("data/g2p/sentencebench.jsonl") if it["source"] == "sentencebench/homograph"]
    differ = [it for it in items if SYNTH["control"].frontend.ipa(it["g"]) != SYNTH["treatment"].frontend.ipa(it["g"])]
    random.Random(0).shuffle(differ)
    return differ[:n]


_LISTEN_ITEMS = _listen_items()


@app.get("/listen/item")
def listen_item(rater: str = "anon") -> JSONResponse:
    done = {r[0] for r in _db().execute("SELECT item FROM votes WHERE rater=?", (rater,))}
    todo = [i for i in range(len(_LISTEN_ITEMS)) if i not in done]
    if not todo:
        return JSONResponse({"done": True})
    i = random.choice(todo)
    left = random.choice(["control", "treatment"])
    right = "treatment" if left == "control" else "control"
    text = _LISTEN_ITEMS[i]["g"]
    clips = {}
    for side, arm in (("left", left), ("right", right)):
        res = SYNTH[arm].synthesize(text)
        clips[side] = base64.b64encode(_wav_bytes(res.audio, res.sample_rate)).decode()
    token = uuid.uuid4().hex
    _PENDING_PAIRS[token] = (i, left)
    return JSONResponse({"token": token, "text": text, "left": clips["left"], "right": clips["right"],
                         "remaining": len(todo)})


class Vote(BaseModel):
    rater: str
    token: str
    choice: str  # "left" | "right" | "same"


@app.post("/listen/vote")
def listen_vote(v: Vote) -> JSONResponse:
    if v.token not in _PENDING_PAIRS or v.choice not in ("left", "right", "same"):
        return JSONResponse({"ok": False}, status_code=400)
    item, left = _PENDING_PAIRS.pop(v.token)
    con = _db()
    con.execute("INSERT INTO votes VALUES (?,?,?,?,?)", (time.time(), v.rater, item, left, v.choice))
    con.commit()
    if v.choice != "same":
        winner = left if v.choice == "left" else ("treatment" if left == "control" else "control")
        exp = EXPERIMENTS["listener_preference"]  # one-sample: P(prefer treatment) vs 0.5
        exp.add("b", float(winner == "treatment"))
        EXP_P.labels("listener_preference").set(exp.p_value)
    return JSONResponse({"ok": True})


@app.get("/listen")
def listen_page() -> HTMLResponse:
    return HTMLResponse((Path(__file__).parent / "listen.html").read_text())
