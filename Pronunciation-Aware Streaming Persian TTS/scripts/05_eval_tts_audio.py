"""Audio-level pronunciation evaluation of complete TTS systems on SentenceBench.

Every system uses the same Piper voice; only the front-end differs:
    espeak-ng (stock)  |  tagger, full sentence  |  tagger, streaming k=1

For every written word where the gold reading and espeak's reading differ, the
phoneme recognizer's minimal-pair forced choice decides which of the two the
*audio* realizes. We report, per system:
    gold_rate      share of contested words the audio pronounces the gold way
    intent_rate    share of words the front-end changed that the voice actually
                   realized as intended (does the acoustic model follow edits?)
    greedy_PER     free phoneme recognition vs. gold (noisier, reported for context)
    RTF            synthesis real-time factor (CPU)

    python scripts/05_eval_tts_audio.py --run runs/g2p_base
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from persian_tts.eval.phoneme_ctc import PhonemeCTC, r2_symbols, same_for_scoring
from persian_tts.eval.readings import espeak_word_readings
from persian_tts.g2p.baselines import EspeakG2P
from persian_tts.g2p.data import read_jsonl
from persian_tts.g2p.infer import Phonemizer
from persian_tts.g2p.metrics import edit_distance
from persian_tts.phonemes import split_ezafe
from persian_tts.tts.frontend import PersianFrontend
from persian_tts.tts.synth import Synthesizer

_PUNCT = ".,!?؟،؛:«»()\"'"


def contested_kind(gold: str, alt: str, is_hom: bool) -> str:
    if is_hom:
        return "homograph"
    gb, ge = split_ezafe(gold)
    ab, ae = split_ezafe(alt)
    if ge != ae and same_for_scoring(gb, ab):
        return "ezafe"
    return "other vowel/consonant"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/g2p_base")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--samples-dir", default="docs/samples")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = read_jsonl("data/g2p/sentencebench.jsonl")
    if args.limit:
        items = items[: args.limit]
    ph = Phonemizer.from_dir(args.run, device="cpu")
    systems = {
        "espeak-ng (stock)": Synthesizer(PersianFrontend()),
        "tagger, full sentence": Synthesizer(PersianFrontend(ph, lookahead=None)),
        "tagger, streaming k=1": Synthesizer(PersianFrontend(ph, lookahead=1)),
    }
    rec = PhonemeCTC(device=args.device)
    esp = EspeakG2P()

    stats = {s: {"gold": collections.defaultdict(lambda: [0, 0]), "intent": [0, 0], "edits": 0, "ref_len": 0,
                 "audio_s": 0.0, "compute_s": 0.0} for s in systems}
    per_sentence = []
    showcase = []
    paired: dict[tuple, dict[str, float]] = collections.defaultdict(dict)
    for it in tqdm(items, desc="sentences"):
        g = it["g"]
        cores = [t.strip(_PUNCT) for t in g.split(" ") if t.strip(_PUNCT)]
        gold = it["ref"].split()
        alt = espeak_word_readings(g, esp)
        comparable = len(gold) == len(cores) == len(alt)
        row = {"g": g, "source": it["source"], "systems": {}}
        for sname, synth in systems.items():
            res = synth.synthesize(g)
            st = stats[sname]
            st["audio_s"] += len(res.audio) / res.sample_rate
            st["compute_s"] += (res.frontend_ms + res.acoustic_ms) / 1000
            clp = rec.class_logprobs(rec.token_logprobs(res.audio, res.sample_rate))
            hyp = rec.greedy_r2(clp)
            ref = "".join(r2_symbols(it["ref"]))
            st["edits"] += edit_distance(hyp, ref)
            st["ref_len"] += len(ref)
            srow = {"ipa": res.ipa, "contested": []}
            if comparable:
                for k, (gw, aw, core) in enumerate(zip(gold, alt, cores)):
                    if same_for_scoring(gw, aw):
                        continue
                    kind = contested_kind(gw, aw, is_hom=bool(it.get("hom")) and it["hom"] in core)
                    llr = rec.forced_choice(clp, gold, k, aw)
                    paired[(len(per_sentence), k, kind)][sname] = llr
                    st["gold"][kind][0] += llr > 0
                    st["gold"][kind][1] += 1
                    srow["contested"].append({"word": core, "gold": gw, "espeak": aw, "kind": kind, "llr": round(llr, 2)})
            # intent fidelity: words where this front-end deviates from espeak
            words = synth.frontend.words(g)
            intended = [w.r2 for w in words]
            if comparable and len(intended) == len(alt) and all(intended):
                for k, (iw, aw) in enumerate(zip(intended, alt)):
                    if not same_for_scoring(iw, aw):
                        st["intent"][0] += rec.forced_choice(clp, intended, k, aw) > 0
                        st["intent"][1] += 1
            row["systems"][sname] = srow
            if len(showcase) < args.n_samples * len(systems) and it["source"] == "sentencebench/homograph" and comparable:
                showcase.append((len(per_sentence), sname, res))
        per_sentence.append(row)

    # write showcase audio for the README / samples page
    out_dir = Path(args.samples_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, sname, res in showcase:
        tag = {"espeak-ng (stock)": "espeak", "tagger, full sentence": "tagger", "tagger, streaming k=1": "tagger_k1"}[sname]
        path = out_dir / f"sb{idx:03d}_{tag}.wav"
        sf.write(path, (np.clip(res.audio, -1, 1) * 32767).astype(np.int16), res.sample_rate)
        manifest.append({"file": path.name, "system": sname, "text": per_sentence[idx]["g"], "ipa": res.ipa,
                         "contested": per_sentence[idx]["systems"][sname]["contested"]})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))

    summary = {}
    for sname, st in stats.items():
        gold_tot = [sum(v[0] for v in st["gold"].values()), sum(v[1] for v in st["gold"].values())]
        summary[sname] = {
            "gold_rate_all": round(100 * gold_tot[0] / max(gold_tot[1], 1), 2),
            "gold_rate_by_kind": {k: {"rate": round(100 * a / max(n, 1), 2), "n": n} for k, (a, n) in st["gold"].items()},
            "intent_rate": round(100 * st["intent"][0] / max(st["intent"][1], 1), 2) if st["intent"][1] else None,
            "intent_n": st["intent"][1],
            "greedy_PER_vs_gold": round(100 * st["edits"] / max(st["ref_len"], 1), 2),
            "RTF_cpu": round(st["compute_s"] / max(st["audio_s"], 1e-6), 4),
        }
    # Paired comparison (the criterion validated at 88.8% on synthetic minimal pairs):
    # for each contested word, whose audio gives the gold reading the higher relative likelihood?
    base = "espeak-ng (stock)"
    pairwise = {}
    for sname in systems:
        if sname == base:
            continue
        by_kind = collections.defaultdict(lambda: [0, 0, 0])  # wins, losses, ties
        for (_, _, kind), d in paired.items():
            a, b = d.get(sname), d.get(base)
            if a is None or b is None:
                continue
            for key in (kind, "all"):
                by_kind[key][0] += a > b + 0.5
                by_kind[key][1] += b > a + 0.5
                by_kind[key][2] += abs(a - b) <= 0.5
        pairwise[f"{sname} vs {base}"] = {
            k: {"win": w, "loss": l, "tie": t, "win_rate_excl_ties": round(100 * w / max(w + l, 1), 2)}
            for k, (w, l, t) in by_kind.items()}
    out = Path(args.run) / "eval_tts_audio.json"
    out.write_text(json.dumps({"summary": summary, "pairwise": pairwise, "per_sentence": per_sentence},
                              indent=1, ensure_ascii=False))
    print(json.dumps(pairwise, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
