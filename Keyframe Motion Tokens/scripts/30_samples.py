"""Qualitative samples: GIFs, filmstrips and BVH files for a few held-out (subject 5) windows.

For each window: ground truth, SLERP, and every generator given (greedy decoding). Keyframe
generators flag their generated keys in red. BVH files open in Maya / Blender / MotionBuilder."""
import argparse
import os
import sys
from importlib import import_module

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from motion_tokens import baselines, bvh, lafan, render
from motion_tokens import quat as Q
from motion_tokens.generator import N_CTX

ev = import_module("21_eval_generator")

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--labels", default=None, help="comma-separated display names")
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--themes", default="dance,run,fight,walk,jumps")
ap.add_argument("--out", default="docs/samples")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
d = lafan.load("data/lafan1.npz")
P = d.skeleton.parents
off = torch.from_numpy(d.skeleton.offsets).float().to(dev)
w = lafan.benchmark_windows(d, lafan.TEST_ACTORS, 101, 40, dtype=torch.float32)
rng = np.random.default_rng(a.seed)
picks = []
for th in a.themes.split(","):
    cand = [i for i, nm in enumerate(w["names"]) if nm.rstrip("0123456789") == th]
    if cand:
        picks.append((th, int(rng.choice(cand))))
root_w, lq_w = w["root"][[i for _, i in picks]].to(dev), w["lq"][[i for _, i in picks]].to(dev)
n, L = a.n, N_CTX + a.n + 1
shift = root_w[:, N_CTX - 1] * root_w.new_tensor([1.0, 0.0, 1.0])
ra, qa = root_w[:, :L] - shift[:, None], lq_w[:, :L]

clips = {"ground truth": (root_w[:, :L], lq_w[:, :L], None)}
r, q = baselines.interpolation(root_w[:, :L], lq_w[:, :L], N_CTX, n)
clips["SLERP"] = (torch.cat([root_w[:, :N_CTX], r, root_w[:, N_CTX + n:L]], 1),
                  torch.cat([lq_w[:, :N_CTX], q, lq_w[:, N_CTX + n:L]], 1), None)
labels = a.labels.split(",") if a.labels else a.runs
for run, lab in zip(a.runs, labels):
    cfg, tok, model, cs = ev.load_generator(run, dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    keys = None
    if cfg["kind"] == "kf":
        r, q, _, keys = ev.generate_kf(model, tok, ra, qa, n, 0.0, gen, return_keys=True, cont_stats=cs)
        keys = [{N_CTX - 1 + k for k in ks} for ks in keys]
    else:
        r, q, _ = ev.generate_fr(model, tok, ra, qa, n, 0.0, gen, cs)
    r = r + shift[:, None]
    clips[lab] = (torch.cat([root_w[:, :N_CTX], r, root_w[:, N_CTX + n:L]], 1),
                  torch.cat([lq_w[:, :N_CTX], Q.qnormalize(q), lq_w[:, N_CTX + n:L]], 1), keys)

for k, (th, wi) in enumerate(picks):
    pos, titles, kf = [], [], []
    for lab, (r, q, keys) in clips.items():
        _, gp = Q.fk(q[k], lafan.local_positions(r[k], off), P)
        pos.append(gp.cpu().numpy())
        titles.append(lab)
        kf.append(keys[k] if keys is not None else set())
        bvh.write_bvh(os.path.join(a.out, f"{th}_{wi}_{lab.replace(' ', '_')}.bvh"), d.skeleton,
                      r[k].cpu().double().numpy(), q[k].cpu().double().numpy())
    render.animate(pos, titles, P, os.path.join(a.out, f"{th}_{wi}_n{n}.gif"), key_frames=kf,
                   highlight=(N_CTX, N_CTX + n))
    frames = list(range(N_CTX - 1, N_CTX + n + 1, max(1, (n + 2) // 8)))
    render.filmstrip(pos, titles, P, os.path.join(a.out, f"{th}_{wi}_n{n}.png"), frames, kf)
    print("wrote", th, wi)
