"""Reproduce the official LAFAN1 zero-velocity and interpolation baselines (sanity check of
the data pipeline and metric code), and add physical-plausibility diagnostics."""
import argparse
import json

import torch

from motion_tokens import baselines, lafan, metrics
from motion_tokens import quat as Q

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/lafan1.npz")
ap.add_argument("--out", default="runs/baselines.json")
a = ap.parse_args()

d = lafan.load(a.data)
stats = lafan.train_stats(d)
test = lafan.benchmark_windows(d, lafan.TEST_ACTORS, 65, 40)
off = torch.from_numpy(d.skeleton.offsets)
P = d.skeleton.parents
print("test windows:", test["root"].shape[0])
res = {}
for n in (5, 15, 30, 45):
    s, e = lafan.N_PAST, lafan.N_PAST + n
    gt_r, gt_q = test["root"][:, s:e], test["lq"][:, s:e]
    for name, fn in (("zerov", baselines.zero_velocity), ("interp", baselines.interpolation)):
        r, q = fn(test["root"], test["lq"], lafan.N_PAST, n)
        sc = metrics.benchmark_scores(q, r, gt_q, gt_r, off, P, stats)
        sc_c = metrics.benchmark_scores(q, r, gt_q, gt_r, off, P, stats, ref_lq=test["lq"][:, s - 1])
        assert all(abs(sc[k] - sc_c[k]) < 1e-9 for k in sc), (sc, sc_c)
        res[f"{name}/{n}"] = sc
        print(f"{name:7s} n={n:2d}  " + "  ".join(f"{k} {v:.4f}" for k, v in sc.items()))
json.dump(res, open(a.out, "w"), indent=1)
