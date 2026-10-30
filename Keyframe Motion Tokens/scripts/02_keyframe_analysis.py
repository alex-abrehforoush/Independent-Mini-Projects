"""Classical warm-up: how well does plain LERP/SLERP through key frames reconstruct test motion,
for uniform vs adaptive key placement at the same key rate?  No learning involved."""
import argparse
import json

import torch

from motion_tokens import keyframes, lafan, metrics
from motion_tokens import quat as Q

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/lafan1.npz")
ap.add_argument("--window", type=int, default=128)
ap.add_argument("--out", default="runs/keyframe_analysis.json")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = lafan.load(a.data)
P = d.skeleton.parents
off = torch.from_numpy(d.skeleton.offsets).float().to(dev)
w = lafan.benchmark_windows(d, lafan.TEST_ACTORS, a.window, a.window // 2, dtype=torch.float32)
root, lq, themes = w["root"].to(dev), w["lq"].to(dev), [n.rstrip("0123456789") for n in w["names"]]
B, T = root.shape[:2]
_, gp = Q.fk(lq, lafan.local_positions(root, off), P)
contacts = metrics.foot_contacts(gp)
print(f"{B} test windows of {T} frames; contact rate {contacts.float().mean():.2f}")


def evaluate(mask):
    r, q = keyframes.interpolate_mask(root, lq, mask)
    _, ip = Q.fk(q, lafan.local_positions(r, off), P)
    err = (ip - gp).norm(dim=-1)  # (B,T,J)
    keys_per_s = mask.float().sum(1) / T * 30
    return {
        "keys_per_s": keys_per_s.mean().item(),
        "mpjpe_cm": err.mean().item(),
        "max_err_cm": err.amax((1, 2)).mean().item(),
        "skate_cm_s": metrics.foot_skate(ip, contacts).mean().item(),
        "jitter": metrics.jitter(ip).mean().item(),
        "per_window_keys_per_s": keys_per_s.tolist(),
        "per_window_mpjpe": err.mean((1, 2)).tolist(),
    }


res = {"gt": {"skate_cm_s": metrics.foot_skate(gp, contacts).mean().item(), "jitter": metrics.jitter(gp).mean().item()},
       "themes": themes, "uniform": {}, "adaptive": {}}
for stride in (2, 3, 4, 6, 8, 12, 16):
    r = evaluate(keyframes.uniform(B, T, stride, dev))
    res["uniform"][stride] = r
    print(f"uniform  stride {stride:2d}: {r['keys_per_s']:5.2f} keys/s  MPJPE {r['mpjpe_cm']:5.2f} cm  max {r['max_err_cm']:5.1f}  skate {r['skate_cm_s']:5.1f}")
for eps in (1, 2, 3, 5, 8, 12, 20):
    r = evaluate(keyframes.adaptive(root, lq, off, P, eps, max_gap=32))
    res["adaptive"][eps] = r
    print(f"adaptive eps {eps:4.1f}: {r['keys_per_s']:5.2f} keys/s  MPJPE {r['mpjpe_cm']:5.2f} cm  max {r['max_err_cm']:5.1f}  skate {r['skate_cm_s']:5.1f}")
print("GT skate", res["gt"])
json.dump(res, open(a.out, "w"))
