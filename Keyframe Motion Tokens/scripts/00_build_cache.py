"""Parse the 77 LAFAN1 BVH files once into data/lafan1.npz."""
import argparse
import time

from motion_tokens import lafan

ap = argparse.ArgumentParser()
ap.add_argument("--bvh", default="data/bvh")
ap.add_argument("--out", default="data/lafan1.npz")
a = ap.parse_args()
t0 = time.time()
lafan.build_cache(a.bvh, a.out)
d = lafan.load(a.out)
n = sum(len(s.root_pos) for s in d.sequences)
print(f"{len(d.sequences)} sequences, {n} frames ({n / 30 / 3600:.2f} h), {time.time() - t0:.0f}s")
print("joints:", d.skeleton.names)
