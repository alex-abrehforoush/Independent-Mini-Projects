"""LAFAN1 loading and the official in-betweening benchmark protocol (Harvey et al. 2020).

Protocol (mirrors ubisoft-laforge-animation-dataset/lafan1/{extract,benchmarks}.py):
  * train actors subject1-4, test actor subject5;
  * normalisation stats from train windows of 50 frames every 20 frames;
  * test windows of 65 frames every 40 frames (2232 windows);
  * each window is centred on its mean root xz and rotated so that the last of the
    10 context frames faces +x;
  * transition of n frames sits between 10 context frames and the target frame.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch

from . import quat as Q
from .bvh import Skeleton, read_bvh

TRAIN_ACTORS = ("subject1", "subject2", "subject3", "subject4")
TEST_ACTORS = ("subject5",)
FEET = (3, 4, 7, 8)  # LeftFoot, LeftToe, RightFoot, RightToe
N_PAST = 10


@dataclass
class Sequence:
    name: str  # e.g. "walk1"
    subject: str  # e.g. "subject5"
    root_pos: np.ndarray  # (T, 3) float32, cm, y up
    local_q: np.ndarray  # (T, J, 4) float32, discontinuities removed

    @property
    def theme(self) -> str:
        return self.name.rstrip("0123456789")


@dataclass
class Lafan:
    skeleton: Skeleton
    sequences: list[Sequence]

    def select(self, actors) -> list[Sequence]:
        return [s for s in self.sequences if s.subject in actors]


def build_cache(bvh_dir: str, out_path: str) -> None:
    files = sorted(glob.glob(os.path.join(bvh_dir, "*.bvh")))
    arrays, meta, skel = {}, [], None
    for k, f in enumerate(files):
        b = read_bvh(f)
        name, subject = os.path.basename(f)[:-4].split("_")
        if skel is None:
            skel = b.skeleton
        else:
            assert b.skeleton.names == skel.names and b.skeleton.parents == skel.parents
            assert np.allclose(b.skeleton.offsets[1:], skel.offsets[1:], atol=1e-3), f
        arrays[f"root_{k}"] = b.root_pos.astype(np.float32)
        arrays[f"quat_{k}"] = b.local_quats().astype(np.float32)
        meta.append((name, subject))
    np.savez(out_path, names=np.array(skel.names), parents=np.array(skel.parents),
             offsets=skel.offsets.astype(np.float32), meta=np.array(meta), **arrays)


def load(path: str) -> Lafan:
    z = np.load(path)
    skel = Skeleton([str(n) for n in z["names"]], [int(p) for p in z["parents"]], z["offsets"].astype(np.float64))
    seqs = [Sequence(str(n), str(s), z[f"root_{k}"], z[f"quat_{k}"]) for k, (n, s) in enumerate(z["meta"])]
    return Lafan(skel, seqs)


def local_positions(root_pos: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """(…,T,3) root positions + (J,3) offsets -> (…,T,J,3) FK input."""
    off = offsets.to(root_pos).expand(*root_pos.shape[:-1], *offsets.shape).clone()
    off[..., 0, :] = root_pos
    return off


def benchmark_windows(data: Lafan, actors, window: int, offset: int, n_past: int = N_PAST,
                      dtype=torch.float64) -> dict:
    """Windows exactly as get_lafan1_set: centred on mean root xz, facing +x at frame n_past-1."""
    roots, quats, names = [], [], []
    for s in data.select(actors):
        i = 0
        while i + window < len(s.root_pos):
            roots.append(s.root_pos[i : i + window])
            quats.append(s.local_q[i : i + window])
            names.append(s.name)
            i += offset
    root = torch.from_numpy(np.stack(roots)).to(dtype)
    lq = torch.from_numpy(np.stack(quats)).to(dtype)
    root = root.clone()
    root[..., [0, 2]] -= root[..., [0, 2]].mean(1, keepdim=True)
    root, lq = rotate_at_frame(root, lq, n_past - 1)
    return {"root": root, "lq": lq, "names": names}


def rotate_at_frame(root: torch.Tensor, lq: torch.Tensor, frame: int):
    """Rotate about y (around the origin) so the root faces +x at `frame`."""
    yrot = Q.heading_quat(lq[:, frame : frame + 1, 0])  # (B,1,4)
    inv = Q.qinv(yrot)
    root = Q.qrot(inv, root)
    lq = lq.clone()
    lq[..., 0, :] = Q.qmul(inv, lq[..., 0, :])
    return root, lq


def train_stats(data: Lafan, offsets: torch.Tensor | None = None) -> dict:
    """Mean/std of normalised global positions on train windows (50 frames, offset 20)."""
    w = benchmark_windows(data, TRAIN_ACTORS, 50, 20)
    off = torch.from_numpy(data.skeleton.offsets) if offsets is None else offsets
    _, gp = Q.fk(w["lq"], local_positions(w["root"], off.to(w["root"])), data.skeleton.parents)
    flat = gp.reshape(gp.shape[0], gp.shape[1], -1)  # (B,T,J*3)
    return {"x_mean": flat.mean((0, 1)), "x_std": flat.std((0, 1), unbiased=False)}
