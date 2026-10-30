"""Minimal BVH reader/writer (root with 6 channels, other joints with 3 rotation channels)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import quat as Q

_CH = {"Xrotation": "x", "Yrotation": "y", "Zrotation": "z"}


@dataclass
class Skeleton:
    names: list[str]
    parents: list[int]
    offsets: np.ndarray  # (J, 3) bone offsets; row 0 unused (root position comes from channels)

    @property
    def n_joints(self) -> int:
        return len(self.names)


@dataclass
class BVH:
    skeleton: Skeleton
    root_pos: np.ndarray  # (T, 3)
    euler_deg: np.ndarray  # (T, J, 3) in channel order
    order: str  # e.g. "zyx"
    frame_time: float

    def local_quats(self) -> np.ndarray:
        e = torch.from_numpy(np.radians(self.euler_deg))
        q = Q.euler_to_quat(e, self.order)
        return Q.remove_discontinuities(q, dim=0).numpy()


def read_bvh(path: str) -> BVH:
    with open(path) as f:
        lines = f.read().splitlines()
    names, parents, offsets = [], [], []
    stack, order, in_end_site, i = [], None, False, 0
    while i < len(lines):
        tok = lines[i].split()
        i += 1
        if not tok:
            continue
        key = tok[0]
        if key in ("ROOT", "JOINT"):
            names.append(tok[1])
            parents.append(stack[-1] if stack else -1)
            offsets.append([0.0, 0.0, 0.0])
            stack.append(len(names) - 1)
        elif key == "End":
            in_end_site = True
        elif key == "OFFSET" and not in_end_site:
            offsets[stack[-1]] = [float(v) for v in tok[1:4]]
        elif key == "CHANNELS":
            rot = [_CH[c] for c in tok[2:] if c in _CH]
            joint_order = "".join(rot)
            if order is None:
                order = joint_order
            elif joint_order != order:
                raise ValueError(f"mixed rotation orders in {path}")
        elif key == "}":
            if in_end_site:
                in_end_site = False
            else:
                stack.pop()
        elif key == "MOTION":
            break
    n_frames = int(lines[i].split()[1])
    frame_time = float(lines[i + 1].split()[2])
    data = np.array(" ".join(lines[i + 2 : i + 2 + n_frames]).split(), dtype=np.float64)
    data = data.reshape(n_frames, -1)
    J = len(names)
    assert data.shape[1] == 3 + 3 * J, (path, data.shape)
    skel = Skeleton(names, parents, np.asarray(offsets, dtype=np.float64))
    return BVH(skel, data[:, :3].copy(), data[:, 3:].reshape(n_frames, J, 3), order, frame_time)


def write_bvh(path: str, skel: Skeleton, root_pos: np.ndarray, local_q: np.ndarray,
              frame_time: float = 1 / 30, order: str = "zyx") -> None:
    """Write local quaternions (T,J,4) + root positions (T,3) as a BVH file."""
    axis_name = {"x": "Xrotation", "y": "Yrotation", "z": "Zrotation"}
    rot_ch = " ".join(axis_name[c] for c in order)
    children = {j: [k for k, p in enumerate(skel.parents) if p == j] for j in range(skel.n_joints)}
    out = ["HIERARCHY"]

    def emit(j: int, depth: int) -> None:
        ind = "\t" * depth
        kind = "ROOT" if skel.parents[j] < 0 else "JOINT"
        out.append(f"{ind}{kind} {skel.names[j]}")
        out.append(f"{ind}{{")
        o = skel.offsets[j] if skel.parents[j] >= 0 else np.zeros(3)
        out.append(f"{ind}\tOFFSET {o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
        if kind == "ROOT":
            out.append(f"{ind}\tCHANNELS 6 Xposition Yposition Zposition {rot_ch}")
        else:
            out.append(f"{ind}\tCHANNELS 3 {rot_ch}")
        if children[j]:
            for c in children[j]:
                emit(c, depth + 1)
        else:
            out.append(f"{ind}\tEnd Site")
            out.append(f"{ind}\t{{")
            out.append(f"{ind}\t\tOFFSET 0.000000 0.000000 0.000000")
            out.append(f"{ind}\t}}")
        out.append(f"{ind}}}")

    emit(0, 0)
    euler = Q.mat_to_euler(Q.quat_to_mat(torch.from_numpy(np.asarray(local_q, dtype=np.float64))), order)
    euler = np.degrees(euler.numpy())  # (T, J, 3)
    T = len(root_pos)
    out += ["MOTION", f"Frames: {T}", f"Frame Time: {frame_time:.6f}"]
    for t in range(T):
        vals = np.concatenate([root_pos[t], euler[t].reshape(-1)])
        out.append(" ".join(f"{v:.6f}" for v in vals))
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
