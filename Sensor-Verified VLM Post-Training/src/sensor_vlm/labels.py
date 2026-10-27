"""Hand-object manipulation labels derived from motion capture.

HOT3D objects rest on the table unless a hand moves them, and the mocap poses
of resting objects are steady (speed p99.9 0.007 m/s, max 0.053 m/s over the
static objects of the first 27 clips). A hand manipulates an object when it
touches it during a contact that also moves the object; touching a resting
object (typing on a keyboard, a hand resting against a mug) is not
manipulation.

Two labelers share that definition and differ in how "touch" is measured:

contact (default)  UmeTrack hand mesh (788 vertices, Meta's hand_tracking_toolkit)
                   against the object surface: a touch is >= TOUCH_VERTS
                   vertices within TOUCH_DIST. The whole touch is labeled, so
                   a grasp starts when the fingers close on the object, before
                   it moves (typically 0.2-0.4 s earlier).
wrist (baseline)   wrist-to-object-surface distance only. While an object moves
                   the nearest wrist is 0.05-0.13 m from it (p1-p99), so a
                   wrist within WRIST_REACH during motion counts. Kept to
                   measure what label quality is worth downstream.

On 534 clips the two agree on 89.5% of hand-frames. Contact adds 9.0% (holds
of an object kept still, and holds of large objects with the wrist beyond
WRIST_REACH; 8 of 8 random cases checked by eye were real holds); the wrist
method adds 1.5%.

Per clip, at 30 fps (h: 0 left, 1 right):
    speed[t, o]          smoothed object speed (m/s)
    moving[t, o]         object o is being moved
    wrist_dist[t, h, o]  wrist-to-object-surface distance (m)
    touch[t, h, o]       hand mesh touches object o (contact method only)
    held[t, h, o]        hand h manipulates object o
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree

from sensor_vlm.hot3d import FPS

SPEED_MOVING = 0.05  # m/s
ANGULAR_MOVING = 1.0  # rad/s; resting objects: p99.9 0.50, max 0.86 (59 clips)
FILL_GAP = 9  # frames: merge motion bursts separated by < 0.3 s
MIN_RUN = 6  # frames: drop motion shorter than 0.2 s
PAD = 3  # frames: wrist method, contact starts before and ends after the motion
WRIST_REACH = 0.135  # m
BRIDGE = 15  # frames: wrist method, keep a hold through pauses < 0.5 s in reach
TOUCH_DIST = 0.01  # m
TOUCH_VERTS = 5
TOUCH_FILL = 6  # frames: bridge contact flicker < 0.2 s
TOUCH_MIN = 3  # frames
MOTION_OVERLAP = 3  # frames of object motion a touch must contain to count
CANDIDATE_REACH = 0.25  # m: only test mesh contact when the wrist is this close


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """[..., (w, x, y, z)] -> [..., 3, 3]."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


class SurfaceIndex:
    """Nearest-surface queries against HOT3D object models, in object coordinates."""

    def __init__(self, models_dir: Path, n_samples: int = 4000):
        self.models_dir = Path(models_dir)
        self.n_samples = n_samples
        self._trees: dict[int, cKDTree] = {}

    def tree(self, oid: int) -> cKDTree:
        if oid not in self._trees:
            mesh = trimesh.load(self.models_dir / f"obj_{oid:06d}.glb", force="mesh")
            pts, _ = trimesh.sample.sample_surface_even(mesh, self.n_samples, seed=0)
            self._trees[oid] = cKDTree(np.vstack([pts, mesh.vertices]))
        return self._trees[oid]

    def distance(self, oid: int, obj_pose: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Distance from world points (T, 3) to the surface of object `oid` posed by (T, 7)."""
        out = np.full(len(points), np.nan)
        ok = ~(np.isnan(obj_pose).any(-1) | np.isnan(points).any(-1))
        if ok.any():
            R = quat_to_rot(obj_pose[ok, :4])
            local = np.einsum("tji,tj->ti", R, points[ok] - obj_pose[ok, 4:])  # R^T (p - t)
            out[ok] = self.tree(oid).query(local)[0]
        return out

    def count_within(self, oid: int, obj_pose: np.ndarray, verts: np.ndarray, radius: float) -> np.ndarray:
        """Per frame, how many of the (T, V, 3) world points lie within `radius` of the surface."""
        T, V, _ = verts.shape
        R = quat_to_rot(obj_pose[:, :4])
        local = np.einsum("tji,tvj->tvi", R, verts - obj_pose[:, None, 4:])
        d = self.tree(oid).query(local.reshape(-1, 3), distance_upper_bound=radius, workers=-1)[0]
        return np.isfinite(d).reshape(T, V).sum(-1)


def hand_vertices(hand_shape: dict, wrist_pose: np.ndarray, joint_angles: np.ndarray) -> np.ndarray:
    """UmeTrack mesh vertices (T, 2, V, 3) in world meters; NaN where a hand is untracked."""
    from hand_tracking_toolkit.hand_models.umetrack_hand_model import from_json, skin_vertices

    model = from_json(hand_shape)
    V = model.mesh_vertices.shape[0]
    out = np.full(wrist_pose.shape[:2] + (V, 3), np.nan, dtype=np.float32)
    for h in range(2):
        ok = ~np.isnan(wrist_pose[:, h]).any(-1)
        n = int(ok.sum())
        if n == 0:
            continue
        M = np.zeros((n, 4, 4))
        M[:, :3, :3] = quat_to_rot(wrist_pose[ok, h, :4])
        M[:, :3, 3] = wrist_pose[ok, h, 4:]
        M[:, 3, 3] = 1
        if h == 1:
            M[:, :, 0] *= -1  # the right hand is the mirrored model, as in umetrack forward_kinematics
        batched = dataclasses.replace(model, **{
            k: getattr(model, k).expand(n, *getattr(model, k).shape)
            for k in ("joint_rest_positions", "joint_rotation_axes", "mesh_vertices", "dense_bone_weights")
        })
        verts = skin_vertices(batched, torch.tensor(joint_angles[ok, h]).float(), torch.tensor(M).float())
        out[ok, h] = verts.numpy()
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each True run."""
    d = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def clean_runs(mask: np.ndarray, fill_gap: int, min_run: int) -> np.ndarray:
    """Fill interior False gaps shorter than fill_gap, then drop True runs shorter than min_run."""
    m = mask.copy()
    for start, end in runs(~m):
        if start > 0 and end < len(m) and end - start < fill_gap:
            m[start:end] = True
    for start, end in runs(m):
        if end - start < min_run:
            m[start:end] = False
    return m


def dilate(mask: np.ndarray, n: int) -> np.ndarray:
    out = mask.copy()
    for k in range(1, n + 1):
        out[k:] |= mask[:-k]
        out[:-k] |= mask[k:]
    return out


def angular_speed(q: np.ndarray) -> np.ndarray:
    """Per-frame rotation rate (rad/s) of a (T, 4) quaternion track."""
    dot = np.abs(np.sum(q[1:] * q[:-1], axis=-1)).clip(max=1.0)
    rate = 2 * np.arccos(dot) * FPS
    return np.r_[rate[:1], rate] if len(rate) else np.zeros(len(q))


def label_clip(clip_dir: Path, index: SurfaceIndex, method: str = "contact") -> dict[str, np.ndarray]:
    kin = np.load(clip_dir / "kin.npz")
    pose, wrist = kin["obj_pose"], kin["wrist_pose"]
    T, O = pose.shape[:2]

    vel = np.gradient(pose[:, :, 4:], axis=0) * FPS
    speed = uniform_filter1d(np.nan_to_num(np.linalg.norm(vel, axis=-1)), 5, axis=0)
    ang = np.stack([uniform_filter1d(np.nan_to_num(angular_speed(pose[:, o, :4])), 5) for o in range(O)], -1)
    moving = np.zeros((T, O), dtype=bool)
    wrist_dist = np.full((T, 2, O), np.nan)
    for o, oid in enumerate(kin["obj_ids"]):
        raw = (speed[:, o] > SPEED_MOVING) | (ang[:, o] > ANGULAR_MOVING)
        moving[:, o] = clean_runs(raw, FILL_GAP, MIN_RUN)
        for h in range(2):
            wrist_dist[:, h, o] = index.distance(int(oid), pose[:, o], wrist[:, h, 4:])

    held = np.zeros((T, 2, O), dtype=bool)
    out = {"speed": speed.astype(np.float32), "angular_speed": ang.astype(np.float32), "moving": moving,
           "wrist_dist": wrist_dist.astype(np.float32)}

    if method == "wrist":
        padded = np.stack([dilate(moving[:, o], PAD) for o in range(O)], -1)
        near = wrist_dist < WRIST_REACH
        for o in range(O):
            for h in range(2):
                held[:, h, o] = clean_runs(padded[:, o] & near[:, h, o], BRIDGE, 1) & near[:, h, o]
    elif method == "contact":
        meta = json.loads((clip_dir / "meta.json").read_text())
        verts = hand_vertices(meta["hand_shape"]["umetrack"], wrist, kin["joint_angles"])
        touch = np.zeros((T, 2, O), dtype=bool)
        for o, oid in enumerate(kin["obj_ids"]):
            if not moving[:, o].any():
                continue  # touching a resting object is never manipulation
            for h in range(2):
                cand = np.flatnonzero((wrist_dist[:, h, o] < CANDIDATE_REACH) & ~np.isnan(pose[:, o]).any(-1))
                if len(cand) == 0:
                    continue
                n = index.count_within(int(oid), pose[cand, o], verts[cand, h], TOUCH_DIST)
                touch[cand, h, o] = n >= TOUCH_VERTS
                touch[:, h, o] = clean_runs(touch[:, h, o], TOUCH_FILL, TOUCH_MIN)
                for start, end in runs(touch[:, h, o]):
                    if moving[start:end, o].sum() >= MOTION_OVERLAP:
                        held[start:end, h, o] = True
        out["touch"] = touch
    else:
        raise ValueError(f"unknown method {method!r}")
    out["held"] = held
    return out


def object_names(clip_dir: Path) -> list[str]:
    meta = json.loads((clip_dir / "meta.json").read_text())
    return [meta["objects"][str(int(o))] for o in np.load(clip_dir / "kin.npz")["obj_ids"]]
