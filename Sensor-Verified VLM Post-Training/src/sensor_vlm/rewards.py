"""Verifiable rewards for the tile task, computed only from the sample.

Each component is in [0, 1] and scored on its own, so one malformed field
does not zero the others:
    format  every field is well-formed
    hands   fraction of the 8 per-hand, per-frame bits that match
    object  exact object name (or "none")
    point   the point lands on the target object's mask in the right frame,
            or is null when the target is "none"; NaN (left out of the
            total) when the target object is never visible while held
"""

from __future__ import annotations

import json
import re

from sensor_vlm.hot3d import MASK_SIZE
from sensor_vlm.tasks import decode_mask

WEIGHTS = {"format": 0.1, "hands": 0.4, "object": 0.25, "point": 0.25}
_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse(completion: str) -> dict | None:
    """The first {...} JSON object, tolerating code fences and surrounding text."""
    m = _JSON.search(completion)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def _bits(v) -> bool:
    return isinstance(v, list) and len(v) == 4 and all(x in (0, 1) and not isinstance(x, float) for x in v)


def _point(v) -> bool:
    return v is None or (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v))


def point_on_mask(point, frame: int, mask_b64: str) -> bool:
    """Tile point (0-1000) inside the mask of quadrant `frame`."""
    x, y = point[0] / 1000 * 2, point[1] / 1000 * 2
    qx, qy = frame % 2, frame // 2
    if not (qx <= x < qx + 1 and qy <= y < qy + 1):
        return False
    col, row = int((x - qx) * MASK_SIZE), int((y - qy) * MASK_SIZE)
    return bool(decode_mask(mask_b64)[row, col])


def score(completion: str, sample: dict) -> dict[str, float]:
    """Each field is scored on its own; `format` requires all of them to be well-formed."""
    zero = {"format": 0.0, "hands": 0.0, "object": 0.0, "point": 0.0, "total": 0.0}
    d = parse(completion)
    if d is None:
        return zero
    ans = sample["answer"]
    valid = {"left": _bits(d.get("left")), "right": _bits(d.get("right")),
             "object": isinstance(d.get("object"), str), "point": _point(d.get("point"))}
    bits = [valid[k] and a == b for k in ("left", "right") for a, b in zip(d.get(k) or [0] * 4, ans[k])]
    r = {
        "format": float(all(valid.values())),
        "hands": sum(bits) / 8,
        "object": float(valid["object"] and d["object"].strip().lower() == ans["object"].lower()),
    }
    if ans["object"] == "none":
        r["point"] = float(valid["point"] and d.get("point") is None)
    elif sample["point_mask"] is None:
        r["point"] = float("nan")
    else:
        p = d.get("point")
        r["point"] = float(valid["point"] and p is not None and point_on_mask(p, sample["point_frame"], sample["point_mask"]))
    scored = [k for k in WEIGHTS if r[k] == r[k]]
    r["total"] = sum(WEIGHTS[k] * r[k] for k in scored) / sum(WEIGHTS[k] for k in scored)
    return r
