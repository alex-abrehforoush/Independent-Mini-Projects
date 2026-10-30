"""Random key placements used to train the (rate-agnostic) keyframe tokenizer."""
from __future__ import annotations

import math
import random

import torch

from . import keyframes

EPS_GRID = tuple(round(math.exp(x), 2) for x in torch.linspace(math.log(1.0), math.log(40.0), 12).tolist())


def sample_mask(clips, idx, rng: random.Random, p_uniform: float = 0.4):
    """Uniform strides 2..16 with probability p_uniform, otherwise adaptive keys at a random eps
    from the precomputed grid (1..40 cm)."""
    B, T = idx.shape
    if rng.random() < p_uniform:
        return keyframes.uniform(B, T, rng.randint(2, 16), idx.device), "uniform"
    return clips.crop_mask(idx, rng.randrange(len(clips.key_eps))), "adaptive"
