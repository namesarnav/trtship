"""Seeding for reproducible runs."""

from __future__ import annotations

import random
from typing import TypedDict

import numpy as np
import torch


class SeedInfo(TypedDict):
    seed: int
    seeded: list[str]


def seed_everything(seed: int) -> SeedInfo:
    """Seed Python, NumPy, and torch (CPU and, if present, CUDA). Returns what was seeded.

    Seeding does not make every operation deterministic (some CUDA kernels are not); it makes the
    trtship-controlled randomness (example inputs, calibration sampling) repeatable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    seeded = ["python.random", "numpy.random", "torch.cpu"]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        seeded.append("torch.cuda")
    return {"seed": seed, "seeded": seeded}
