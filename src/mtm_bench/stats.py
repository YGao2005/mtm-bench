"""Dependency-free statistics helpers shared by every scorer."""

from __future__ import annotations

import math


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Dependency-free (no numpy/scipy).
    Returns (0.0, 0.0) for n == 0. Chosen over Wald because n is small and our rates sit near
    the boundaries, where Wald degenerates (intervals exceeding [0,1] or zero-width at p=1)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))
