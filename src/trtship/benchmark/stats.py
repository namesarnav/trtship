"""Latency statistics.

Percentiles use linear interpolation between order statistics (``numpy.percentile`` "linear"). With
few samples the tail percentiles are estimates, not observations: p99 of 50 samples is essentially
the maximum. Reports carry the sample count so that is visible.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from trtship.errors import BenchmarkError


class LatencySummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    count: int
    min_ms: float
    mean_ms: float
    stdev_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    @property
    def cv(self) -> float:
        """Coefficient of variation (stdev / mean); a rough measure of measurement noise."""
        return self.stdev_ms / self.mean_ms if self.mean_ms > 0 else 0.0


def summarize(samples_ms: Sequence[float]) -> LatencySummary:
    if not len(samples_ms):
        raise BenchmarkError("cannot summarize an empty set of latency samples")
    data = np.asarray(samples_ms, dtype=np.float64)
    if not np.isfinite(data).all() or (data < 0).any():
        raise BenchmarkError("latency samples must be finite and non-negative")
    p50, p90, p95, p99 = (float(v) for v in np.percentile(data, [50, 90, 95, 99]))
    return LatencySummary(
        count=int(data.size),
        min_ms=float(data.min()),
        mean_ms=float(data.mean()),
        stdev_ms=float(data.std(ddof=1)) if data.size > 1 else 0.0,
        p50_ms=p50,
        p90_ms=p90,
        p95_ms=p95,
        p99_ms=p99,
        max_ms=float(data.max()),
    )
