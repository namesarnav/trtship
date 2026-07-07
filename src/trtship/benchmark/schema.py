"""The benchmark report schema (versioned)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from trtship.benchmark.stats import LatencySummary

SCHEMA_VERSION: Final = 1
PHASES: Final = ("preprocess", "execute", "postprocess", "end_to_end")

METHODOLOGY: Final = (
    "Each measurement runs `warmup_iters` iterations that are excluded from the statistics (the "
    "first call is recorded separately as first_call_ms), then `iters` timed iterations per "
    "worker. Phases: preprocess (host data preparation and upload), execute (the model on its "
    "backend; GPU time by CUDA events for TensorRT), postprocess (result retrieval), end_to_end "
    "(wall clock of the whole request, with synchronization). Percentiles use linear "
    "interpolation. Throughput is samples completed per second of wall-clock time across all "
    "workers. Results depend on the machine, clocks, power state and concurrent load; compare "
    "only runs from the same machine."
)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MemoryUsage(_Frozen):
    gpu_mb: float | None = None  # peak GPU memory attributable to the run, where measurable
    cpu_rss_mb: float | None = None  # peak resident set size of this process


class BenchmarkMeasurement(_Frozen):
    backend: str  # e.g. "onnxruntime-cpu", "tensorrt", "triton-http"
    precision: str | None
    device: str
    batch_size: int
    concurrency: int
    warmup_iters: int
    iters: int  # per worker
    first_call_ms: float
    phases: dict[str, LatencySummary]
    throughput_samples_per_s: float
    requests_per_s: float
    duration_s: float
    memory: MemoryUsage
    notes: list[str] = Field(default_factory=list)
    raw_ms: dict[str, list[float]] = Field(default_factory=dict)  # per-phase samples


class BenchmarkSubject(_Frozen):
    kind: str  # "onnx" | "engine" | "triton"
    path: str | None
    sha256: str | None
    precision: str | None = None


class BenchmarkReport(_Frozen):
    schema_version: int = SCHEMA_VERSION
    generated_at: datetime
    model_name: str
    weights_sha256: str | None
    subject: BenchmarkSubject
    environment: dict[str, Any]  # versions and devices at measurement time
    seed: int
    methodology: str = METHODOLOGY
    measurements: list[BenchmarkMeasurement]
    # Requested combinations that were not measured, with the reason (never silently dropped).
    skipped: list[str] = Field(default_factory=list)
