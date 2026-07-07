"""Benchmarking: statistics, the measurement runner, backends, and reports."""

from trtship.benchmark.run import benchmark_engines, benchmark_onnx, run_matrix
from trtship.benchmark.runner import BenchmarkTarget, PhaseTimes, measure
from trtship.benchmark.schema import (
    METHODOLOGY,
    PHASES,
    BenchmarkMeasurement,
    BenchmarkReport,
    BenchmarkSubject,
    MemoryUsage,
)
from trtship.benchmark.stats import LatencySummary, summarize
from trtship.benchmark.targets import (
    OrtTarget,
    TensorRTTarget,
    batch_inputs,
    torch_gpu_used_mb,
)

__all__ = [
    "METHODOLOGY",
    "PHASES",
    "BenchmarkMeasurement",
    "BenchmarkReport",
    "BenchmarkSubject",
    "BenchmarkTarget",
    "LatencySummary",
    "MemoryUsage",
    "OrtTarget",
    "PhaseTimes",
    "TensorRTTarget",
    "batch_inputs",
    "benchmark_engines",
    "benchmark_onnx",
    "measure",
    "run_matrix",
    "summarize",
    "torch_gpu_used_mb",
]
