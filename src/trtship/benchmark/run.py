"""Benchmark drivers: run a target across the configured batch sizes and concurrency levels."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from trtship.benchmark.runner import BenchmarkTarget, measure
from trtship.benchmark.schema import (
    BenchmarkMeasurement,
    BenchmarkReport,
    BenchmarkSubject,
)
from trtship.benchmark.targets import OrtTarget, TensorRTTarget, describe_environment
from trtship.benchmark.triton_target import TritonTarget
from trtship.config import Precision, TrtshipConfig
from trtship.errors import BenchmarkError
from trtship.logging import get_logger
from trtship.models import LoadedModel
from trtship.tensorrt.executor import TensorRTExecutor
from trtship.triton.client import TritonClient
from trtship.utils.env import EnvironmentReport
from trtship.utils.hashing import sha256_file
from trtship.utils.timeutil import utc_now

log = get_logger(__name__)


class _Matrix:
    """The measurements and skipped combinations of one benchmark run."""

    def __init__(self) -> None:
        self.measurements: list[BenchmarkMeasurement] = []
        self.skipped: list[str] = []


def run_matrix(
    target: BenchmarkTarget,
    config: TrtshipConfig,
    *,
    include_raw: bool = True,
    matrix: _Matrix | None = None,
) -> _Matrix:
    """Measure ``target`` at every configured (batch size, concurrency) pair it supports."""
    settings = config.benchmark
    result = matrix if matrix is not None else _Matrix()
    for batch in settings.batch_sizes:
        for concurrency in settings.concurrency:
            if concurrency > 1 and not target.supports_concurrency:
                result.skipped.append(
                    f"{target.backend} {target.precision or ''} batch={batch} "
                    f"concurrency={concurrency}: this backend runs one request at a time"
                )
                continue
            log.info("benchmarking %s batch=%d concurrency=%d", target.backend, batch, concurrency)
            result.measurements.append(
                measure(
                    target,
                    batch_size=batch,
                    concurrency=concurrency,
                    warmup_iters=settings.warmup_iters,
                    iters=settings.iters,
                    include_raw=include_raw,
                )
            )
    return result


def _environment(report: EnvironmentReport) -> dict[str, Any]:
    gpus = [f"{g.name} (sm_{(g.compute_capability or '?').replace('.', '')})" for g in report.gpus]
    versions = {**report.versions(), "platform": report.platform}
    return describe_environment(versions, gpus)


def benchmark_onnx(
    onnx_path: Path,
    model: LoadedModel,
    config: TrtshipConfig,
    environment: EnvironmentReport,
    *,
    include_raw: bool = True,
) -> BenchmarkReport:
    """Benchmark an ONNX model on ONNX Runtime's CPU provider. Real measurements, labelled CPU."""
    seed = config.benchmark.seed if config.benchmark.seed is not None else config.seed
    target = OrtTarget(onnx_path, config, seed)
    try:
        matrix = run_matrix(target, config, include_raw=include_raw)
    finally:
        target.close()
    return BenchmarkReport(
        generated_at=utc_now(),
        model_name=model.name,
        weights_sha256=model.weights_sha256,
        subject=BenchmarkSubject(kind="onnx", path=str(onnx_path), sha256=sha256_file(onnx_path)),
        environment=_environment(environment),
        seed=seed,
        measurements=matrix.measurements,
        skipped=matrix.skipped,
    )


def benchmark_engines(
    engines: Sequence[tuple[Precision, Path]],
    model: LoadedModel,
    config: TrtshipConfig,
    environment: EnvironmentReport,
    *,
    executor_factory: Callable[[Path], TensorRTExecutor],
    gpu_used_mb: Callable[[], float | None] | None = None,
    device_label: str = "cuda:0",
    include_raw: bool = True,
) -> BenchmarkReport:
    """Benchmark each engine directly (no serving layer)."""
    if not engines:
        raise BenchmarkError("no engines to benchmark", hint="Run the build stage first.")
    seed = config.benchmark.seed if config.benchmark.seed is not None else config.seed
    wanted = config.benchmark.precisions
    matrix = _Matrix()
    for precision, path in engines:
        if wanted is not None and precision not in wanted:
            continue
        before = gpu_used_mb() if gpu_used_mb else None
        executor = executor_factory(path)
        target = TensorRTTarget(
            executor,
            config,
            seed,
            precision=precision.value,
            device=device_label,
            gpu_used_mb=gpu_used_mb,
            gpu_baseline_mb=before,
        )
        try:
            run_matrix(target, config, include_raw=include_raw, matrix=matrix)
        finally:
            target.close()
    first = engines[0]
    return BenchmarkReport(
        generated_at=utc_now(),
        model_name=model.name,
        weights_sha256=model.weights_sha256,
        subject=BenchmarkSubject(
            kind="engine",
            path=str(first[1]) if len(engines) == 1 else None,
            sha256=sha256_file(first[1]) if len(engines) == 1 else None,
            precision=first[0].value if len(engines) == 1 else None,
        ),
        environment=_environment(environment),
        seed=seed,
        measurements=matrix.measurements,
        skipped=matrix.skipped,
    )


def benchmark_triton(
    client_factory: Callable[[], TritonClient],
    model: LoadedModel,
    config: TrtshipConfig,
    environment: EnvironmentReport,
    *,
    protocol: str,
    precision: Precision | None,
    endpoint: str,
    plan: Path | None = None,
    include_raw: bool = True,
) -> BenchmarkReport:
    """Benchmark the model served by a running Triton server through its client API."""
    seed = config.benchmark.seed if config.benchmark.seed is not None else config.seed
    target = TritonTarget(
        client_factory,
        config.model.name,
        config,
        seed,
        protocol=protocol,
        precision=precision.value if precision else None,
        device=f"triton server at {endpoint}",
    )
    try:
        matrix = run_matrix(target, config, include_raw=include_raw)
    finally:
        target.close()
    return BenchmarkReport(
        generated_at=utc_now(),
        model_name=model.name,
        weights_sha256=model.weights_sha256,
        subject=BenchmarkSubject(
            kind="triton",
            path=str(plan) if plan else None,
            sha256=sha256_file(plan) if plan else None,
            precision=precision.value if precision else None,
        ),
        environment=_environment(environment),
        seed=seed,
        measurements=matrix.measurements,
        skipped=matrix.skipped,
    )
