"""Numerical validation of TensorRT engines: PyTorch vs ONNX Runtime vs TensorRT.

For every engine (one per precision) the model is run at the optimization profile's min/opt/max
shapes on deterministic inputs through three paths: PyTorch (the reference), ONNX Runtime on the
CPU, and TensorRT. TensorRT is compared with PyTorch against the tolerance configured for its
precision, which decides pass/fail; the comparison with ONNX Runtime is reported alongside it.

Agreement is measured on the sampled inputs. It says an engine reproduces the reference on those
inputs within tolerance, not that the model is accurate on a task.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy.typing as npt
import torch
from pydantic import BaseModel, ConfigDict, Field

from trtship.config import Precision, Tolerance, TrtshipConfig
from trtship.errors import TrtshipError, ValidationFailedError
from trtship.logging import get_logger
from trtship.models import LoadedModel, ModelSignature, make_inputs
from trtship.onnx.runtime import OrtSession
from trtship.onnx.validate import select_shape_points
from trtship.utils.hashing import sha256_file
from trtship.utils.timeutil import utc_now
from trtship.validation.metrics import TensorComparison, compare_outputs, worst_comparison

log = get_logger(__name__)

REPORT_SCHEMA_VERSION = 1


class EngineExecutor(Protocol):
    """What validation needs from an engine runner (see ``TensorRTExecutor``)."""

    def run(self, inputs: Mapping[str, npt.NDArray[Any]]) -> dict[str, npt.NDArray[Any]]: ...

    def close(self) -> None: ...


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EngineUnderTest(_Frozen):
    precision: Precision
    path: str


class EnginePointResult(_Frozen):
    label: str
    sizes: dict[str, int]
    samples: int
    samples_passed: int
    vs_pytorch: list[TensorComparison]  # decides pass/fail
    vs_onnx: list[TensorComparison]  # reported for context
    error: str | None = None
    passed: bool


class PrecisionResult(_Frozen):
    precision: Precision
    engine_path: str
    engine_sha256: str
    tolerance: Tolerance
    points: list[EnginePointResult]
    passed: bool


class EngineValidationReport(_Frozen):
    schema_version: int = REPORT_SCHEMA_VERSION
    generated_at: datetime
    onnx_path: str
    onnx_sha256: str
    model_name: str
    weights_sha256: str
    ort_version: str
    seed: int
    results: list[PrecisionResult]
    duration_s: float
    passed: bool
    failures: list[str] = Field(default_factory=list)

    def raise_for_failure(self) -> None:
        if not self.passed:
            shown = self.failures[:5]
            more = f" (+{len(self.failures) - 5} more)" if len(self.failures) > 5 else ""
            raise ValidationFailedError(
                "engine validation failed: " + "; ".join(shown) + more,
                hint="See the report for per-output metrics. Tolerances are per precision "
                "(validation.tolerances); loosen one only if the deviation is understood.",
                details={"failures": self.failures},
            )


def _to_numpy(tensors: Mapping[str, torch.Tensor]) -> dict[str, npt.NDArray[Any]]:
    return {name: t.detach().cpu().numpy() for name, t in tensors.items()}


def _validate_point(
    label: str,
    sizes: dict[str, int],
    engine: EngineExecutor,
    model: LoadedModel,
    ort: OrtSession,
    tolerance: Tolerance,
    *,
    samples: int,
    seed: int,
) -> EnginePointResult:
    vs_pytorch: dict[str, list[TensorComparison]] = {}
    vs_onnx: dict[str, list[TensorComparison]] = {}
    samples_passed = 0
    try:
        for index in range(samples):
            inputs = make_inputs(model.inputs, sizes, seed=seed + index, device=model.device)
            host_inputs = _to_numpy(inputs)
            reference = _to_numpy(model.run(inputs))
            onnx_out = ort.run(host_inputs)
            candidate = engine.run(host_inputs)
            against_torch = compare_outputs(reference, candidate, tolerance)
            samples_passed += all(c.passed for c in against_torch)
            for comparison in against_torch:
                vs_pytorch.setdefault(comparison.name, []).append(comparison)
            for comparison in compare_outputs(onnx_out, candidate, tolerance):
                vs_onnx.setdefault(comparison.name, []).append(comparison)
    except TrtshipError as exc:
        return EnginePointResult(
            label=label,
            sizes=sizes,
            samples=samples,
            samples_passed=samples_passed,
            vs_pytorch=[],
            vs_onnx=[],
            error=exc.message,
            passed=False,
        )
    torch_worst = [worst_comparison(items) for items in vs_pytorch.values()]
    return EnginePointResult(
        label=label,
        sizes=sizes,
        samples=samples,
        samples_passed=samples_passed,
        vs_pytorch=torch_worst,
        vs_onnx=[worst_comparison(items) for items in vs_onnx.values()],
        passed=samples_passed == samples and all(c.passed for c in torch_worst),
    )


def validate_engines(
    engines: Sequence[EngineUnderTest],
    onnx_path: Path,
    model: LoadedModel,
    signature: ModelSignature,
    config: TrtshipConfig,
    executor_factory: Callable[[Path], EngineExecutor],
) -> EngineValidationReport:
    """Validate each engine in ``engines``. Comparison failures are recorded in the report (see
    ``raise_for_failure``); only a missing or unloadable engine raises."""
    started = time.perf_counter()
    settings = config.validation
    seed = settings.seed if settings.seed is not None else config.seed
    ort = OrtSession(onnx_path)
    points = select_shape_points(config, signature)
    failures: list[str] = []
    results: list[PrecisionResult] = []

    for under_test in engines:
        tolerance = settings.tolerances[under_test.precision]
        path = Path(under_test.path)
        executor = executor_factory(path)
        try:
            point_results = []
            for label, sizes in points:
                log.info("validating %s engine at %s %s", under_test.precision.value, label, sizes)
                point = _validate_point(
                    label,
                    sizes,
                    executor,
                    model,
                    ort,
                    tolerance,
                    samples=settings.num_samples,
                    seed=seed,
                )
                point_results.append(point)
                tag = f"{under_test.precision.value} {label} {sizes}"
                if point.error:
                    failures.append(f"{tag}: {point.error}")
                for comparison in point.vs_pytorch:
                    failures.extend(f"{tag}: {comparison.name}: {f}" for f in comparison.failures)
        finally:
            executor.close()
        results.append(
            PrecisionResult(
                precision=under_test.precision,
                engine_path=str(path),
                engine_sha256=sha256_file(path),
                tolerance=tolerance,
                points=point_results,
                passed=all(p.passed for p in point_results) and bool(point_results),
            )
        )
    return EngineValidationReport(
        generated_at=utc_now(),
        onnx_path=str(onnx_path),
        onnx_sha256=sha256_file(onnx_path),
        model_name=model.name,
        weights_sha256=model.weights_sha256,
        ort_version=ort.version,
        seed=seed,
        results=results,
        duration_s=round(time.perf_counter() - started, 3),
        passed=bool(results) and all(r.passed for r in results) and not failures,
        failures=failures,
    )
