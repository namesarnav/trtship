"""PyTorch-vs-ONNX validation.

The reference (PyTorch) and the ONNX graph, executed by ONNX Runtime on the CPU, are run on the same
deterministic inputs at several *shape points* and compared with the configured tolerance.

Shape points come from the first optimization profile (its min, opt, and max shapes), or from the
two probe sizes used for signature inference when there is no profile. Running at more than the
traced shape is what exposes a dynamic dimension that tracing turned into a constant: such a graph
declares dynamic shapes but returns the traced sizes at every other size, which shows up here as a
shape mismatch.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy.typing as npt
import torch
from pydantic import BaseModel, ConfigDict, Field

from trtship.config import Tolerance, TrtshipConfig
from trtship.errors import ValidationFailedError
from trtship.logging import get_logger
from trtship.models import LoadedModel, ModelSignature, make_inputs, resolve_symbol_sizes
from trtship.onnx.graph import GraphReport, analyze_graph
from trtship.onnx.runtime import OrtSession
from trtship.utils.hashing import sha256_file
from trtship.utils.timeutil import utc_now
from trtship.validation import TensorComparison, compare_outputs

log = get_logger(__name__)

REPORT_SCHEMA_VERSION = 1


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ShapePointResult(_Frozen):
    label: str
    sizes: dict[str, int]
    samples: int
    samples_passed: int
    outputs: list[TensorComparison]  # the worst comparison per output across samples
    error: str | None = None  # set when the point could not be executed at all
    passed: bool


class OnnxValidationReport(_Frozen):
    schema_version: int = REPORT_SCHEMA_VERSION
    generated_at: datetime
    onnx_path: str
    onnx_sha256: str
    onnx_size_bytes: int
    model_name: str
    weights_sha256: str
    ort_version: str
    providers: list[str]
    ort_graph_optimization: bool
    seed: int
    tolerance: Tolerance
    graph: GraphReport
    points: list[ShapePointResult]
    duration_s: float
    passed: bool
    failures: list[str] = Field(default_factory=list)

    def raise_for_failure(self) -> None:
        if not self.passed:
            shown = self.failures[:5]
            more = f" (+{len(self.failures) - 5} more)" if len(self.failures) > 5 else ""
            raise ValidationFailedError(
                "ONNX validation failed: " + "; ".join(shown) + more,
                hint="See the report for per-output metrics. Raise validation.onnx_tolerance only "
                "if the deviation is understood and acceptable.",
                details={"failures": self.failures, "onnx": self.onnx_path},
            )


def select_shape_points(
    config: TrtshipConfig, signature: ModelSignature
) -> list[tuple[str, dict[str, int]]]:
    """``(label, symbol sizes)`` pairs to validate at; identical size sets are merged."""
    symbols = {s for spec in config.model.inputs for s in spec.symbols}
    if not symbols:
        return [("static", {})]
    candidates: list[tuple[str, dict[str, int]]]
    if config.tensorrt.profiles:
        profiles = config.tensorrt.profiles
        candidates = [
            (which, resolve_symbol_sizes(config.model, profiles, which))
            for which in ("min", "opt", "max")
        ]
    else:
        candidates = [
            (f"probe-{i + 1}", dict(sizes)) for i, sizes in enumerate(signature.probe_sizes)
        ]
    merged: list[tuple[str, dict[str, int]]] = []
    for label, sizes in candidates:
        for index, (existing_label, existing_sizes) in enumerate(merged):
            if existing_sizes == sizes:
                merged[index] = (f"{existing_label}={label}", existing_sizes)
                break
        else:
            merged.append((label, sizes))
    return merged


def _to_numpy(tensors: Mapping[str, torch.Tensor]) -> dict[str, npt.NDArray[Any]]:
    return {name: t.detach().cpu().numpy() for name, t in tensors.items()}


def _worst(comparisons: list[TensorComparison]) -> TensorComparison:
    failing = [c for c in comparisons if not c.passed]
    pool = failing or comparisons
    return max(pool, key=lambda c: c.max_abs_error if c.max_abs_error is not None else float("inf"))


def _run_point(
    label: str,
    sizes: dict[str, int],
    model: LoadedModel,
    session: OrtSession,
    tolerance: Tolerance,
    *,
    samples: int,
    seed: int,
) -> ShapePointResult:
    per_output: dict[str, list[TensorComparison]] = {}
    samples_passed = 0
    try:
        for index in range(samples):
            inputs = make_inputs(model.inputs, sizes, seed=seed + index, device=model.device)
            reference = _to_numpy(model.run(inputs))
            candidate = session.run(_to_numpy(inputs))
            results = compare_outputs(reference, candidate, tolerance)
            samples_passed += all(r.passed for r in results)
            for result in results:
                per_output.setdefault(result.name, []).append(result)
    except ValidationFailedError as exc:
        return ShapePointResult(
            label=label,
            sizes=sizes,
            samples=samples,
            samples_passed=samples_passed,
            outputs=[],
            error=exc.message,
            passed=False,
        )
    outputs = [_worst(items) for items in per_output.values()]
    return ShapePointResult(
        label=label,
        sizes=sizes,
        samples=samples,
        samples_passed=samples_passed,
        outputs=outputs,
        passed=samples_passed == samples and all(o.passed for o in outputs),
    )


def _io_problems(session: OrtSession, signature: ModelSignature) -> list[str]:
    problems: list[str] = []
    expected_in = [s.name for s in signature.inputs]
    expected_out = [s.name for s in signature.outputs]
    if session.input_names != expected_in:
        problems.append(f"ONNX inputs {session.input_names} differ from the model's {expected_in}")
    if session.output_names != expected_out:
        problems.append(
            f"ONNX outputs {session.output_names} differ from the model's {expected_out}"
        )
    return problems


def validate_onnx(
    onnx_path: Path,
    model: LoadedModel,
    signature: ModelSignature,
    config: TrtshipConfig,
) -> OnnxValidationReport:
    """Validate ``onnx_path`` against ``model``. Raises only if the graph cannot be analyzed or
    loaded; comparison failures are recorded in the returned report (see ``raise_for_failure``)."""
    started = time.perf_counter()
    graph = analyze_graph(onnx_path)
    session = OrtSession(onnx_path)
    settings = config.validation
    seed = settings.seed if settings.seed is not None else config.seed
    failures = _io_problems(session, signature)

    points: list[ShapePointResult] = []
    if not failures:  # comparing outputs is meaningless if the interfaces differ
        for label, sizes in select_shape_points(config, signature):
            log.info("validating %s at %s (%d samples)", label, sizes, settings.num_samples)
            point = _run_point(
                label,
                sizes,
                model,
                session,
                settings.onnx_tolerance,
                samples=settings.num_samples,
                seed=seed,
            )
            points.append(point)
            if point.error:
                failures.append(f"{point.label} {sizes}: {point.error}")
            for output in point.outputs:
                failures.extend(
                    f"{point.label} {sizes}: {output.name}: {f}" for f in output.failures
                )

    return OnnxValidationReport(
        generated_at=utc_now(),
        onnx_path=str(onnx_path),
        onnx_sha256=sha256_file(onnx_path),
        onnx_size_bytes=onnx_path.stat().st_size,
        model_name=model.name,
        weights_sha256=model.weights_sha256,
        ort_version=session.version,
        providers=session.providers,
        ort_graph_optimization=session.optimize,
        seed=seed,
        tolerance=settings.onnx_tolerance,
        graph=graph,
        points=points,
        duration_s=round(time.perf_counter() - started, 3),
        passed=not failures and all(p.passed for p in points) and bool(points),
        failures=failures,
    )


__all__ = [
    "OnnxValidationReport",
    "ShapePointResult",
    "select_shape_points",
    "validate_onnx",
]
