"""Metrics for comparing a candidate's outputs against a reference.

Definitions (all computed in float64):

* absolute error ``|c - r|``; relative error ``|c - r| / max(|r|, 1e-6)``
* an element is a *violation* when ``|c - r| > atol + rtol * |r|``
* cosine similarity over the whole tensor, and the minimum over samples (first axis) for
  rank >= 2 tensors; two all-zero vectors count as identical, one zero vector as orthogonal
* for rank-2 floating outputs ``[samples, classes]`` (classification-like):
  top-1 agreement = fraction of samples whose argmax matches; top-k agreement = fraction of
  samples whose *reference* top-1 class is within the candidate's top-k; and the mean
  KL divergence ``KL(softmax(reference) || softmax(candidate))``
* integer and boolean outputs must match exactly

A comparison passes only if every applicable criterion holds. Nothing here is a benchmark or a
guarantee of task accuracy: agreement with a reference on the sampled inputs is what is measured.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from trtship.config import Tolerance

_REL_EPS = 1e-6

Array = npt.NDArray[np.generic]
Float64 = npt.NDArray[np.float64]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TensorStats(_Frozen):
    mean: float
    std: float
    min: float
    max: float


class TensorComparison(_Frozen):
    name: str
    dtype: str
    reference_shape: list[int]
    candidate_shape: list[int]
    shape_match: bool
    element_count: int
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    max_rel_error: float | None = None
    mean_rel_error: float | None = None
    cosine_similarity: float | None = None
    cosine_min_per_sample: float | None = None
    violations: int = 0
    non_finite_candidate: int = 0
    top1_agreement: float | None = None
    topk: int | None = None
    topk_agreement: float | None = None
    kl_divergence: float | None = None
    reference_stats: TensorStats | None = None
    candidate_stats: TensorStats | None = None
    passed: bool
    failures: list[str]


def _stats(values: Float64) -> TensorStats | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return TensorStats(
        mean=float(finite.mean()),
        std=float(finite.std()),
        min=float(finite.min()),
        max=float(finite.max()),
    )


def _cosine(a: Float64, b: Float64) -> float:
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


def _cosine_per_sample(ref: Float64, cand: Float64) -> float:
    rows_r, rows_c = ref.reshape(ref.shape[0], -1), cand.reshape(cand.shape[0], -1)
    return min(_cosine(r, c) for r, c in zip(rows_r, rows_c, strict=True))


def _softmax(x: Float64) -> Float64:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    result: Float64 = exp / exp.sum(axis=-1, keepdims=True)
    return result


def _kl(ref: Float64, cand: Float64) -> float:
    p, q = _softmax(ref), _softmax(cand)
    tiny = np.finfo(np.float64).tiny
    return float(np.mean(np.sum(p * (np.log(p + tiny) - np.log(q + tiny)), axis=-1)))


class _Errors(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_abs: float
    mean_abs: float
    max_rel: float
    mean_rel: float


def _error_stats(diff: Float64, ref: Float64, both_finite: npt.NDArray[np.bool_]) -> _Errors:
    """Error statistics over the elements where both tensors are finite."""
    diff_finite, ref_finite = diff[both_finite], ref[both_finite]
    if not diff_finite.size:
        return _Errors(max_abs=0.0, mean_abs=0.0, max_rel=0.0, mean_rel=0.0)
    rel = diff_finite / np.maximum(np.abs(ref_finite), _REL_EPS)
    return _Errors(
        max_abs=float(diff_finite.max()),
        mean_abs=float(diff_finite.mean()),
        max_rel=float(rel.max()),
        mean_rel=float(rel.mean()),
    )


def _cosine_metrics(
    ref: Float64, cand: Float64, tolerance: Tolerance
) -> tuple[float | None, float | None, list[str]]:
    """(global cosine, per-sample minimum, failures); ``None`` if any value is non-finite."""
    flat_ref, flat_cand = ref.reshape(-1), cand.reshape(-1)
    if not (np.isfinite(flat_ref).all() and np.isfinite(flat_cand).all()):
        return None, None, []
    cosine = _cosine(flat_ref, flat_cand)
    cosine_min = _cosine_per_sample(ref, cand) if ref.ndim >= 2 else cosine
    failures = []
    if cosine_min < tolerance.cosine_min:
        failures.append(
            f"cosine similarity {cosine_min:.6f} is below the minimum {tolerance.cosine_min:g}"
        )
    return cosine, cosine_min, failures


class _Agreement(BaseModel):
    model_config = ConfigDict(frozen=True)

    top1: float | None = None
    topk: int | None = None
    topk_agreement: float | None = None
    kl: float | None = None
    failures: list[str] = []


def _classification_metrics(ref: Float64, cand: Float64, tolerance: Tolerance) -> _Agreement:
    """Top-1/top-k agreement and KL divergence for ``[samples, classes]`` outputs."""
    applicable = (
        ref.ndim == 2
        and ref.shape[1] >= 2
        and bool(np.isfinite(ref).all())
        and bool(np.isfinite(cand).all())
    )
    if not applicable:
        return _Agreement()
    ref_top = ref.argmax(axis=1)
    top1 = float(np.mean(ref_top == cand.argmax(axis=1)))
    k = min(tolerance.topk, ref.shape[1])
    candidate_topk = np.argpartition(-cand, k - 1, axis=1)[:, :k]
    topk_agreement = float(np.mean((candidate_topk == ref_top[:, None]).any(axis=1)))
    failures = []
    if tolerance.top1_agreement_min is not None and top1 < tolerance.top1_agreement_min:
        failures.append(
            f"top-1 agreement {top1:.4f} is below the minimum {tolerance.top1_agreement_min:g}"
        )
    if tolerance.topk_agreement_min is not None and topk_agreement < tolerance.topk_agreement_min:
        failures.append(
            f"top-{k} agreement {topk_agreement:.4f} is below the minimum "
            f"{tolerance.topk_agreement_min:g}"
        )
    return _Agreement(
        top1=top1, topk=k, topk_agreement=topk_agreement, kl=_kl(ref, cand), failures=failures
    )


def _shape_mismatch(name: str, reference: Array, candidate: Array) -> TensorComparison:
    ref_shape, cand_shape = list(reference.shape), list(candidate.shape)
    return TensorComparison(
        name=name,
        dtype=str(reference.dtype),
        reference_shape=ref_shape,
        candidate_shape=cand_shape,
        shape_match=False,
        element_count=int(reference.size),
        passed=False,
        failures=[f"shape mismatch: reference {ref_shape}, candidate {cand_shape}"],
    )


def compare_tensors(
    name: str, reference: Array, candidate: Array, tolerance: Tolerance
) -> TensorComparison:
    """Compare one output tensor. Shapes must match for value metrics to be computed."""
    if reference.shape != candidate.shape:
        return _shape_mismatch(name, reference, candidate)

    exact = reference.dtype.kind in "biu"  # bool / signed / unsigned integers
    ref, cand = reference.astype(np.float64), candidate.astype(np.float64)
    failures: list[str] = []

    non_finite = max(int(np.sum(~np.isfinite(cand)) - np.sum(~np.isfinite(ref))), 0)
    if non_finite:
        failures.append(f"{non_finite} non-finite value(s) in the candidate the reference lacks")

    both_finite = np.isfinite(ref) & np.isfinite(cand)
    diff = np.abs(cand - ref)
    errors = _error_stats(diff, ref, both_finite)

    if exact:
        violations = int(np.sum(reference != candidate))
        limit_text = "exactly (integer/bool output)"
    else:
        violations = int(
            np.sum(both_finite & (diff > tolerance.atol + tolerance.rtol * np.abs(ref)))
        )
        limit_text = f"atol={tolerance.atol:g} + rtol={tolerance.rtol:g}*|reference|"
    if violations:
        failures.append(
            f"{violations} of {reference.size} element(s) differ beyond {limit_text}; "
            f"max abs error {errors.max_abs:.6g}"
        )

    cosine = cosine_min = None
    agreement = _Agreement()
    if not exact and reference.size:
        cosine, cosine_min, cosine_failures = _cosine_metrics(ref, cand, tolerance)
        failures.extend(cosine_failures)
        agreement = _classification_metrics(ref, cand, tolerance)
        failures.extend(agreement.failures)

    return TensorComparison(
        name=name,
        dtype=str(reference.dtype),
        reference_shape=list(reference.shape),
        candidate_shape=list(candidate.shape),
        shape_match=True,
        element_count=int(reference.size),
        max_abs_error=errors.max_abs,
        mean_abs_error=errors.mean_abs,
        max_rel_error=errors.max_rel,
        mean_rel_error=errors.mean_rel,
        cosine_similarity=cosine,
        cosine_min_per_sample=cosine_min,
        violations=violations,
        non_finite_candidate=non_finite,
        top1_agreement=agreement.top1,
        topk=agreement.topk,
        topk_agreement=agreement.topk_agreement,
        kl_divergence=agreement.kl,
        reference_stats=_stats(ref),
        candidate_stats=_stats(cand),
        passed=not failures,
        failures=failures,
    )


def compare_outputs(
    reference: Mapping[str, Array], candidate: Mapping[str, Array], tolerance: Tolerance
) -> list[TensorComparison]:
    """Compare named outputs. Missing or unexpected outputs are reported as failed comparisons."""
    results: list[TensorComparison] = []
    for name, ref in reference.items():
        if name not in candidate:
            results.append(
                TensorComparison(
                    name=name,
                    dtype=str(ref.dtype),
                    reference_shape=list(ref.shape),
                    candidate_shape=[],
                    shape_match=False,
                    element_count=int(ref.size),
                    passed=False,
                    failures=["output is missing from the candidate"],
                )
            )
        else:
            results.append(compare_tensors(name, ref, candidate[name], tolerance))
    for name, extra in candidate.items():
        if name not in reference:
            results.append(
                TensorComparison(
                    name=name,
                    dtype=str(extra.dtype),
                    reference_shape=[],
                    candidate_shape=list(extra.shape),
                    shape_match=False,
                    element_count=int(extra.size),
                    passed=False,
                    failures=["candidate has an output the reference does not"],
                )
            )
    return results


def worst_comparison(comparisons: Sequence[TensorComparison]) -> TensorComparison:
    """The most informative comparison of one output across samples: a failing one if any, otherwise
    the one with the largest absolute error."""
    failing = [c for c in comparisons if not c.passed]
    pool = failing or list(comparisons)
    return max(pool, key=lambda c: c.max_abs_error if c.max_abs_error is not None else float("inf"))
