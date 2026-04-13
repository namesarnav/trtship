from __future__ import annotations

import math

import numpy as np
import pytest

from trtship.config import Tolerance
from trtship.validation import TensorComparison, compare_outputs, compare_tensors

STRICT = Tolerance(atol=1e-6, rtol=1e-6, cosine_min=0.9999, top1_agreement_min=1.0)
LOOSE = Tolerance(atol=1.0, rtol=0.0, cosine_min=0.0)


def f32(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def test_identical_tensors_pass_with_zero_error() -> None:
    x = np.random.default_rng(0).standard_normal((4, 6)).astype(np.float32)
    result = compare_tensors("y", x, x.copy(), STRICT)
    assert result.passed
    assert result.failures == []
    assert result.max_abs_error == 0.0
    assert result.mean_abs_error == 0.0
    assert result.max_rel_error == 0.0
    assert result.violations == 0
    assert result.cosine_similarity == pytest.approx(1.0)
    assert result.top1_agreement == 1.0
    assert result.kl_divergence == pytest.approx(0.0, abs=1e-12)


def test_error_metrics_match_hand_computed_values() -> None:
    ref, cand = f32(1, 2, 3), f32(1, 2, 4)
    result = compare_tensors("y", ref, cand, LOOSE)
    assert result.max_abs_error == pytest.approx(1.0)
    assert result.mean_abs_error == pytest.approx(1 / 3)
    assert result.max_rel_error == pytest.approx(1 / 3)  # |4-3| / 3
    assert result.mean_rel_error == pytest.approx(1 / 9)
    expected_cos = (1 + 4 + 12) / (math.sqrt(14) * math.sqrt(21))
    assert result.cosine_similarity == pytest.approx(expected_cos)
    assert result.element_count == 3


def test_relative_error_is_guarded_against_zero_references() -> None:
    result = compare_tensors("y", f32(0, 1), f32(1e-3, 1), LOOSE)
    assert result.max_rel_error == pytest.approx(1e-3 / 1e-6)  # floor of 1e-6, not a division by 0
    assert math.isfinite(result.max_rel_error or math.nan)


def test_atol_boundary_is_inclusive() -> None:
    tolerance = Tolerance(atol=1.0, rtol=0.0, cosine_min=0.0)
    assert compare_tensors("y", f32(5), f32(6), tolerance).passed  # diff == atol
    over = compare_tensors("y", f32(5), f32(6.001), tolerance)
    assert not over.passed
    assert over.violations == 1
    assert "1 of 1 element(s) differ beyond atol=1 + rtol=0*|reference|" in over.failures[0]


def test_rtol_scales_with_the_reference_magnitude() -> None:
    tolerance = Tolerance(atol=0.0, rtol=0.01, cosine_min=0.0)
    assert compare_tensors("y", f32(100), f32(101), tolerance).passed  # allowed 1.0
    assert not compare_tensors("y", f32(1), f32(1.5), tolerance).passed  # allowed 0.01


def test_cosine_edge_cases() -> None:
    no_error = Tolerance(atol=10, rtol=0, cosine_min=0.0)
    assert compare_tensors("y", f32(1, 0), f32(0, 1), no_error).cosine_similarity == pytest.approx(
        0.0
    )
    assert compare_tensors(
        "y", f32(1, 2), f32(-1, -2), no_error
    ).cosine_similarity == pytest.approx(-1.0)
    assert compare_tensors("y", f32(0, 0), f32(0, 0), no_error).cosine_similarity == 1.0
    assert compare_tensors("y", f32(0, 0), f32(1, 2), no_error).cosine_similarity == 0.0


def test_per_sample_cosine_catches_one_bad_row_that_the_global_value_hides() -> None:
    ref = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    cand = np.array([[1, 2, 3], [-4, -5, -6]], dtype=np.float32)
    tolerance = Tolerance(atol=100, rtol=0, cosine_min=0.9)
    result = compare_tensors("y", ref, cand, tolerance)
    assert result.cosine_min_per_sample == pytest.approx(-1.0)
    assert result.cosine_min_per_sample is not None
    assert result.cosine_similarity is not None
    assert result.cosine_similarity > result.cosine_min_per_sample
    assert not result.passed
    assert any("cosine similarity -1.000000 is below the minimum 0.9" in f for f in result.failures)


def test_non_finite_candidate_values_fail_and_are_counted() -> None:
    result = compare_tensors("y", f32(1, 2, 3), f32(1, float("nan"), float("inf")), LOOSE)
    assert not result.passed
    assert result.non_finite_candidate == 2
    assert any("2 non-finite" in f for f in result.failures)
    assert result.cosine_similarity is None  # undefined with non-finite values
    assert result.max_abs_error == 0.0  # statistics cover only the finite pairs


def test_matching_non_finite_values_are_not_a_failure() -> None:
    nan = float("nan")
    result = compare_tensors("y", f32(1, nan), f32(1, nan), LOOSE)
    assert result.passed
    assert result.non_finite_candidate == 0


def test_integer_and_bool_outputs_must_match_exactly() -> None:
    ints = compare_tensors("i", np.array([1, 2, 3]), np.array([1, 2, 4]), LOOSE)
    assert not ints.passed
    assert ints.violations == 1
    assert "exactly" in ints.failures[0]
    assert ints.cosine_similarity is None
    assert compare_tensors("i", np.array([1, 2]), np.array([1, 2]), STRICT).passed

    flags = compare_tensors("b", np.array([True, False]), np.array([True, True]), LOOSE)
    assert not flags.passed
    assert flags.violations == 1


def test_shape_mismatch_fails_without_value_metrics() -> None:
    result = compare_tensors("y", np.zeros((2, 3), np.float32), np.zeros((4, 1), np.float32), LOOSE)
    assert not result.passed
    assert not result.shape_match
    assert result.max_abs_error is None
    assert result.failures == ["shape mismatch: reference [2, 3], candidate [4, 1]"]


def test_top1_and_topk_agreement_definitions() -> None:
    ref = np.array([[3, 2, 1], [3, 2, 1]], dtype=np.float32)  # reference picks class 0 twice
    tolerance = Tolerance(atol=100, rtol=0, cosine_min=0, topk=2)

    disagree = np.array([[3, 2, 1], [1, 3, 2]], dtype=np.float32)  # sample 2: top-2 = {1, 2}
    r = compare_tensors("y", ref, disagree, tolerance)
    assert r.top1_agreement == 0.5
    assert r.topk == 2
    assert r.topk_agreement == 0.5

    near = np.array(
        [[3, 2, 1], [2, 3, 1]], dtype=np.float32
    )  # sample 2: top-2 = {0, 1} has class 0
    r = compare_tensors("y", ref, near, tolerance)
    assert r.top1_agreement == 0.5
    assert r.topk_agreement == 1.0


def test_topk_is_capped_by_the_number_of_classes() -> None:
    x = np.array([[3, 2, 1]], dtype=np.float32)
    result = compare_tensors("y", x, x, Tolerance(atol=1, rtol=0, cosine_min=0, topk=5))
    assert result.topk == 3
    assert result.topk_agreement == 1.0


def test_agreement_thresholds_are_enforced_only_when_configured() -> None:
    ref = np.array([[3, 2, 1], [3, 2, 1]], dtype=np.float32)
    cand = np.array([[3, 2, 1], [1, 3, 2]], dtype=np.float32)
    base = {"atol": 100.0, "rtol": 0.0, "cosine_min": 0.0, "topk": 2}
    assert compare_tensors("y", ref, cand, Tolerance(**base)).passed
    top1 = compare_tensors("y", ref, cand, Tolerance(**base, top1_agreement_min=0.9))
    assert not top1.passed
    assert "top-1 agreement 0.5000 is below the minimum 0.9" in top1.failures[0]
    topk = compare_tensors("y", ref, cand, Tolerance(**base, topk_agreement_min=0.9))
    assert "top-2 agreement 0.5000 is below the minimum 0.9" in topk.failures[0]


@pytest.mark.parametrize("shape", [(2, 3, 4), (5,), (4, 1)])
def test_classification_metrics_only_apply_to_rank2_multiclass_outputs(
    shape: tuple[int, ...],
) -> None:
    x = np.random.default_rng(1).standard_normal(shape).astype(np.float32)
    result = compare_tensors("y", x, x, STRICT)
    assert result.top1_agreement is None
    assert result.topk_agreement is None
    assert result.kl_divergence is None
    assert result.passed


def test_kl_divergence_is_positive_for_different_distributions() -> None:
    ref = np.array([[4.0, 0.0, 0.0]], dtype=np.float32)
    cand = np.array([[0.0, 4.0, 0.0]], dtype=np.float32)
    kl = compare_tensors("y", ref, cand, LOOSE).kl_divergence
    assert kl is not None
    p = np.exp([4, 0, 0]) / np.exp([4, 0, 0]).sum()
    q = np.exp([0, 4, 0]) / np.exp([0, 4, 0]).sum()
    assert kl == pytest.approx(float(np.sum(p * np.log(p / q))))


def test_distribution_statistics_are_reported() -> None:
    result = compare_tensors("y", f32(1, 2, 3), f32(2, 3, 4), LOOSE)
    assert result.reference_stats is not None
    assert result.reference_stats.mean == pytest.approx(2.0)
    assert result.reference_stats.min == 1.0
    assert result.candidate_stats is not None
    assert result.candidate_stats.max == 4.0
    nan_only = compare_tensors("y", f32(float("nan")), f32(float("nan")), LOOSE)
    assert nan_only.reference_stats is None


def test_float16_inputs_are_compared_in_float64() -> None:
    ref = np.array([1.0, 2.0], dtype=np.float16)
    result = compare_tensors("y", ref, ref + np.float16(0.5), LOOSE)
    assert result.dtype == "float16"
    assert result.max_abs_error == pytest.approx(0.5)


def test_empty_tensors_pass() -> None:
    empty = np.zeros((0, 3), dtype=np.float32)
    result = compare_tensors("y", empty, empty, STRICT)
    assert result.passed
    assert result.element_count == 0


def test_compare_outputs_reports_missing_and_unexpected_outputs() -> None:
    ref = {"a": f32(1, 2), "b": f32(3)}
    cand = {"a": f32(1, 2), "c": f32(9)}
    results = {r.name: r for r in compare_outputs(ref, cand, STRICT)}
    assert results["a"].passed
    assert results["b"].failures == ["output is missing from the candidate"]
    assert results["c"].failures == ["candidate has an output the reference does not"]
    assert not results["b"].passed
    assert not results["c"].passed


def test_comparison_round_trips_through_json() -> None:
    result = compare_tensors("y", f32(1, 2, 3), f32(1, 2, 4), LOOSE)
    assert TensorComparison.model_validate_json(result.model_dump_json()) == result
