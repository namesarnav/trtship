from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tests.helpers import (
    MLP_INPUT,
    MLP_PROFILE,
    TOKEN_INPUTS,
    TOKEN_PROFILE,
    build_config,
    export_to,
)
from trtship.errors import ValidationFailedError
from trtship.models import infer_signature, load_model
from trtship.onnx import (
    OnnxValidationReport,
    OrtSession,
    analyze_graph,
    select_shape_points,
    validate_onnx,
)
from trtship.utils.hashing import sha256_file

# --------------------------------------------------------------------------- graph analysis


@pytest.fixture(scope="module")
def mlp_onnx(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("mlp") / "m.onnx"
    export_to(build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE), path)
    return path


def test_graph_report_for_an_exported_mlp(mlp_onnx: Path) -> None:
    report = analyze_graph(mlp_onnx)
    assert report.opset == 17
    assert report.node_count == 3
    assert report.op_counts == {"Gemm": 2, "Relu": 1}
    assert next(iter(report.op_counts)) == "Gemm"  # most frequent first
    assert report.initializer_count == 4
    assert report.initializer_bytes == 676 * 4
    assert report.inputs == ["x"]
    assert report.outputs == ["output"]
    assert report.custom_domains == []
    assert report.outputs_without_shape == []
    assert report.warnings == []
    assert report.producer.startswith("pytorch")


def test_checker_rejects_a_broken_graph(tmp_path: Path) -> None:
    node = helper.make_node("Relu", ["ghost"], ["y"])  # 'ghost' is never defined
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    path = tmp_path / "bad.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))
    with pytest.raises(ValidationFailedError, match="checker rejected") as info:
        analyze_graph(path)
    assert info.value.details["stage"] == "checker"
    assert info.value.exit_code == 6


def test_checker_rejects_a_type_inconsistent_graph(tmp_path: Path) -> None:
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.INT64, [1])],  # Relu(float) is float
    )
    path = tmp_path / "typed.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))
    with pytest.raises(ValidationFailedError):
        analyze_graph(path)


def test_unreadable_files_are_validation_errors(tmp_path: Path) -> None:
    junk = tmp_path / "junk.onnx"
    junk.write_bytes(b"not a protobuf at all")
    with pytest.raises(ValidationFailedError, match="cannot read ONNX model"):
        analyze_graph(junk)
    with pytest.raises(ValidationFailedError, match="cannot read ONNX model"):
        analyze_graph(tmp_path / "missing.onnx")


def test_custom_operator_domains_are_flagged(tmp_path: Path) -> None:
    node = helper.make_node("Frobnicate", ["x"], ["y"], domain="com.acme")
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.acme", 1)]
    )
    path = tmp_path / "custom.onnx"
    onnx.save(model, str(path))
    report = analyze_graph(path)
    assert report.custom_domains == ["com.acme"]
    assert any("com.acme" in w and "TensorRT" in w for w in report.warnings)


# --------------------------------------------------------------------------- ORT runner


def test_ort_session_is_cpu_only_and_runs(mlp_onnx: Path) -> None:
    session = OrtSession(mlp_onnx)
    assert session.providers == ["CPUExecutionProvider"]
    assert session.input_names == ["x"]
    assert session.output_names == ["output"]
    assert session.optimize is False
    assert session.version
    out = session.run({"x": np.random.default_rng(0).standard_normal((3, 16)).astype(np.float32)})
    assert out["output"].shape == (3, 4)


def test_ort_session_reports_load_and_run_failures(mlp_onnx: Path, tmp_path: Path) -> None:
    junk = tmp_path / "junk.onnx"
    junk.write_bytes(b"garbage")
    with pytest.raises(ValidationFailedError, match="cannot load") as load_info:
        OrtSession(junk)
    assert load_info.value.details["stage"] == "onnxruntime_load"

    session = OrtSession(mlp_onnx)
    with pytest.raises(ValidationFailedError, match="missing ONNX Runtime inputs"):
        session.run({})
    with pytest.raises(ValidationFailedError, match="ONNX Runtime failed") as run_info:
        session.run({"x": np.zeros((2, 5), dtype=np.float32)})  # wrong feature size
    assert run_info.value.details["input_shapes"] == {"x": [2, 5]}


def test_ort_optimization_flag_is_recorded(mlp_onnx: Path) -> None:
    assert OrtSession(mlp_onnx, optimize=True).optimize is True


# --------------------------------------------------------------------------- shape points


def _points(config_kwargs: dict[str, Any]) -> list[tuple[str, dict[str, int]]]:
    config = build_config(**config_kwargs)
    model = load_model(config.model)
    signature = infer_signature(model, config.model, config.tensorrt.profiles)
    return select_shape_points(config, signature)


def test_static_models_have_a_single_point() -> None:
    points = _points({"factory": "tiny_mlp", "inputs": [{"name": "x", "shape": [3, 16]}]})
    assert points == [("static", {})]


def test_profile_gives_min_opt_max() -> None:
    points = _points({"factory": "tiny_mlp", "inputs": [MLP_INPUT], "profile": MLP_PROFILE})
    assert points == [("min", {"batch": 1}), ("opt", {"batch": 4}), ("max", {"batch": 8})]


def test_identical_points_are_merged() -> None:
    same_min_opt = {"x": {"min": [4, 16], "opt": [4, 16], "max": [8, 16]}}
    points = _points({"factory": "tiny_mlp", "inputs": [MLP_INPUT], "profile": same_min_opt})
    assert points == [("min=opt", {"batch": 4}), ("max", {"batch": 8})]
    pinned = {"x": {"min": [4, 16], "opt": [4, 16], "max": [4, 16]}}
    points = _points({"factory": "tiny_mlp", "inputs": [MLP_INPUT], "profile": pinned})
    assert points == [("min=opt=max", {"batch": 4})]


def test_without_a_profile_the_probe_sizes_are_used() -> None:
    points = _points({"factory": "tiny_mlp", "inputs": [MLP_INPUT]})
    assert [label for label, _ in points] == ["probe-1", "probe-2"]
    assert points[0][1] != points[1][1]


# --------------------------------------------------------------------------- end to end


def _validate(
    factory: str,
    inputs: list[dict[str, Any]],
    tmp_path: Path,
    *,
    profile: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    **model_extra: Any,
) -> OnnxValidationReport:
    config = build_config(
        factory,
        inputs,
        profile=profile,
        validation=validation,
        **model_extra,
    )
    model, signature = export_to(config, tmp_path / "m.onnx")
    return validate_onnx(tmp_path / "m.onnx", model, signature, config)


def test_mlp_passes_at_every_shape_point(tmp_path: Path) -> None:
    report = _validate("tiny_mlp", [MLP_INPUT], tmp_path, profile=MLP_PROFILE)
    assert report.passed, report.failures
    assert report.failures == []
    assert [p.label for p in report.points] == ["min", "opt", "max"]
    assert all(p.samples == 8 and p.samples_passed == 8 for p in report.points)
    assert report.providers == ["CPUExecutionProvider"]
    assert report.ort_graph_optimization is False
    assert report.onnx_sha256 == sha256_file(tmp_path / "m.onnx")
    assert report.onnx_size_bytes == (tmp_path / "m.onnx").stat().st_size
    assert report.graph.op_counts == {"Gemm": 2, "Relu": 1}
    output = report.points[0].outputs[0]
    assert output.max_abs_error is not None
    assert output.max_abs_error < 1e-5
    assert (output.cosine_similarity or 0) > 0.99999
    assert output.candidate_shape == [1, 4]
    assert report.points[2].outputs[0].candidate_shape == [8, 4]


def test_multi_input_multi_output_model_passes(tmp_path: Path) -> None:
    report = _validate(
        "token_classifier",
        TOKEN_INPUTS,
        tmp_path,
        profile=TOKEN_PROFILE,
        output_names=["logits", "hidden"],
    )
    assert report.passed, report.failures
    names = {o.name for p in report.points for o in p.outputs}
    assert names == {"logits", "hidden"}
    shapes = {(p.label, o.name): o.candidate_shape for p in report.points for o in p.outputs}
    assert shapes[("max", "hidden")] == [8, 64, 8]
    assert shapes[("min", "logits")] == [1, 3]


def test_static_model_passes_at_its_single_point(tmp_path: Path) -> None:
    report = _validate("tiny_mlp", [{"name": "x", "shape": [3, 16]}], tmp_path)
    assert report.passed
    assert [p.label for p in report.points] == ["static"]


def test_validation_settings_control_samples_and_seed(tmp_path: Path) -> None:
    report = _validate(
        "tiny_mlp",
        [MLP_INPUT],
        tmp_path,
        profile=MLP_PROFILE,
        validation={"num_samples": 2, "seed": 42},
    )
    assert report.seed == 42
    assert all(p.samples == 2 for p in report.points)


def test_validation_is_deterministic(tmp_path: Path) -> None:
    config = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)
    model, signature = export_to(config, tmp_path / "m.onnx")
    first = validate_onnx(tmp_path / "m.onnx", model, signature, config)
    second = validate_onnx(tmp_path / "m.onnx", model, signature, config)
    assert first.points == second.points


def test_baked_batch_is_caught_by_running_at_other_shapes(tmp_path: Path) -> None:
    """Export's structural check cannot see this; running at min/max shapes does."""
    report = _validate(
        "baked_batch",
        [{"name": "x", "shape": ["batch", 6]}],
        tmp_path,
        profile={"x": {"min": [1, 6], "opt": [4, 6], "max": [8, 6]}},
    )
    assert not report.passed
    by_label = {p.label: p for p in report.points}
    assert by_label["opt"].passed  # the traced size looks fine
    assert not by_label["min"].passed
    assert by_label["min"].error is not None
    assert "ONNX Runtime failed" in by_label["min"].error
    assert not by_label["max"].passed
    (output,) = by_label["max"].outputs
    assert output.shape_match is False
    assert output.reference_shape == [8, 1]
    assert output.candidate_shape == [4, 1]
    with pytest.raises(ValidationFailedError, match="ONNX validation failed") as info:
        report.raise_for_failure()
    assert info.value.exit_code == 6
    assert info.value.details["failures"] == report.failures


def test_a_too_strict_tolerance_fails_with_the_measured_error(tmp_path: Path) -> None:
    exact = {"atol": 0.0, "rtol": 0.0, "cosine_min": 1.0}
    report = _validate(
        "tiny_mlp", [MLP_INPUT], tmp_path, profile=MLP_PROFILE, validation={"onnx_tolerance": exact}
    )
    assert not report.passed
    failing = [o for p in report.points for o in p.outputs if not o.passed]
    assert failing
    assert failing[0].violations > 0
    assert any("differ beyond atol=0" in f for f in report.failures)
    assert report.tolerance.atol == 0.0


def test_corrupted_weights_are_detected(tmp_path: Path) -> None:
    config = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)
    model, signature = export_to(config, tmp_path / "good.onnx")
    proto = onnx.load(str(tmp_path / "good.onnx"))
    target = next(i for i in proto.graph.initializer if len(i.dims) == 2)
    weights = numpy_helper.to_array(target).copy()
    weights[0, :] += 5.0
    target.CopyFrom(numpy_helper.from_array(weights, name=target.name))
    onnx.save(proto, str(tmp_path / "bad.onnx"))

    report = validate_onnx(tmp_path / "bad.onnx", model, signature, config)
    assert not report.passed
    worst = max(o.max_abs_error or 0.0 for p in report.points for o in p.outputs)
    assert worst > 0.1


def test_interface_mismatch_is_reported_and_no_comparison_is_attempted(tmp_path: Path) -> None:
    exported = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE, output_names=["logits"])
    export_to(exported, tmp_path / "m.onnx")
    checked = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)  # expects 'output'
    model = load_model(checked.model)
    signature = infer_signature(model, checked.model, checked.tensorrt.profiles)
    report = validate_onnx(tmp_path / "m.onnx", model, signature, checked)
    assert not report.passed
    assert report.points == []
    assert any("ONNX outputs ['logits'] differ" in f for f in report.failures)


def test_report_round_trips_through_json(tmp_path: Path) -> None:
    report = _validate("tiny_mlp", [MLP_INPUT], tmp_path, profile=MLP_PROFILE)
    assert OnnxValidationReport.model_validate_json(report.model_dump_json()) == report
    assert report.generated_at.tzinfo is not None
    assert report.schema_version == 1
