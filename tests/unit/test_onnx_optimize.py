from __future__ import annotations

import typing
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
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
from trtship.config import OptimizePass
from trtship.errors import ArtifactConflictError, ConfigError, ExportError
from trtship.onnx import OptimizeResult, optimize_onnx, validate_onnx
from trtship.onnx import optimize as opt
from trtship.utils.hashing import sha256_file

F32 = TensorProto.FLOAT


def vi(name: str, shape: list[int | str], elem: int = F32) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem, shape)


def model_of(
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto] | None = None,
) -> onnx.ModelProto:
    graph = helper.make_graph(nodes, "g", inputs, outputs, initializers or [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def run(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return list(session.run(None, feeds))


def op_types(model: onnx.ModelProto) -> list[str]:
    return [n.op_type for n in model.graph.node]


X2 = np.array([1.0, -2.0], dtype=np.float32)


def init(name: str, values: list[float], dtype: type = np.float32) -> onnx.TensorProto:
    return numpy_helper.from_array(np.array(values, dtype=dtype), name=name)


# --------------------------------------------------------------------------- extract_constants


def test_constant_nodes_become_initializers_without_changing_results() -> None:
    const = helper.make_node("Constant", [], ["c"], value=init("v", [10.0, 20.0]))
    model = model_of(
        [const, helper.make_node("Add", ["x", "c"], ["y"])], [vi("x", [2])], [vi("y", [2])]
    )
    before = run(model, {"x": X2})
    assert opt.extract_constants(model) == 1
    assert op_types(model) == ["Add"]
    assert [i.name for i in model.graph.initializer] == ["c"]
    onnx.checker.check_model(model)
    np.testing.assert_array_equal(run(model, {"x": X2})[0], before[0])


def test_constants_that_are_outputs_or_non_tensor_are_left_alone() -> None:
    as_output = helper.make_node("Constant", [], ["y"], value=init("v", [1.0]))
    model = model_of([as_output], [vi("x", [1])], [vi("y", [1])])
    assert opt.extract_constants(model) == 0
    assert op_types(model) == ["Constant"]

    scalar = helper.make_node("Constant", [], ["c"], value_float=2.0)
    model = model_of(
        [scalar, helper.make_node("Mul", ["x", "c"], ["y"])], [vi("x", [2])], [vi("y", [2])]
    )
    assert opt.extract_constants(model) == 0


def test_extract_constants_skips_old_ir_versions() -> None:
    const = helper.make_node("Constant", [], ["c"], value=init("v", [1.0]))
    model = model_of(
        [const, helper.make_node("Add", ["x", "c"], ["y"])], [vi("x", [1])], [vi("y", [1])]
    )
    model.ir_version = 3
    assert opt.extract_constants(model) == 0


# --------------------------------------------------------------------------- eliminate_identity


def test_identity_chains_are_removed() -> None:
    nodes = [
        helper.make_node("Identity", ["x"], ["a"]),
        helper.make_node("Identity", ["a"], ["b"]),
        helper.make_node("Relu", ["b"], ["y"]),
    ]
    model = model_of(nodes, [vi("x", [2])], [vi("y", [2])])
    before = run(model, {"x": X2})
    assert opt.eliminate_identity(model) == 2
    assert op_types(model) == ["Relu"]
    assert list(model.graph.node[0].input) == ["x"]
    np.testing.assert_array_equal(run(model, {"x": X2})[0], before[0])


def test_identity_producing_a_graph_output_is_kept() -> None:
    nodes = [
        helper.make_node("Relu", ["x"], ["r"]),
        helper.make_node("Identity", ["r"], ["y"]),
    ]
    model = model_of(nodes, [vi("x", [2])], [vi("y", [2])])
    assert opt.eliminate_identity(model) == 0
    assert op_types(model) == ["Relu", "Identity"]


def test_identity_removal_rewires_reads_inside_subgraphs() -> None:
    """The outer Identity's output is read inside an If branch, which must be renamed too."""
    then_branch = helper.make_graph(
        [helper.make_node("Add", ["x2", "w"], ["t"])], "then", [], [vi("t", [2])]
    )
    else_branch = helper.make_graph(
        [helper.make_node("Neg", ["x2"], ["e"])], "else", [], [vi("e", [2])]
    )
    nodes = [
        helper.make_node("Identity", ["x"], ["x2"]),
        helper.make_node("If", ["cond"], ["y"], then_branch=then_branch, else_branch=else_branch),
    ]
    model = model_of(
        nodes,
        [vi("x", [2]), vi("cond", [], TensorProto.BOOL)],
        [vi("y", [2])],
        [init("w", [100.0, 200.0])],
    )
    feeds_true = {"x": X2, "cond": np.array(True)}
    feeds_false = {"x": X2, "cond": np.array(False)}
    before = (run(model, feeds_true)[0], run(model, feeds_false)[0])

    assert opt.eliminate_identity(model) == 1
    assert op_types(model) == ["If"]
    inner_reads = {
        n for sub in opt._subgraphs(model.graph.node[0]) for m in sub.node for n in m.input
    }
    assert "x2" not in inner_reads
    assert "x" in inner_reads
    onnx.checker.check_model(model)
    np.testing.assert_array_equal(run(model, feeds_true)[0], before[0])
    np.testing.assert_array_equal(run(model, feeds_false)[0], before[1])


# ---------------------------------------- deduplicate_initializers


def _two_adds(w1: onnx.TensorProto, w2: onnx.TensorProto, *, dtype: int = F32) -> onnx.ModelProto:
    nodes = [
        helper.make_node("Add", ["x", w1.name], ["a"]),
        helper.make_node("Add", ["a", w2.name], ["y"]),
    ]
    return model_of(nodes, [vi("x", [2], dtype)], [vi("y", [2], dtype)], [w1, w2])


def test_identical_initializers_are_merged() -> None:
    model = _two_adds(init("w1", [1.0, 2.0]), init("w2", [1.0, 2.0]))
    before = run(model, {"x": X2})
    assert opt.deduplicate_initializers(model) == 1
    assert [i.name for i in model.graph.initializer] == ["w1"]
    assert [n.input[1] for n in model.graph.node] == ["w1", "w1"]
    np.testing.assert_array_equal(run(model, {"x": X2})[0], before[0])


@pytest.mark.parametrize(
    "second",
    [
        init("w2", [1.0, 2.5]),  # different values
    ],
)
def test_different_values_are_not_merged(second: onnx.TensorProto) -> None:
    model = _two_adds(init("w1", [1.0, 2.0]), second)
    assert opt.deduplicate_initializers(model) == 0
    assert len(model.graph.initializer) == 2


def test_dtype_and_shape_are_part_of_the_identity() -> None:
    doubles = numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float64), name="w2")
    nodes = [
        helper.make_node("Cast", ["w2"], ["w2f"], to=F32),
        helper.make_node("Add", ["x", "w1"], ["a"]),
        helper.make_node("Add", ["a", "w2f"], ["y"]),
    ]
    model = model_of(nodes, [vi("x", [2])], [vi("y", [2])], [init("w1", [1.0, 2.0]), doubles])
    assert opt.deduplicate_initializers(model) == 0

    column = numpy_helper.from_array(np.array([[1.0], [2.0]], dtype=np.float32), name="w2")
    nodes = [
        helper.make_node("Reshape", ["w2", "shape"], ["w2f"]),
        helper.make_node("Add", ["x", "w1"], ["a"]),
        helper.make_node("Add", ["a", "w2f"], ["y"]),
    ]
    shape = numpy_helper.from_array(np.array([2], dtype=np.int64), name="shape")
    model = model_of(nodes, [vi("x", [2])], [vi("y", [2])], [init("w1", [1.0, 2.0]), column, shape])
    assert opt.deduplicate_initializers(model) == 0


def test_initializers_that_are_graph_inputs_are_not_merged() -> None:
    w1, w2 = init("w1", [1.0, 2.0]), init("w2", [1.0, 2.0])
    nodes = [
        helper.make_node("Add", ["x", "w1"], ["a"]),
        helper.make_node("Add", ["a", "w2"], ["y"]),
    ]
    model = model_of(nodes, [vi("x", [2]), vi("w2", [2])], [vi("y", [2])], [w1, w2])
    assert opt.deduplicate_initializers(model) == 0


# ---------------------------------------- dead nodes / initializers


def test_dead_branches_are_removed_but_live_nodes_kept() -> None:
    nodes = [
        helper.make_node("Neg", ["x"], ["dead1"]),
        helper.make_node("Abs", ["dead1"], ["dead2"]),
        helper.make_node("Relu", ["x"], ["y"]),
    ]
    model = model_of(nodes, [vi("x", [2])], [vi("y", [2])])
    assert opt.eliminate_dead_nodes(model) == 2
    assert op_types(model) == ["Relu"]
    assert opt.eliminate_dead_nodes(model) == 0


def test_nodes_feeding_only_a_subgraph_are_not_dead() -> None:
    then_branch = helper.make_graph(
        [helper.make_node("Neg", ["helper"], ["t"])], "t", [], [vi("t", [2])]
    )
    else_branch = helper.make_graph(
        [helper.make_node("Abs", ["helper"], ["e"])], "e", [], [vi("e", [2])]
    )
    nodes = [
        helper.make_node("Relu", ["x"], ["helper"]),  # read only inside the branches
        helper.make_node("If", ["cond"], ["y"], then_branch=then_branch, else_branch=else_branch),
    ]
    model = model_of(nodes, [vi("x", [2]), vi("cond", [], TensorProto.BOOL)], [vi("y", [2])])
    assert opt.eliminate_dead_nodes(model) == 0
    assert op_types(model) == ["Relu", "If"]


def test_unused_initializers_are_removed_but_subgraph_reads_count_as_uses() -> None:
    then_branch = helper.make_graph(
        [helper.make_node("Add", ["x", "w_inner"], ["t"])], "t", [], [vi("t", [2])]
    )
    else_branch = helper.make_graph(
        [helper.make_node("Neg", ["x"], ["e"])], "e", [], [vi("e", [2])]
    )
    nodes = [
        helper.make_node("If", ["cond"], ["y"], then_branch=then_branch, else_branch=else_branch)
    ]
    model = model_of(
        nodes,
        [vi("x", [2]), vi("cond", [], TensorProto.BOOL)],
        [vi("y", [2])],
        [init("w_inner", [1.0, 1.0]), init("junk", [9.0])],
    )
    assert opt.eliminate_unused_initializers(model) == 1
    assert [i.name for i in model.graph.initializer] == ["w_inner"]
    onnx.checker.check_model(model)
    assert run(model, {"x": X2, "cond": np.array(True)})[0].tolist() == [2.0, -1.0]


def test_infer_shapes_populates_value_info() -> None:
    model = model_of(
        [helper.make_node("Relu", ["x"], ["mid"]), helper.make_node("Neg", ["mid"], ["y"])],
        [vi("x", [2])],
        [vi("y", [2])],
    )
    assert len(model.graph.value_info) == 0
    assert opt.infer_shapes(model) == 1
    assert [v.name for v in model.graph.value_info] == ["mid"]


# --------------------------------------------------------------------------- optimize_onnx


def test_pass_names_match_the_config_type() -> None:
    assert set(opt.PASS_ORDER) == set(typing.get_args(OptimizePass))
    assert set(opt.PASS_ORDER) == set(opt._PASSES)


def test_baked_model_gets_its_constants_extracted(tmp_path: Path) -> None:
    config = build_config(
        "baked_batch",
        [{"name": "x", "shape": ["batch", 6]}],
        profile={"x": {"min": [1, 6], "opt": [4, 6], "max": [8, 6]}},
    )
    _, signature = export_to(config, tmp_path / "orig.onnx")
    source_hash = sha256_file(tmp_path / "orig.onnx")

    result = optimize_onnx(tmp_path / "orig.onnx", tmp_path / "optimized.onnx", signature)
    assert result.changed
    assert result.before.node_count == 6
    assert result.after.node_count == 3
    assert result.before.op_counts["Constant"] == 3
    assert "Constant" not in result.after.op_counts
    by_name = {p.name: p.changes for p in result.passes}
    assert by_name["extract_constants"] == 3
    assert result.after.initializer_count == 3
    # provenance and integrity
    assert sha256_file(tmp_path / "orig.onnx") == source_hash  # the original is untouched
    assert result.source_sha256 == source_hash
    assert result.sha256 == sha256_file(tmp_path / "optimized.onnx")
    assert result.size_bytes == (tmp_path / "optimized.onnx").stat().st_size
    props = {p.key: p.value for p in onnx.load(str(tmp_path / "optimized.onnx")).metadata_props}
    assert props["trtship.optimized_from_sha256"] == source_hash
    assert props["trtship.optimize_passes"] == ",".join(opt.PASS_ORDER)
    assert props["trtship.weights_sha256"]  # metadata from the export survives
    assert sorted(p.name for p in tmp_path.iterdir()) == ["optimized.onnx", "orig.onnx"]


def test_optimized_model_still_matches_pytorch(tmp_path: Path) -> None:
    config = build_config(
        "token_classifier", TOKEN_INPUTS, profile=TOKEN_PROFILE, output_names=["logits", "hidden"]
    )
    model, signature = export_to(config, tmp_path / "orig.onnx")
    result = optimize_onnx(tmp_path / "orig.onnx", tmp_path / "optimized.onnx", signature)
    assert result.after.node_count < result.before.node_count
    report = validate_onnx(tmp_path / "optimized.onnx", model, signature, config)
    assert report.passed, report.failures


def test_a_model_with_nothing_to_optimize_is_unchanged_and_the_run_is_idempotent(
    tmp_path: Path,
) -> None:
    config = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)
    _, signature = export_to(config, tmp_path / "orig.onnx")
    first = optimize_onnx(tmp_path / "orig.onnx", tmp_path / "opt1.onnx", signature)
    assert first.before.op_counts == first.after.op_counts
    assert first.after.node_count == first.before.node_count

    config2 = build_config(
        "baked_batch",
        [{"name": "x", "shape": ["batch", 6]}],
        profile={"x": {"min": [1, 6], "opt": [4, 6], "max": [8, 6]}},
    )
    _, sig2 = export_to(config2, tmp_path / "b.onnx")
    once = optimize_onnx(tmp_path / "b.onnx", tmp_path / "b1.onnx", sig2)
    twice = optimize_onnx(tmp_path / "b1.onnx", tmp_path / "b2.onnx", sig2)
    assert once.changed
    assert not twice.changed
    assert twice.before.node_count == twice.after.node_count


def test_pass_selection_and_canonical_order(tmp_path: Path) -> None:
    config = build_config(
        "baked_batch",
        [{"name": "x", "shape": ["batch", 6]}],
        profile={"x": {"min": [1, 6], "opt": [4, 6], "max": [8, 6]}},
    )
    _, signature = export_to(config, tmp_path / "orig.onnx")
    result = optimize_onnx(
        tmp_path / "orig.onnx",
        tmp_path / "o.onnx",
        signature,
        passes=["infer_shapes", "extract_constants"],  # listed out of order
    )
    assert [p.name for p in result.passes] == ["extract_constants", "infer_shapes"]
    with pytest.raises(ConfigError, match="unknown optimization pass"):
        optimize_onnx(tmp_path / "orig.onnx", tmp_path / "x.onnx", signature, passes=["bogus"])
    assert not (tmp_path / "x.onnx").exists()


def test_existing_destinations_and_self_overwrites_are_refused(tmp_path: Path) -> None:
    config = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)
    _, signature = export_to(config, tmp_path / "orig.onnx")
    (tmp_path / "taken.onnx").write_bytes(b"keep")
    with pytest.raises(ArtifactConflictError, match="refusing to overwrite"):
        optimize_onnx(tmp_path / "orig.onnx", tmp_path / "taken.onnx", signature)
    assert (tmp_path / "taken.onnx").read_bytes() == b"keep"
    with pytest.raises(ArtifactConflictError):
        optimize_onnx(tmp_path / "orig.onnx", tmp_path / "orig.onnx", signature)


def test_a_pass_that_breaks_the_interface_is_rejected_and_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)
    _, signature = export_to(config, tmp_path / "orig.onnx")

    def rename_output(model: onnx.ModelProto) -> int:
        model.graph.output[0].name = "renamed"
        return 1

    monkeypatch.setitem(opt._PASSES, "extract_constants", rename_output)
    with pytest.raises(ExportError, match="changed the model's interface") as info:
        optimize_onnx(tmp_path / "orig.onnx", tmp_path / "o.onnx", signature)
    assert info.value.details["problems"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["orig.onnx"]


def test_result_round_trips_through_json(tmp_path: Path) -> None:
    config = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)
    _, signature = export_to(config, tmp_path / "orig.onnx")
    result = optimize_onnx(tmp_path / "orig.onnx", tmp_path / "o.onnx", signature)
    assert OptimizeResult.model_validate_json(result.model_dump_json()) == result
