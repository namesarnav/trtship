from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import onnx
import pytest
from onnx import TensorProto, helper

from trtship.config import TrtshipConfig
from trtship.errors import ArtifactConflictError, EnvironmentUnavailableError, ExportError
from trtship.export import ExportResult, check_onnx_signature, export_onnx
from trtship.models import ModelSignature, infer_signature, load_model
from trtship.specs import DType, TensorSpec
from trtship.utils.hashing import sha256_file

F = "trtship_fixtures.models"
HAS_ONNXSCRIPT = importlib.util.find_spec("onnxscript") is not None
needs_dynamo = pytest.mark.skipif(
    not HAS_ONNXSCRIPT, reason="the torch.export exporter needs onnxscript (uv sync --extra dynamo)"
)
X = {"name": "x", "shape": ["batch", 16]}
TOKENS = [
    {"name": "input_ids", "dtype": "int64", "shape": ["batch", "seq"], "value_range": [0, 100]},
    {"name": "attention_mask", "dtype": "int64", "shape": ["batch", "seq"], "value_range": [0, 2]},
]
TOKEN_PROFILE = {
    "input_ids": {"min": [1, 8], "opt": [4, 32], "max": [16, 128]},
    "attention_mask": {"min": [1, 8], "opt": [4, 32], "max": [16, 128]},
}


def make_config(
    factory: str,
    inputs: list[dict[str, Any]],
    *,
    profile: dict[str, Any] | None = None,
    export: dict[str, Any] | None = None,
    **model_extra: Any,
) -> TrtshipConfig:
    data: dict[str, Any] = {
        "model": {
            "name": "m",
            "kind": "module",
            "factory": f"{F}:{factory}",
            "inputs": inputs,
            **model_extra,
        },
        "export": export or {},
    }
    if profile:
        data["tensorrt"] = {"profiles": [{"inputs": profile}]}
    return TrtshipConfig.model_validate(data)


def run_export(config: TrtshipConfig, path: Path) -> tuple[ExportResult, onnx.ModelProto]:
    model = load_model(config.model)
    signature = infer_signature(model, config.model, config.tensorrt.profiles, seed=config.seed)
    result = export_onnx(model, signature, config, path)
    return result, onnx.load(str(path))


def dims(value_info: onnx.ValueInfoProto) -> list[int | str]:
    return [d.dim_param or d.dim_value for d in value_info.type.tensor_type.shape.dim]


MLP_PROFILE = {"x": {"min": [1, 16], "opt": [4, 16], "max": [8, 16]}}


# --------------------------------------------------------------------------- torchscript exporter


def test_static_export(tmp_path: Path) -> None:
    config = make_config("tiny_mlp", [{"name": "x", "shape": [3, 16]}])
    result, proto = run_export(config, tmp_path / "m.onnx")
    onnx.checker.check_model(proto)
    assert dims(proto.graph.input[0]) == [3, 16]
    assert dims(proto.graph.output[0]) == [3, 4]
    assert result.metadata.dynamic_axes == {}
    assert result.metadata.exporter == "torchscript"
    assert result.metadata.trace_sizes == {}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.onnx"]  # no temp leftovers


def test_dynamic_batch_export_and_metadata(tmp_path: Path) -> None:
    config = make_config("tiny_mlp", [X], profile=MLP_PROFILE)
    result, proto = run_export(config, tmp_path / "m.onnx")
    onnx.checker.check_model(proto)
    assert dims(proto.graph.input[0]) == ["batch", 16]
    assert dims(proto.graph.output[0]) == ["batch", 4]
    meta = result.metadata
    assert meta.dynamic_axes == {"x": {0: "batch"}, "output": {0: "batch"}}
    assert meta.opset == 17
    assert meta.trace_sizes == {"batch": 4}  # profile opt
    assert meta.input_names == ["x"]
    assert meta.output_names == ["output"]
    assert meta.constant_folding is True
    assert meta.model_name == "m"
    assert meta.onnx_version == onnx.__version__
    assert result.sha256 == sha256_file(tmp_path / "m.onnx")
    assert result.size_bytes == (tmp_path / "m.onnx").stat().st_size
    assert result.path == str(tmp_path / "m.onnx")


def test_opset_and_constant_folding_settings_are_applied(tmp_path: Path) -> None:
    config = make_config(
        "tiny_mlp", [X], profile=MLP_PROFILE, export={"opset": 13, "constant_folding": False}
    )
    result, proto = run_export(config, tmp_path / "m.onnx")
    assert [o.version for o in proto.opset_import if o.domain == ""] == [13]
    assert result.metadata.opset == 13
    assert result.metadata.constant_folding is False


def test_multi_input_multi_output_with_shared_symbols(tmp_path: Path) -> None:
    config = make_config(
        "token_classifier",
        TOKENS,
        profile=TOKEN_PROFILE,
        output_names=["logits", "hidden"],
    )
    result, proto = run_export(config, tmp_path / "m.onnx")
    onnx.checker.check_model(proto)
    assert [i.name for i in proto.graph.input] == ["input_ids", "attention_mask"]
    assert dims(proto.graph.input[0]) == ["batch", "seq"]
    assert dims(proto.graph.output[0]) == ["batch", 3]
    assert dims(proto.graph.output[1]) == ["batch", "seq", 8]
    assert result.metadata.trace_sizes == {"batch": 4, "seq": 32}
    assert result.metadata.output_names == ["logits", "hidden"]


def test_dict_outputs_keep_their_names(tmp_path: Path) -> None:
    config = make_config("dict_output", [{"name": "x", "shape": ["batch", 6]}])
    _, proto = run_export(config, tmp_path / "m.onnx")
    assert [o.name for o in proto.graph.output] == ["doubled", "total"]


def test_traceability_metadata_is_embedded_in_the_file(tmp_path: Path) -> None:
    config = make_config("tiny_mlp", [X], profile=MLP_PROFILE)
    model = load_model(config.model)
    _, proto = run_export(config, tmp_path / "m.onnx")
    props = {p.key: p.value for p in proto.metadata_props}
    assert props["trtship.weights_sha256"] == model.weights_sha256
    assert props["trtship.model_name"] == "m"
    assert props["trtship.exporter"] == "torchscript"


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "m.onnx"
    target.write_bytes(b"precious")
    config = make_config("tiny_mlp", [X], profile=MLP_PROFILE)
    model = load_model(config.model)
    signature = infer_signature(model, config.model, config.tensorrt.profiles)
    with pytest.raises(ArtifactConflictError, match="refusing to overwrite"):
        export_onnx(model, signature, config, target)
    assert target.read_bytes() == b"precious"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.onnx"]


def test_output_directory_is_created(tmp_path: Path) -> None:
    config = make_config("tiny_mlp", [X], profile=MLP_PROFILE)
    run_export(config, tmp_path / "deep" / "er" / "m.onnx")
    assert (tmp_path / "deep" / "er" / "m.onnx").is_file()


def test_unsupported_operator_is_a_clean_export_error(tmp_path: Path) -> None:
    config = make_config("unsupported_op", [{"name": "x", "shape": ["batch", 8]}])
    model = load_model(config.model)
    signature = infer_signature(model, config.model, [])
    with pytest.raises(ExportError, match="ONNX export failed") as info:
        export_onnx(model, signature, config, tmp_path / "m.onnx")
    assert info.value.details["operator"] == "aten::fft_rfft"
    assert info.value.details["exporter"] == "torchscript"
    assert info.value.details["opset"] == 17
    assert "opset" in (info.value.hint or "")
    assert list(tmp_path.iterdir()) == []  # neither the artifact nor a temp file remains


def test_tracer_warnings_are_captured(tmp_path: Path) -> None:
    config = make_config("data_dependent_branch", [{"name": "x", "shape": ["batch", 4]}])
    result, _ = run_export(config, tmp_path / "m.onnx")
    assert any("TracerWarning" in w for w in result.metadata.warnings)
    assert len(set(result.metadata.warnings)) == len(result.metadata.warnings)  # de-duplicated


def test_baked_batch_passes_the_structural_check_but_is_documented_as_a_limit(
    tmp_path: Path,
) -> None:
    """The graph *declares* a dynamic batch, so the structural check passes even though tracing
    froze the batch into a constant. Only running at another shape reveals it (ONNX validation)."""
    config = make_config("baked_batch", [{"name": "x", "shape": ["batch", 6]}])
    result, proto = run_export(config, tmp_path / "m.onnx")
    assert dims(proto.graph.output[0]) == ["batch", 1]
    assert result.metadata.dynamic_axes["x"] == {0: "batch"}


def test_export_json_round_trip(tmp_path: Path) -> None:
    config = make_config("tiny_mlp", [X], profile=MLP_PROFILE)
    result, _ = run_export(config, tmp_path / "m.onnx")
    assert ExportResult.model_validate_json(result.model_dump_json()) == result


# --------------------------------------------------------------------------- dynamo exporter


def test_dynamo_requires_onnxscript(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("trtship.export.onnx_export.importlib.util.find_spec", lambda name: None)
    config = make_config("tiny_mlp", [X], profile=MLP_PROFILE, export={"dynamo": True})
    model = load_model(config.model)
    signature = infer_signature(model, config.model, config.tensorrt.profiles)
    with pytest.raises(EnvironmentUnavailableError, match="onnxscript") as info:
        export_onnx(model, signature, config, tmp_path / "m.onnx")
    assert "--extra dynamo" in (info.value.hint or "")
    assert list(tmp_path.iterdir()) == []


@needs_dynamo
def test_dynamo_static_export(tmp_path: Path) -> None:
    config = make_config(
        "tiny_mlp", [{"name": "x", "shape": [3, 16]}], export={"dynamo": True, "opset": 18}
    )
    result, proto = run_export(config, tmp_path / "m.onnx")
    onnx.checker.check_model(proto)
    assert dims(proto.graph.input[0]) == [3, 16]
    assert result.metadata.exporter == "dynamo"
    assert result.metadata.opset == 18
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.onnx"]


@needs_dynamo
def test_dynamo_dynamic_export_with_shared_symbols(tmp_path: Path) -> None:
    config = make_config(
        "token_classifier",
        TOKENS,
        profile=TOKEN_PROFILE,
        export={"dynamo": True, "opset": 18},
        output_names=["logits", "hidden"],
    )
    result, proto = run_export(config, tmp_path / "m.onnx")
    onnx.checker.check_model(proto)
    assert dims(proto.graph.input[0]) == ["batch", "seq"]
    assert dims(proto.graph.input[1]) == ["batch", "seq"]
    assert [o.name for o in proto.graph.output] == ["logits", "hidden"]
    assert result.metadata.exporter == "dynamo"
    props = {p.key: p.value for p in proto.metadata_props}
    assert props["trtship.exporter"] == "dynamo"


@needs_dynamo
def test_dynamo_pinned_symbol_is_exported_static(tmp_path: Path) -> None:
    pinned = {"x": {"min": [4, 16], "opt": [4, 16], "max": [4, 16]}}
    config = make_config("tiny_mlp", [X], profile=pinned, export={"dynamo": True, "opset": 18})
    _, proto = run_export(config, tmp_path / "m.onnx")
    assert dims(proto.graph.input[0]) == [4, 16]  # the profile fixes batch; verification accepts it


# --------------------------------------------------------------------------- structural checker


def signature(inputs: list[TensorSpec], outputs: list[TensorSpec]) -> ModelSignature:
    return ModelSignature(inputs=inputs, outputs=outputs, probe_sizes=[{}])


def vi(
    name: str, dims_: list[int | str | None], elem: int = TensorProto.FLOAT
) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem, dims_)


def proto_with(
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto] | None = None,
) -> onnx.ModelProto:
    graph = helper.make_graph([], "g", inputs, outputs, initializers or [])
    return helper.make_model(graph)


SPEC_X = TensorSpec(name="x", shape=["batch", 16])
SPEC_Y = TensorSpec(name="y", shape=["batch", 4])


def test_checker_accepts_a_matching_graph() -> None:
    proto = proto_with([vi("x", ["batch", 16])], [vi("y", ["batch", 4])])
    assert check_onnx_signature(proto, signature([SPEC_X], [SPEC_Y])) == []


def test_checker_allows_renamed_output_symbols_but_not_input_symbols() -> None:
    sig = signature([SPEC_X], [SPEC_Y])
    ok = proto_with([vi("x", ["batch", 16])], [vi("y", ["s0", 4])])
    assert check_onnx_signature(ok, sig) == []
    bad = proto_with([vi("x", ["N", 16])], [vi("y", ["batch", 4])])
    (problem,) = check_onnx_signature(bad, sig)
    assert "expected symbol 'batch'" in problem


@pytest.mark.parametrize(
    ("proto", "fragment"),
    [
        (proto_with([vi("z", ["batch", 16])], [vi("y", ["batch", 4])]), "input names/order differ"),
        (
            proto_with([vi("x", ["batch", 16])], [vi("w", ["batch", 4])]),
            "output names/order differ",
        ),
        (proto_with([vi("x", ["batch", 16, 1])], [vi("y", ["batch", 4])]), "rank 3"),
        (proto_with([vi("x", ["batch", 8])], [vi("y", ["batch", 4])]), "expected static 16"),
        (proto_with([vi("x", [4, 16])], [vi("y", ["batch", 4])]), "fixed it to 4"),
        (proto_with([vi("x", ["batch", 16])], [vi("y", [4, 4])]), "fixed it to 4"),
        (
            proto_with([vi("x", ["batch", 16], TensorProto.DOUBLE)], [vi("y", ["batch", 4])]),
            "dtype",
        ),
    ],
)
def test_checker_reports_each_kind_of_mismatch(proto: onnx.ModelProto, fragment: str) -> None:
    problems = check_onnx_signature(proto, signature([SPEC_X], [SPEC_Y]))
    assert any(fragment in p for p in problems), problems


def test_checker_accepts_unnamed_dynamic_dims_and_reports_missing_shape() -> None:
    sig = signature([SPEC_X], [SPEC_Y])
    unnamed = proto_with([vi("x", ["batch", 16])], [vi("y", [None, 4])])
    assert check_onnx_signature(unnamed, sig) == []

    shapeless = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    (problem,) = check_onnx_signature(proto_with([vi("x", ["batch", 16])], [shapeless]), sig)
    assert "declares no shape" in problem


def test_checker_accepts_pinned_symbols_as_constants() -> None:
    sig = signature([SPEC_X], [SPEC_Y])
    proto = proto_with([vi("x", [4, 16])], [vi("y", [4, 4])])
    assert check_onnx_signature(proto, sig) != []  # unpinned: a constant is a mismatch
    assert check_onnx_signature(proto, sig, pinned={"batch": 4}) == []
    assert check_onnx_signature(proto, sig, pinned={"batch": 5}) != []  # wrong constant


def test_checker_ignores_initializers_listed_as_inputs() -> None:
    weight = helper.make_tensor("w", TensorProto.FLOAT, [1], [0.0])
    proto = proto_with(
        [vi("x", ["batch", 16]), vi("w", [1])], [vi("y", ["batch", 4])], initializers=[weight]
    )
    assert check_onnx_signature(proto, signature([SPEC_X], [SPEC_Y])) == []


def test_checker_int_dtypes_map() -> None:
    spec = TensorSpec(name="ids", dtype=DType.INT64, shape=["batch"])
    out = TensorSpec(name="o", dtype=DType.BOOL, shape=["batch"])
    proto = proto_with(
        [vi("ids", ["batch"], TensorProto.INT64)], [vi("o", ["batch"], TensorProto.BOOL)]
    )
    assert check_onnx_signature(proto, signature([spec], [out])) == []
