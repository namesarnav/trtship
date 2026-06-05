"""Acceptance tests that run real TensorRT on a real GPU.

They are skipped, with an explicit reason, wherever a usable NVIDIA GPU and the TensorRT Python
package are absent (see ``pytest_collection_modifyitems`` in tests/conftest.py). A skip is not a
pass: CI reports these separately, and `IMPLEMENTATION_STATUS.md` does not mark the TensorRT builder
as verified until they have run.

Run them on a GPU machine with:  uv sync --extra trt && pytest -m tensorrt -v
"""

from __future__ import annotations

from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from tests.helpers import MLP_INPUT, MLP_PROFILE, build_config, export_to
from trtship.config import Precision
from trtship.errors import EngineBuildError
from trtship.tensorrt import build_engine, parse_version

pytestmark = [pytest.mark.gpu, pytest.mark.tensorrt]


@pytest.fixture
def mlp(tmp_path: Path):  # type: ignore[no-untyped-def]
    config = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)
    _, signature = export_to(config, tmp_path / "m.onnx")
    return config, signature, tmp_path / "m.onnx"


@pytest.mark.parametrize("precision", [Precision.FP32, Precision.FP16])
def test_builds_and_describes_an_engine(  # type: ignore[no-untyped-def]
    mlp, tmp_path: Path, precision: Precision
) -> None:
    config, signature, onnx_path = mlp
    result = build_engine(
        onnx_path, signature, config, precision, tmp_path / f"{precision.value}.plan"
    )
    assert result.size_bytes > 0
    assert parse_version(result.tensorrt_version).major >= 8
    (x,) = result.engine.inputs
    (y,) = result.engine.outputs
    assert (x.name, x.shape) == ("x", [-1, 16])
    assert (y.name, y.shape[1:]) == ("output", [4])
    assert x.profiles[0].min == [1, 16]
    assert x.profiles[0].opt == [4, 16]
    assert x.profiles[0].max == [8, 16]
    assert result.engine.max_batch_size() == 8
    if precision is Precision.FP16:
        assert "FP16" in result.builder_flags


def test_unsupported_operators_are_named(mlp, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    config, signature, _ = mlp
    node = helper.make_node("Frobnicate", ["x"], ["output"], domain="com.acme")
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 4])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.acme", 1)]
    )
    path = tmp_path / "custom.onnx"
    onnx.save(model, str(path))
    with pytest.raises(EngineBuildError) as info:
        build_engine(path, signature, config, Precision.FP32, tmp_path / "c.plan")
    assert "Frobnicate" in info.value.details["unsupported_operators"]
