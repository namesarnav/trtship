from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import onnx
import pytest
from typer.testing import CliRunner

from trtship.cli.main import app

runner = CliRunner()
Writer = Callable[..., Path]
WIDE = {"COLUMNS": "200"}


@pytest.fixture
def config_path(config_dict: dict[str, Any], write_config: Writer) -> str:
    return str(write_config(config_dict, "c.yaml"))


def test_export_writes_a_verified_onnx_model(config_path: str, tmp_path: Path) -> None:
    target = tmp_path / "out" / "m.onnx"
    result = runner.invoke(app, ["export", config_path, "-o", str(target)], env=WIDE)
    assert result.exit_code == 0, result.output
    for expected in ("exported", "opset     17", "torchscript exporter", "inputs    x", "batch"):
        assert expected in result.output
    proto = onnx.load(str(target))
    onnx.checker.check_model(proto)
    assert proto.graph.input[0].name == "x"


def test_export_json_output(config_path: str, tmp_path: Path) -> None:
    target = tmp_path / "m.onnx"
    result = runner.invoke(app, ["export", config_path, "-o", str(target), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["path"] == str(target)
    assert len(data["sha256"]) == 64
    assert data["metadata"]["dynamic_axes"] == {"x": {"0": "batch"}, "output": {"0": "batch"}}
    assert data["metadata"]["trace_sizes"] == {"batch": 4}


def test_export_refuses_to_overwrite(config_path: str, tmp_path: Path) -> None:
    target = tmp_path / "m.onnx"
    target.write_bytes(b"keep me")
    result = runner.invoke(app, ["export", config_path, "-o", str(target)])
    assert result.exit_code == 11
    assert "refusing to overwrite" in result.output
    assert target.read_bytes() == b"keep me"


def test_export_unsupported_operator_exits_5(
    config_dict: dict[str, Any], write_config: Writer, tmp_path: Path
) -> None:
    config_dict["model"]["factory"] = "trtship_fixtures.models:unsupported_op"
    config_dict["model"]["inputs"] = [{"name": "x", "shape": ["batch", 8]}]
    config_dict["tensorrt"]["profiles"] = []
    result = runner.invoke(
        app,
        ["export", str(write_config(config_dict, "c.yaml")), "-o", str(tmp_path / "m.onnx")],
        env=WIDE,
    )
    assert result.exit_code == 5
    assert "aten::fft_rfft" in result.output or "ONNX export failed" in result.output
    assert not (tmp_path / "m.onnx").exists()


def test_export_requires_an_output_path(config_path: str) -> None:
    assert runner.invoke(app, ["export", config_path]).exit_code == 2


def test_export_invalid_config_exits_2(
    config_dict: dict[str, Any], write_config: Writer, tmp_path: Path
) -> None:
    config_dict["export"] = {"opset": 3}
    result = runner.invoke(
        app, ["export", str(write_config(config_dict, "c.yaml")), "-o", str(tmp_path / "m.onnx")]
    )
    assert result.exit_code == 2


@pytest.mark.skipif(
    importlib.util.find_spec("onnxscript") is None,
    reason="the torch.export exporter needs onnxscript (uv sync --extra dynamo)",
)
def test_export_with_the_dynamo_exporter(config_path: str, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "export",
            config_path,
            "-o",
            str(tmp_path / "m.onnx"),
            "--set",
            "export.dynamo=true",
            "--set",
            "export.opset=18",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["metadata"]["exporter"] == "dynamo"
