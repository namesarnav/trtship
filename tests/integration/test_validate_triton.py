"""`trtship validate triton` against a stub that serves the exported ONNX model.

The stub is not Triton (see tests/fakes/fake_triton.py): this proves the validation flow, the shape
points sent through the client, the pass/fail gating, and the report labelling, not that a
TensorRT engine served by Triton is accurate.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from tests.fakes.fake_triton import Arrays, FakeTritonServer, StubModel, StubTensor
from tests.helpers import MLP_INPUT, MLP_PROFILE, build_config, export_to
from trtship.cli.main import app
from trtship.errors import TritonError, ValidationFailedError
from trtship.onnx.runtime import OrtSession

runner = CliRunner()


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[FakeTritonServer, Path, Path]]:
    """(stub server, config file, onnx path) with a repository holding a placeholder plan."""
    config = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)
    onnx_path = tmp_path / "m.onnx"
    export_to(config, onnx_path)
    (tmp_path / "repo/m/1").mkdir(parents=True)
    (tmp_path / "repo/m/1/model.plan").write_bytes(b"placeholder plan")
    ort = OrtSession(onnx_path)
    with FakeTritonServer() as server:
        server.state.add(
            StubModel(
                name="m",
                inputs=[StubTensor("x", "FP32", [-1, 16])],
                outputs=[StubTensor(n, "FP32", [-1, 4]) for n in ort.output_names],
                function=ort.run,
            )
        )
        data = config.model_dump(mode="json")
        data["model"]["factory"] = "trtship_fixtures.models:tiny_mlp"
        data["triton"] = {
            "repository_dir": str(tmp_path / "repo"),
            "http_port": server.http_port,
            "grpc_port": server.grpc_port,
        }
        data["artifacts"] = {"root": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")}
        data["validation"] = {"num_samples": 2}
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(data))
        yield server, path, onnx_path


def invoke(config: Path, onnx: Path, *extra: str) -> Any:
    return runner.invoke(
        app, ["validate", "triton", str(config), "--onnx", str(onnx), "--precision", "fp32", *extra]
    )


@pytest.mark.parametrize("protocol", ["http", "grpc"])
def test_a_faithful_server_passes_at_every_shape_point(
    served: tuple[FakeTritonServer, Path, Path], protocol: str
) -> None:
    server, config, onnx = served
    result = invoke(config, onnx, "--protocol", protocol, "--json")
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["backend"] == f"triton-{protocol}"
    assert report["passed"] is True
    labels = [p["label"] for p in report["results"][0]["points"]]
    assert labels == ["min", "opt", "max"]
    batches = {(p, m) for p, m in server.state.infer_calls}
    assert (protocol, "m") in batches
    assert len(server.state.infer_calls) == 3 * 2  # three shape points, two samples each


def test_the_human_report_names_the_transport(served: tuple[FakeTritonServer, Path, Path]) -> None:
    _, config, onnx = served
    result = invoke(config, onnx)
    assert result.exit_code == 0
    assert "Engine validation via triton-http PASSED" in result.output


def test_a_server_that_returns_wrong_numbers_fails_validation(
    served: tuple[FakeTritonServer, Path, Path],
) -> None:
    server, config, onnx = served
    honest: Callable[[Arrays], Arrays] = server.state.models["m"].function

    def corrupted(inputs: Arrays) -> Arrays:
        return {name: value + np.float32(0.5) for name, value in honest(inputs).items()}

    server.state.models["m"].function = corrupted
    result = invoke(config, onnx)
    assert result.exit_code == ValidationFailedError.exit_code
    assert "FAILED" in result.output


def test_a_missing_served_plan_is_reported_before_contacting_the_server(
    served: tuple[FakeTritonServer, Path, Path],
) -> None:
    server, config, onnx = served
    result = invoke(config, onnx, "--repository", str(config.parent / "elsewhere"))
    assert result.exit_code == TritonError.exit_code
    assert "--repository" in result.output
    assert server.state.infer_calls == []


def test_an_unknown_protocol_is_rejected(served: tuple[FakeTritonServer, Path, Path]) -> None:
    _, config, onnx = served
    result = invoke(config, onnx, "--protocol", "carrier-pigeon")
    assert result.exit_code != 0
    assert "http or grpc" in result.output
