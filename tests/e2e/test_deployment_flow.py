"""PyTorch -> ONNX -> engine -> Triton repository -> served inference, with stand-ins for the GPU.

Real: model loading, export, ONNX validation, optimization, provenance, the Triton repository and
its ``config.pbtxt``, the tritonclient over real sockets, validation and benchmarking of the served
model. Stand-ins (see tests/fakes): TensorRT (a fake builder that writes a placeholder plan) and
Triton itself (a KServe v2 stub that runs the optimized ONNX on CPU and is configured from the
generated ``config.pbtxt``). So this proves the wiring and that the generated repository describes
the model consistently; it does not prove that a real engine builds or that real Triton loads it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2 as pb
from typer.testing import CliRunner

from tests.fakes.fake_tensorrt import FakeOptions, make_fake_trt
from tests.fakes.fake_triton import FakeTritonServer, StubModel, StubTensor
from tests.helpers import (
    MLP_INPUT,
    MLP_PROFILE,
    build_config,
    default_stages_without,
    fake_environment,
)
from trtship.artifacts import ArtifactStore, ArtifactType, RunDirectory, RunStatus
from trtship.cli.main import app
from trtship.config import TrtshipConfig
from trtship.onnx.runtime import OrtSession
from trtship.pipeline import Pipeline
from trtship.tensorrt import build as trt_build

pytestmark = pytest.mark.e2e

runner = CliRunner()
MakeRun = Callable[..., RunDirectory]
DATATYPES = {pb.TYPE_FP32: "FP32", pb.TYPE_INT64: "INT64", pb.TYPE_FP16: "FP16"}


def test_export_to_served_inference(
    tmp_path: Path, make_run: MakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_build = trt_build.build_engine
    trt, _ = make_fake_trt(FakeOptions())
    monkeypatch.setattr(
        "trtship.pipeline.stages.build_stage.build_engine",
        lambda *a, **k: real_build(*a, trt=trt, **k),
    )

    data: dict[str, Any] = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE).model_dump(
        mode="json"
    )
    data["tensorrt"]["precisions"] = ["fp16"]
    data["triton"] = {"repository_dir": str(tmp_path / "served")}
    data["validation"] = {"num_samples": 2}
    data["benchmark"] = {"batch_sizes": [1, 4], "warmup_iters": 2, "iters": 5, "concurrency": [1]}
    data["artifacts"] = {"root": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")}
    config = TrtshipConfig.model_validate(data)

    # 1. the pipeline: inspect, export, validate, optimize, build (fake engine), package
    stages = default_stages_without("validate_engine", "benchmark")
    pipeline = Pipeline(config, make_run("flow"), stages, fake_environment(gpu_ok=True))
    result = pipeline.execute()
    assert result.status is RunStatus.SUCCEEDED
    assert [s.name for s in result.stages] == [
        "inspect",
        "export",
        "validate",
        "optimize",
        "calibrate",
        "build",
        "package",
    ]

    store = ArtifactStore(pipeline.run)
    repo = store.absolute(store.require(ArtifactType.TRITON_REPOSITORY, needed_by="flow"))
    onnx = store.absolute(store.require(ArtifactType.ONNX_OPTIMIZED, needed_by="flow"))
    assert (repo / "m/1/model.plan").is_file()
    served_config = text_format.Parse((repo / "m/config.pbtxt").read_text(), pb.ModelConfig())

    # 2. a stub server whose interface is read from the generated config.pbtxt, running the ONNX
    ort = OrtSession(onnx)
    batched = served_config.max_batch_size > 0

    def tensor(spec: Any) -> StubTensor:
        dims = ([-1] if batched else []) + list(spec.dims)
        return StubTensor(spec.name, DATATYPES[spec.data_type], dims)

    with FakeTritonServer() as server:
        server.state.add(
            StubModel(
                name=served_config.name,
                inputs=[tensor(t) for t in served_config.input],
                outputs=[tensor(t) for t in served_config.output],
                function=ort.run,
            )
        )
        data["triton"].update(http_port=server.http_port, grpc_port=server.grpc_port)
        config_path = tmp_path / "flow.yaml"
        config_path.write_text(yaml.safe_dump(data))

        # 3. validate the served model against the PyTorch reference, on both protocols
        for protocol in ("http", "grpc"):
            validated = runner.invoke(
                app,
                [
                    *("validate", "triton", str(config_path), "--precision", "fp16", "--json"),
                    *("--protocol", protocol, "--onnx", str(onnx), "--repository", str(repo)),
                ],
            )
            assert validated.exit_code == 0, validated.output
            assert json.loads(validated.stdout)["passed"] is True

        # 4. benchmark the served model
        report_path = tmp_path / "served.json"
        benchmarked = runner.invoke(
            app,
            [
                *("benchmark", "triton", str(config_path), "--precision", "fp16"),
                *("--repository", str(repo), "-o", str(report_path)),
            ],
        )
        assert benchmarked.exit_code == 0, benchmarked.output
        report = json.loads(report_path.read_text())
        assert {m["backend"] for m in report["measurements"]} == {"triton-http"}
        assert {m["batch_size"] for m in report["measurements"]} == {1, 4}
        assert all(m["server_side"]["requests"] == 5 for m in report["measurements"])
