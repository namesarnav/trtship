"""The whole deployment on real hardware: TensorRT engine, Triton in Docker, served inference.

NOT VERIFIED on the development machine (no usable GPU, TensorRT or NVIDIA container runtime), so
it has never run. It is skipped, never passed, unless all of these hold: a usable NVIDIA GPU, the
TensorRT Python package, Docker with the NVIDIA runtime, and TRTSHIP_TRITON_IMAGE naming a Triton
image whose TensorRT version matches the installed one (see docs/triton/server.md). On such a
machine:

    TRTSHIP_TRITON_IMAGE=nvcr.io/nvidia/tritonserver:<tag> pytest -m triton -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from tests.conftest import TRITON_IMAGE
from tests.helpers import MLP_INPUT, MLP_PROFILE, build_config
from trtship.artifacts import ArtifactStore, ArtifactType, RunDirectory
from trtship.cli.main import app

pytestmark = [pytest.mark.gpu, pytest.mark.tensorrt, pytest.mark.docker, pytest.mark.triton]

runner = CliRunner()


def test_engine_served_by_triton_matches_pytorch_and_can_be_benchmarked(tmp_path: Path) -> None:
    data: dict[str, Any] = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE).model_dump(
        mode="json"
    )
    data["tensorrt"]["precisions"] = ["fp16"]
    data["triton"] = {
        "image": TRITON_IMAGE,
        "repository_dir": str(tmp_path / "served"),
        "container_name": "trtship-e2e-test",
    }
    data["benchmark"] = {"batch_sizes": [1, 4], "warmup_iters": 5, "iters": 20, "concurrency": [1]}
    data["artifacts"] = {"root": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")}
    config = tmp_path / "e2e.yaml"
    config.write_text(yaml.safe_dump(data))

    ran = runner.invoke(app, ["run", str(config), "--run-id", "hw"])
    assert ran.exit_code == 0, ran.output
    run = RunDirectory.open(tmp_path / "runs" / "hw")
    store = ArtifactStore(run)
    repository = store.absolute(store.require(ArtifactType.TRITON_REPOSITORY, needed_by="hw"))

    try:
        served = runner.invoke(app, ["serve", str(config), "--repository", str(repository)])
        assert served.exit_code == 0, served.output
        for protocol in ("http", "grpc"):
            validated = runner.invoke(
                app,
                [
                    *("validate", "triton", str(config), "--precision", "fp16", "--json"),
                    *("--protocol", protocol, "--repository", str(repository)),
                ],
            )
            assert validated.exit_code == 0, validated.output
            assert json.loads(validated.stdout)["passed"] is True
        benchmarked = runner.invoke(
            app, ["benchmark", "triton", str(config), "--precision", "fp16", "--json"]
        )
        assert benchmarked.exit_code == 0, benchmarked.output
    finally:
        runner.invoke(app, ["stop", str(config)])
