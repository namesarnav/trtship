"""The package stage and `trtship package`, with TensorRT replaced by the fake."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2 as pb
from typer.testing import CliRunner

from tests.fakes.fake_tensorrt import FakeCalls, FakeOptions, make_fake_trt
from tests.helpers import (
    MLP_INPUT,
    MLP_PROFILE,
    build_config,
    default_stages_without,
    export_to,
    fake_environment,
)
from trtship.artifacts import (
    ArtifactRecord,
    ArtifactStore,
    ArtifactType,
    RunDirectory,
    RunStatus,
    StageStatus,
)
from trtship.cli.main import app
from trtship.config import Precision, TrtshipConfig
from trtship.errors import ArtifactError, TritonError, ValidationFailedError
from trtship.pipeline import Pipeline
from trtship.pipeline.stages.package_stage import select_engine
from trtship.tensorrt import build as trt_build
from trtship.utils import env

MakeRun = Callable[..., RunDirectory]
runner = CliRunner()


@pytest.fixture(autouse=True)
def fake_trt(monkeypatch: pytest.MonkeyPatch) -> FakeCalls:
    real = trt_build.build_engine
    trt, calls = make_fake_trt(FakeOptions())
    monkeypatch.setattr(
        "trtship.pipeline.stages.build_stage.build_engine", lambda *a, **k: real(*a, trt=trt, **k)
    )
    monkeypatch.setattr(
        "trtship.cli.commands.build_cmd.build_engine", lambda *a, **k: real(*a, trt=trt, **k)
    )
    return calls


def config_for(tmp_path: Path, precisions: list[str], **triton: Any) -> TrtshipConfig:
    data = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE).model_dump(mode="json")
    data["tensorrt"]["precisions"] = precisions
    data["triton"] = {"repository_dir": str(tmp_path / "served"), **triton}
    data["artifacts"] = {"root": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")}
    return TrtshipConfig.model_validate(data)


def pipeline(config: TrtshipConfig, make_run: MakeRun, run_id: str = "r1") -> Pipeline:
    stages = default_stages_without("validate_engine", "benchmark")
    return Pipeline(config, make_run(run_id), stages, fake_environment(gpu_ok=True))


def parse(path: Path) -> Any:
    return text_format.Parse(path.read_text(), pb.ModelConfig())


def test_a_single_engine_is_packaged_after_the_build(tmp_path: Path, make_run: MakeRun) -> None:
    run = pipeline(config_for(tmp_path, ["fp16"]), make_run)
    result = run.execute()
    assert result.stages[-1].name == "package"
    assert result.stages[-1].status is StageStatus.SUCCEEDED
    store = ArtifactStore(run.run)
    repo = store.require(ArtifactType.TRITON_REPOSITORY, needed_by="t")
    root = store.absolute(repo)
    assert (root / "m/1/model.plan").is_file()
    config = parse(root / "m/config.pbtxt")
    assert config.max_batch_size == 8
    assert config.input[0].name == "x"
    assert repo.metadata["precision"] == "fp16"
    engine = store.require(ArtifactType.ENGINE, needed_by="t")
    assert engine.id in repo.parents
    assert (root / "m/1/model.plan").read_bytes() == store.absolute(engine).read_bytes()
    assert store.verify_all() == []
    assert any("not been validated" in w for w in result.stages[-1].warnings)


def test_several_engines_need_a_precision_choice(tmp_path: Path, make_run: MakeRun) -> None:
    run = pipeline(config_for(tmp_path, ["fp32", "fp16"]), make_run)
    with pytest.raises(TritonError, match=r"triton\.precision"):
        run.execute()
    assert ArtifactStore(run.run).latest(ArtifactType.TRITON_REPOSITORY) is None


def test_the_configured_precision_selects_the_engine(tmp_path: Path, make_run: MakeRun) -> None:
    config = config_for(tmp_path, ["fp32", "fp16"], precision="fp32")
    run = pipeline(config, make_run)
    run.execute()
    store = ArtifactStore(run.run)
    repo = store.require(ArtifactType.TRITON_REPOSITORY, needed_by="t")
    assert repo.metadata["precision"] == "fp32"
    engine = store.get(repo.metadata["engine"])
    assert engine.metadata["precision"] == "fp32"


def test_a_precision_outside_the_build_list_is_rejected_by_the_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"one of tensorrt\.precisions"):
        config_for(tmp_path, ["fp32"], precision="fp16")


def test_select_engine_reports_what_was_built() -> None:
    now = datetime.now(UTC)

    def record(precision: str) -> ArtifactRecord:
        return ArtifactRecord(
            id=f"engine-{precision}",
            type=ArtifactType.ENGINE,
            path=f"artifacts/m.{precision}.plan",
            sha256="0" * 64,
            size_bytes=1,
            created_at=now,
            stage="build",
            metadata={"precision": precision},
        )

    engines = {Precision.FP32: record("fp32")}
    with pytest.raises(TritonError, match="built: fp32"):
        select_engine(engines, Precision.FP16)
    with pytest.raises(ArtifactError, match="no engine"):
        select_engine({}, None)
    assert select_engine(engines, None).id == "engine-fp32"


def add_validation_report(store: ArtifactStore, scratch: Path, *, passed: bool) -> None:
    path = scratch / "engine_validation.json"
    path.write_text(json.dumps({"passed": passed}))
    destination = store.path_for(ArtifactType.VALIDATION_REPORT, "engine_validation.json")
    path.rename(destination)
    store.register(
        destination,
        ArtifactType.VALIDATION_REPORT,
        stage="validate_engine",
        parents=[],
        metadata={"kind": "engine", "passed": passed, "precisions": ["fp16"]},
    )


def test_an_engine_that_failed_validation_is_not_packaged(
    tmp_path: Path, make_run: MakeRun
) -> None:
    run = pipeline(config_for(tmp_path, ["fp16"]), make_run)
    run.execute(until="build")
    add_validation_report(ArtifactStore(run.run), tmp_path, passed=False)
    with pytest.raises(ValidationFailedError, match="failed numerical validation"):
        run.execute(only="package")
    assert ArtifactStore(run.run).latest(ArtifactType.TRITON_REPOSITORY) is None


def test_a_validated_engine_is_packaged_without_a_warning(
    tmp_path: Path, make_run: MakeRun
) -> None:
    run = pipeline(config_for(tmp_path, ["fp16"]), make_run)
    run.execute(until="build")
    add_validation_report(ArtifactStore(run.run), tmp_path, passed=True)
    result = run.execute(only="package")
    assert result.status is RunStatus.SUCCEEDED
    assert result.stages[-1].warnings == []


# ---------------------------------------------------------------- CLI


def write_config(tmp_path: Path, **triton: Any) -> Path:
    config = config_for(tmp_path, ["fp16"], **triton)
    data = config.model_dump(mode="json")
    data["model"]["factory"] = "trtship_fixtures.models:tiny_mlp"
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_build_then_package_produces_a_repository_without_a_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(env, "require_tensorrt", lambda purpose: None)  # the fake builds it
    config_path = write_config(tmp_path)
    config = config_for(tmp_path, ["fp16"])
    export_to(config, tmp_path / "m.onnx")
    built = runner.invoke(
        app, ["build", str(config_path), str(tmp_path / "m.onnx"), "-o", str(tmp_path / "engines")]
    )
    assert built.exit_code == 0, built.output
    sidecar = tmp_path / "engines/m.fp16.plan.json"
    assert json.loads(sidecar.read_text())["engine"]["tensors"][0]["name"] == "x"

    packaged = runner.invoke(
        app, ["package", str(config_path), str(tmp_path / "engines/m.fp16.plan"), "--json"]
    )
    assert packaged.exit_code == 0, packaged.output
    payload = json.loads(packaged.output)
    assert payload["model_name"] == "m"
    assert (tmp_path / "served/m/1/model.plan").is_file()
    assert parse(tmp_path / "served/m/config.pbtxt").max_batch_size == 8


def test_package_refuses_an_existing_directory_and_missing_metadata(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    plan = tmp_path / "x.plan"
    plan.write_bytes(b"p")
    missing = runner.invoke(app, ["package", str(config_path), str(plan)])
    assert missing.exit_code == TritonError.exit_code
    assert "--engine-info" in missing.output

    (tmp_path / "served").mkdir()
    (tmp_path / "x.plan.json").write_text("{}")
    invalid = runner.invoke(app, ["package", str(config_path), str(plan)])
    assert invalid.exit_code == TritonError.exit_code
