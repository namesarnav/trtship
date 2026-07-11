"""The build stage inside the real pipeline, with TensorRT replaced by tests/fakes/fake_tensorrt.py.

This proves the wiring (stage ordering, provenance, caching, failure recording). It does not prove
real TensorRT behavior; see tests/gpu for that.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from tests.fakes.fake_tensorrt import FakeCalls, FakeOptions, ParserError, make_fake_trt
from tests.helpers import (
    MLP_INPUT,
    MLP_PROFILE,
    build_config,
    default_stages_without,
    export_to,
    fake_environment,
)
from trtship.artifacts import ArtifactStore, ArtifactType, RunDirectory, RunStatus, StageStatus
from trtship.cli.main import app
from trtship.config import TrtshipConfig
from trtship.errors import EngineBuildError
from trtship.pipeline import Pipeline, Stage
from trtship.tensorrt import build as trt_build
from trtship.utils import env

MakeRun = Callable[..., RunDirectory]
runner = CliRunner()


@pytest.fixture
def fake_trt(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeOptions | None], FakeCalls]:
    """Route build_engine to the fake TensorRT. Returns a function to (re)configure the fake."""
    real = trt_build.build_engine

    def install(options: FakeOptions | None = None) -> FakeCalls:
        trt, calls = make_fake_trt(options)
        monkeypatch.setattr(
            "trtship.pipeline.stages.build_stage.build_engine",
            lambda *a, **k: real(*a, trt=trt, **k),
        )
        monkeypatch.setattr(
            "trtship.cli.commands.build_cmd.build_engine", lambda *a, **k: real(*a, trt=trt, **k)
        )
        return calls

    return install


def config_for(tmp_path: Path, **tensorrt: Any) -> TrtshipConfig:
    base = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)
    data = base.model_dump(mode="json")
    data["tensorrt"].update({"precisions": ["fp32", "fp16"], **tensorrt})
    data["artifacts"] = {"root": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")}
    return TrtshipConfig.model_validate(data)


def build_stages() -> list[Stage]:
    """Every default stage except the ones that need an engine executor (tested elsewhere)."""
    return default_stages_without("validate_engine", "benchmark")


def pipeline(config: TrtshipConfig, make_run: MakeRun, run_id: str = "r1") -> Pipeline:
    return Pipeline(config, make_run(run_id), build_stages(), fake_environment(gpu_ok=True))


def test_the_engine_is_built_after_optimization_from_the_optimized_model(
    tmp_path: Path, make_run: MakeRun, fake_trt: Callable[..., FakeCalls]
) -> None:
    calls = fake_trt()
    run = pipeline(config_for(tmp_path), make_run)
    result = run.execute()
    assert [o.name for o in result.stages] == [
        "inspect",
        "export",
        "validate",
        "optimize",
        "calibrate",
        "build",
    ]
    statuses = {o.name: o.status for o in result.stages}
    assert statuses.pop("calibrate") is StageStatus.SKIPPED  # no int8 configured
    assert all(status is StageStatus.SUCCEEDED for status in statuses.values())

    store = ArtifactStore(run.run)
    engines = store.records(ArtifactType.ENGINE)
    assert sorted(e.metadata["precision"] for e in engines) == ["fp16", "fp32"]
    assert sorted(e.path for e in engines) == [
        "artifacts/m.fp16.plan",
        "artifacts/m.fp32.plan",
    ]
    optimized = store.require(ArtifactType.ONNX_OPTIMIZED, needed_by="t")
    assert {e.metadata["source_onnx"] for e in engines} == {
        optimized.id
    }  # built from the optimized model
    assert optimized.id in {p for e in engines for p in e.parents}
    for engine in engines:
        build = engine.metadata["build"]
        assert build["tensorrt_version"] == "10.3.0.26"
        assert build["engine"]["tensors"][0]["name"] == "x"
        assert build["profiles"][0]["x"]["max"] == [8, 16]
    assert len(calls.configs) == 2  # one builder config per precision
    assert result.stages[-1].metrics["fp16"]["size_bytes"] > 0
    assert store.verify_all() == []


def test_without_optimization_the_original_onnx_is_used(
    tmp_path: Path, make_run: MakeRun, fake_trt: Callable[..., FakeCalls]
) -> None:
    fake_trt()
    config = config_for(tmp_path).model_copy(
        update={"optimize": config_for(tmp_path).optimize.model_copy(update={"enabled": False})}
    )
    run = pipeline(config, make_run)
    run.execute()
    store = ArtifactStore(run.run)
    onnx = store.require(ArtifactType.ONNX, needed_by="t")
    assert {e.metadata["source_onnx"] for e in store.records(ArtifactType.ENGINE)} == {onnx.id}


def test_rebuilds_are_skipped_and_new_runs_reuse_the_cache(
    tmp_path: Path, make_run: MakeRun, fake_trt: Callable[..., FakeCalls]
) -> None:
    calls = fake_trt()
    config = config_for(tmp_path)
    first = pipeline(config, make_run, "one")
    first.execute()
    builds = calls.builds
    again = first.execute()
    assert again.stages[-1].status is StageStatus.SKIPPED
    assert calls.builds == builds  # TensorRT was not asked to build again

    second = pipeline(config, make_run, "two")
    result = second.execute()
    assert result.stages[-1].status is StageStatus.CACHED
    assert calls.builds == builds
    a = ArtifactStore(first.run).records(ArtifactType.ENGINE)
    b = ArtifactStore(second.run).records(ArtifactType.ENGINE)
    assert {r.sha256 for r in a} == {r.sha256 for r in b}


def test_the_engine_cache_key_depends_on_the_gpu_and_tensorrt_settings(
    tmp_path: Path, make_run: MakeRun, fake_trt: Callable[..., FakeCalls]
) -> None:
    fake_trt()
    pipeline(config_for(tmp_path), make_run, "one").execute()
    changed = pipeline(config_for(tmp_path, workspace_mb=1024), make_run, "two").execute()
    statuses = {o.name: o.status for o in changed.stages}
    assert statuses["optimize"] is StageStatus.CACHED  # upstream results are still valid
    assert statuses["build"] is StageStatus.SUCCEEDED  # the workspace changed, so rebuild


def test_a_build_failure_is_recorded_with_the_builder_log(
    tmp_path: Path, make_run: MakeRun, fake_trt: Callable[..., FakeCalls]
) -> None:
    fake_trt(FakeOptions(build_fails=True))
    run = pipeline(config_for(tmp_path), make_run)
    with pytest.raises(EngineBuildError) as info:
        run.execute()
    assert info.value.exit_code == 7
    manifest = run.run.read_manifest()
    assert manifest.status is RunStatus.FAILED
    failed = manifest.stages["build"]
    assert failed.error is not None
    assert failed.error["details"]["precision"] == "fp32"
    assert any("conv1" in line for line in failed.error["details"]["builder_log"])
    assert ArtifactStore(run.run).records(ArtifactType.ENGINE) == []
    assert not (run.run.path / ".work").exists()
    assert manifest.stages["optimize"].status is StageStatus.SUCCEEDED  # earlier work is kept


def test_unsupported_operators_reach_the_manifest(
    tmp_path: Path, make_run: MakeRun, fake_trt: Callable[..., FakeCalls]
) -> None:
    fake_trt(FakeOptions(parse_errors=[ParserError("No importer registered for op: FFT.", 4)]))
    run = pipeline(config_for(tmp_path), make_run)
    with pytest.raises(EngineBuildError):
        run.execute()
    error = run.run.read_manifest().stages["build"].error
    assert error is not None
    assert error["details"]["unsupported_operators"] == ["FFT"]


def test_resuming_after_a_transient_build_failure_only_repeats_the_build(
    tmp_path: Path, make_run: MakeRun, fake_trt: Callable[..., FakeCalls]
) -> None:
    fake_trt(FakeOptions(build_fails=True))
    run = pipeline(config_for(tmp_path), make_run)
    with pytest.raises(EngineBuildError):
        run.execute()
    fake_trt(FakeOptions())  # the GPU is healthy again
    result = run.execute()
    assert {o.name: o.status for o in result.stages} == {
        "inspect": StageStatus.SKIPPED,
        "export": StageStatus.SKIPPED,
        "validate": StageStatus.SKIPPED,
        "optimize": StageStatus.SKIPPED,
        "calibrate": StageStatus.SKIPPED,
        "build": StageStatus.SUCCEEDED,
    }


# --------------------------------------------------------------------------- the build command


@pytest.fixture
def exported_onnx(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[str, Path]:
    config = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)
    path = tmp_path / "m.onnx"
    export_to(config, path)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    return str(cfg_path), path


def test_build_command_without_a_gpu_exits_3(
    exported_onnx: tuple[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        env, "probe_nvidia_gpu", lambda: ([], env._missing(env.NVIDIA_GPU, "no driver"))
    )
    config, onnx_path = exported_onnx
    result = runner.invoke(
        app, ["build", config, str(onnx_path), "-o", str(tmp_path / "out")], env={"COLUMNS": "200"}
    )
    assert result.exit_code == 3
    assert "building TensorRT engines" in result.output
    assert not (tmp_path / "out").exists()


def test_build_command_builds_the_requested_precisions(
    exported_onnx: tuple[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_trt: Callable[..., FakeCalls],
) -> None:
    fake_trt()
    monkeypatch.setattr(env, "require_tensorrt", lambda purpose: None)
    config, onnx_path = exported_onnx
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["build", config, str(onnx_path), "-o", str(out), "--precision", "fp16", "--json"],
    )
    assert result.exit_code == 0, result.output
    (built,) = json.loads(result.stdout)
    assert built["precision"] == "fp16"
    assert (out / "m.fp16.plan").is_file()
    assert not (out / "m.fp32.plan").exists()

    again = runner.invoke(
        app, ["build", config, str(onnx_path), "-o", str(out), "--precision", "fp16"]
    )
    assert again.exit_code == 7  # never overwrites an existing engine
    assert "refusing to overwrite" in again.output
