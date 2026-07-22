"""The CPU stages (inspect -> export -> validate -> optimize) run for real on fixture models."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from trtship.artifacts import ArtifactStore, ArtifactType, RunDirectory, RunStatus, StageStatus
from trtship.config import TrtshipConfig
from trtship.errors import ArtifactError, ValidationFailedError
from trtship.onnx import validate_onnx as real_validate_onnx
from trtship.pipeline import Pipeline, Stage
from trtship.pipeline.stages import default_stages
from trtship.utils.env import EnvironmentReport

MakeRun = Callable[..., RunDirectory]
BAKED_PROFILE = {"x": {"min": [1, 6], "opt": [4, 6], "max": [8, 6]}}


def config_for(tmp_path: Path, config_dict: dict[str, Any], **overrides: Any) -> TrtshipConfig:
    data = {
        **config_dict,
        "artifacts": {"root": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")},
        **overrides,
    }
    return TrtshipConfig.model_validate(data)


def baked_config(tmp_path: Path) -> TrtshipConfig:
    return TrtshipConfig.model_validate(
        {
            "model": {
                "name": "m",
                "kind": "module",
                "factory": "trtship_fixtures.models:baked_batch",
                "inputs": [{"name": "x", "shape": ["batch", 6]}],
            },
            "tensorrt": {"profiles": [{"inputs": BAKED_PROFILE}]},
            "artifacts": {"root": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")},
        }
    )


def cpu_stages() -> list[Stage]:
    """The default stages that need no GPU (the engine stages are exercised elsewhere)."""
    return [s for s in default_stages() if not s.requires_capabilities and s.name != "package"]


def pipeline(
    config: TrtshipConfig, make_run: MakeRun, environment: EnvironmentReport, run_id: str = "r1"
) -> Pipeline:
    return Pipeline(config, make_run(run_id), cpu_stages(), environment)


@pytest.fixture
def full_run(
    tmp_path: Path,
    config_dict: dict[str, Any],
    make_run: MakeRun,
    environment_report: EnvironmentReport,
) -> Pipeline:
    return pipeline(config_for(tmp_path, config_dict), make_run, environment_report)


def test_the_full_cpu_pipeline(full_run: Pipeline) -> None:
    result = full_run.execute()
    assert result.status is RunStatus.SUCCEEDED
    assert [(o.name, o.status) for o in result.stages] == [
        ("inspect", StageStatus.SUCCEEDED),
        ("export", StageStatus.SUCCEEDED),
        ("validate", StageStatus.SUCCEEDED),
        ("optimize", StageStatus.SUCCEEDED),
    ]
    store = ArtifactStore(full_run.run)
    onnx = store.require(ArtifactType.ONNX, needed_by="test")
    optimized = store.require(ArtifactType.ONNX_OPTIMIZED, needed_by="test")
    assert onnx.path == "artifacts/tiny.onnx"
    assert optimized.path == "artifacts/tiny.optimized.onnx"
    assert optimized.parents == [onnx.id]
    assert optimized.metadata["validation_passed"] is True
    assert onnx.metadata["exporter"] == "torchscript"
    assert onnx.metadata["opset"] == 17
    assert store.verify_all() == []

    reports = store.records(ArtifactType.VALIDATION_REPORT)
    assert sorted(r.metadata["kind"] for r in reports) == ["onnx", "onnx_optimized"]
    assert all(r.metadata["passed"] is True for r in reports)
    model_report = json.loads(
        store.absolute(store.require(ArtifactType.MODEL_REPORT, needed_by="t")).read_text()
    )
    assert model_report["parameters"]["total"] == 676

    manifest = full_run.run.read_manifest()
    assert manifest.status is RunStatus.SUCCEEDED
    assert manifest.model_sha256 == model_report["weights_sha256"]
    assert manifest.stages["validate"].metrics["max_abs_error"] < 1e-5
    assert not (full_run.run.path / ".work").exists()


def test_a_second_execution_skips_everything(full_run: Pipeline) -> None:
    first = full_run.execute()
    second = full_run.execute()
    assert all(o.status is StageStatus.SKIPPED for o in second.stages)
    assert [o.artifacts for o in second.stages] == [o.artifacts for o in first.stages]
    assert len(list((full_run.run.path / "artifacts").iterdir())) == 2  # nothing was rewritten


def test_a_new_run_reuses_the_shared_cache(
    tmp_path: Path,
    config_dict: dict[str, Any],
    make_run: MakeRun,
    environment_report: EnvironmentReport,
) -> None:
    config = config_for(tmp_path, config_dict)
    first = pipeline(config, make_run, environment_report, "one")
    first.execute()
    second = pipeline(config, make_run, environment_report, "two")
    result = second.execute()
    assert all(o.status is StageStatus.CACHED for o in result.stages), result.stages
    a = ArtifactStore(first.run).require(ArtifactType.ONNX_OPTIMIZED, needed_by="t")
    b = ArtifactStore(second.run).require(ArtifactType.ONNX_OPTIMIZED, needed_by="t")
    assert a.sha256 == b.sha256
    assert b.metadata["reused_from_cache"]
    assert b.parents == [ArtifactStore(second.run).require(ArtifactType.ONNX, needed_by="t").id]
    assert ArtifactStore(second.run).verify_all() == []


def test_changing_the_export_config_invalidates_the_downstream_cache(
    tmp_path: Path,
    config_dict: dict[str, Any],
    make_run: MakeRun,
    environment_report: EnvironmentReport,
) -> None:
    pipeline(config_for(tmp_path, config_dict), make_run, environment_report, "one").execute()
    other = config_for(tmp_path, config_dict, export={"opset": 18})
    result = pipeline(other, make_run, environment_report, "two").execute()
    statuses = {o.name: o.status for o in result.stages}
    assert statuses["inspect"] is StageStatus.CACHED  # the model report does not depend on export
    assert statuses["export"] is StageStatus.SUCCEEDED
    assert statuses["validate"] is StageStatus.SUCCEEDED
    assert statuses["optimize"] is StageStatus.SUCCEEDED


def test_disabling_optimization_skips_the_stage(
    tmp_path: Path,
    config_dict: dict[str, Any],
    make_run: MakeRun,
    environment_report: EnvironmentReport,
) -> None:
    config = config_for(tmp_path, config_dict, optimize={"enabled": False})
    run = pipeline(config, make_run, environment_report)
    result = run.execute()
    optimize = next(o for o in result.stages if o.name == "optimize")
    assert (optimize.status, optimize.reason) == (StageStatus.SKIPPED, "optimize.enabled is false")
    assert ArtifactStore(run.run).latest(ArtifactType.ONNX_OPTIMIZED) is None


def test_partial_runs_need_the_artifacts_of_earlier_stages(full_run: Pipeline) -> None:
    with pytest.raises(ArtifactError, match="needs a onnx artifact"):
        full_run.execute(only="validate")
    assert full_run.run.read_manifest().stages["validate"].status is StageStatus.FAILED
    full_run.execute(until="export")
    result = full_run.execute(from_stage="validate")
    assert [o.name for o in result.stages] == ["validate", "optimize"]
    assert [o.status for o in result.stages] == [StageStatus.SUCCEEDED] * 2


def test_a_wrong_model_fails_validation_and_leaves_evidence(
    tmp_path: Path, make_run: MakeRun, environment_report: EnvironmentReport
) -> None:
    run = pipeline(baked_config(tmp_path), make_run, environment_report)
    with pytest.raises(ValidationFailedError, match="ONNX validation failed") as info:
        run.execute()
    assert info.value.exit_code == 6

    manifest = run.run.read_manifest()
    assert manifest.status is RunStatus.FAILED
    assert manifest.stages["inspect"].status is StageStatus.SUCCEEDED
    assert manifest.stages["export"].status is StageStatus.SUCCEEDED
    failed = manifest.stages["validate"]
    assert failed.status is StageStatus.FAILED
    assert failed.error is not None
    assert failed.error["exit_code"] == 6
    assert failed.error["details"]["failures"]
    assert "optimize" not in manifest.stages  # the run stopped

    store = ArtifactStore(run.run)
    evidence = store.require(ArtifactType.VALIDATION_REPORT, needed_by="t")
    assert evidence.metadata["passed"] is False
    assert json.loads(store.absolute(evidence).read_text())["passed"] is False
    assert not (run.run.path / ".work").exists()


def test_an_optimized_model_that_fails_validation_is_never_registered(
    tmp_path: Path, make_run: MakeRun, environment_report: EnvironmentReport
) -> None:
    run = pipeline(baked_config(tmp_path), make_run, environment_report)
    run.execute(until="export")
    with pytest.raises(ValidationFailedError):
        run.execute(only="optimize")
    store = ArtifactStore(run.run)
    assert store.latest(ArtifactType.ONNX_OPTIMIZED) is None
    report = store.require(ArtifactType.VALIDATION_REPORT, needed_by="t")
    assert report.metadata == {
        "subject": "optimized onnx",
        "kind": "onnx_optimized",
        "passed": False,
    }
    assert [p.name for p in (run.run.path / "artifacts").iterdir()] == ["m.onnx"]


def test_resuming_after_a_transient_failure(
    full_run: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient: out of memory")
        return real_validate_onnx(*args, **kwargs)

    monkeypatch.setattr("trtship.pipeline.stages.validate_stage.validate_onnx", flaky)
    with pytest.raises(RuntimeError, match="out of memory"):
        full_run.execute()
    assert full_run.run.read_manifest().status is RunStatus.FAILED

    result = full_run.execute()
    statuses = {o.name: o.status for o in result.stages}
    assert statuses["inspect"] is StageStatus.SKIPPED  # already done, verified, reused
    assert statuses["export"] is StageStatus.SKIPPED
    assert statuses["validate"] is StageStatus.SUCCEEDED
    assert statuses["optimize"] is StageStatus.SUCCEEDED
    assert full_run.run.read_manifest().status is RunStatus.SUCCEEDED
    assert calls["n"] >= 2
