"""End to end on the shipped example config: CLI -> pipeline -> artifacts -> reports (CPU only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.helpers import fake_environment
from trtship.artifacts import ArtifactStore, ArtifactType, RunDirectory
from trtship.cli.main import app
from trtship.reporting import summarize_run
from trtship.utils import env

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = str(REPO / "configs" / "examples" / "custom_model.yaml")
runner = CliRunner()


def overrides(tmp_path: Path) -> list[str]:
    return [
        "--set",
        f"artifacts.root={tmp_path / 'runs'}",
        "--set",
        f"artifacts.cache_dir={tmp_path / 'cache'}",
    ]


def test_example_config_runs_from_a_fresh_clone(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["run", EXAMPLE, "--run-id", "e2e", "--until", "optimize", *overrides(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    run = RunDirectory.open(tmp_path / "runs" / "e2e")
    summary = summarize_run(run)
    assert summary.status.value == "succeeded"
    assert [s.name for s in summary.stages] == ["inspect", "export", "validate", "optimize"]
    assert summary.integrity_problems == []
    assert all(v.passed for v in summary.validations)
    assert summary.model_sha256
    onnx = ArtifactStore(run).require(ArtifactType.ONNX, needed_by="e2e")
    assert onnx.metadata["dynamic_axes"]["image"] == {"0": "batch"}
    assert onnx.metadata["output_names"] == ["logits"]

    inspected = json.loads(
        ArtifactStore(run)
        .absolute(ArtifactStore(run).require(ArtifactType.MODEL_REPORT, needed_by="e2e"))
        .read_text()
    )
    assert inspected["parameters"]["total"] == 5514


def test_the_produced_onnx_validates_independently_through_the_cli(tmp_path: Path) -> None:
    assert (
        runner.invoke(
            app, ["run", EXAMPLE, "--run-id", "v", "--until", "export", *overrides(tmp_path)]
        ).exit_code
        == 0
    )
    run = RunDirectory.open(tmp_path / "runs" / "v")
    store = ArtifactStore(run)
    onnx_path = store.absolute(store.require(ArtifactType.ONNX, needed_by="e2e"))
    result = runner.invoke(app, ["validate", "onnx", EXAMPLE, str(onnx_path), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["passed"] is True
    assert [p["label"] for p in report["points"]] == ["min", "opt", "max"]


def test_a_second_run_is_served_from_the_cache_with_identical_artifacts(tmp_path: Path) -> None:
    args = ["--until", "optimize", "--json", *overrides(tmp_path)]
    first = runner.invoke(app, ["run", EXAMPLE, "--run-id", "a", *args])
    second = runner.invoke(app, ["run", EXAMPLE, "--run-id", "b", *args])
    assert first.exit_code == 0
    assert second.exit_code == 0, second.output
    assert {s["status"] for s in json.loads(second.stdout)["stages"]} == {"cached"}
    a = ArtifactStore(RunDirectory.open(tmp_path / "runs" / "a"))
    b = ArtifactStore(RunDirectory.open(tmp_path / "runs" / "b"))
    assert {r.sha256 for r in a.records()} == {r.sha256 for r in b.records()}
    assert b.verify_all() == []


def test_a_full_run_without_a_gpu_fails_fast_and_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(env, "probe_all", lambda repo_dir=None: fake_environment(gpu_ok=False))
    result = runner.invoke(app, ["run", EXAMPLE, *overrides(tmp_path)], env={"COLUMNS": "200"})
    assert result.exit_code == 3
    assert "stage 'build'" in result.output
    assert "nvidia_gpu is missing" in result.output
    assert "--until optimize" in result.output  # how to run what this machine supports
    assert not (tmp_path / "runs").exists()  # no run directory was created


def test_the_plan_includes_the_engine_stage(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", EXAMPLE, "--dry-run"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "build" in result.output
