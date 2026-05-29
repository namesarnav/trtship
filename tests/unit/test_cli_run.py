from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from trtship.cli.main import app
from trtship.config import TrtshipConfig, config_schema_json, load_config

runner = CliRunner()
Writer = Callable[..., Path]
WIDE = {"COLUMNS": "200"}
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg(config_dict: dict[str, Any], write_config: Writer, tmp_path: Path) -> str:
    """A config for the tiny MLP with run/cache locations inside tmp_path."""
    config_dict["artifacts"] = {
        "root": str(tmp_path / "runs"),
        "cache_dir": str(tmp_path / "cache"),
    }
    return str(write_config(config_dict, "c.yaml"))


def baked(tmp_path: Path) -> str:
    data = {
        "model": {
            "name": "m",
            "kind": "module",
            "factory": "trtship_fixtures.models:baked_batch",
            "inputs": [{"name": "x", "shape": ["batch", 6]}],
        },
        "tensorrt": {
            "profiles": [{"inputs": {"x": {"min": [1, 6], "opt": [4, 6], "max": [8, 6]}}}]
        },
        "artifacts": {"root": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")},
    }
    path = tmp_path / "baked.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


# --------------------------------------------------------------------------- run


def test_run_executes_the_pipeline_and_writes_the_run_directory(cfg: str, tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", cfg, "--run-id", "r1"], env=WIDE)
    assert result.exit_code == 0, result.output
    for expected in ("run r1", "inspect", "export", "validate", "optimize", "succeeded"):
        assert expected in result.output
    run = tmp_path / "runs" / "r1"
    for name in ("config.yaml", "manifest.json", "environment.json"):
        assert (run / name).is_file()
    assert (run / "artifacts" / "tiny.onnx").is_file()
    assert (run / "artifacts" / "tiny.optimized.onnx").is_file()
    log_lines = [
        json.loads(line) for line in (run / "logs" / "trtship.jsonl").read_text().splitlines()
    ]
    assert {"inspect", "export", "validate", "optimize"} <= {r.get("stage") for r in log_lines}
    assert all(r["run_id"] == "r1" for r in log_lines if "stage" in r)


def test_run_json_output_is_pure_json(cfg: str) -> None:
    result = runner.invoke(app, ["run", cfg, "--run-id", "j1", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["status"] == "succeeded"
    assert [s["name"] for s in data["stages"]] == ["inspect", "export", "validate", "optimize"]


def test_dry_run_creates_nothing(cfg: str, tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", cfg, "--dry-run", "--until", "validate"], env=WIDE)
    assert result.exit_code == 0, result.output
    assert "validate" in result.output
    assert "optimize" not in result.output
    assert not (tmp_path / "runs").exists()


def test_dry_run_shows_disabled_stages(cfg: str) -> None:
    result = runner.invoke(
        app, ["run", cfg, "--dry-run", "--set", "optimize.enabled=false"], env=WIDE
    )
    assert "optimize.enabled is false" in result.output


def test_partial_selection_needs_an_existing_run(cfg: str, tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", cfg, "--from", "validate"], env=WIDE)
    assert result.exit_code == 2
    assert "needs artifacts from earlier stages" in result.output
    assert not (tmp_path / "runs").exists()
    assert runner.invoke(app, ["run", cfg, "--only", "inspect", "--run-id", "o1"]).exit_code == 0


def test_unknown_stage_is_a_usage_error(cfg: str) -> None:
    result = runner.invoke(app, ["run", cfg, "--until", "nope"], env=WIDE)
    assert result.exit_code == 2
    assert "unknown stage 'nope'" in result.output


def test_resuming_a_run_skips_finished_stages(cfg: str, tmp_path: Path) -> None:
    assert runner.invoke(app, ["run", cfg, "--run-id", "r1"]).exit_code == 0
    again = runner.invoke(app, ["run", cfg, "--run", str(tmp_path / "runs" / "r1"), "--json"])
    assert again.exit_code == 0, again.output
    data = json.loads(again.stdout)
    assert {s["status"] for s in data["stages"]} == {"skipped"}
    assert {s["reason"] for s in data["stages"]} == {"up to date"}


def test_resuming_with_a_different_config_is_refused(cfg: str, tmp_path: Path) -> None:
    assert runner.invoke(app, ["run", cfg, "--run-id", "r1"]).exit_code == 0
    result = runner.invoke(
        app,
        ["run", cfg, "--run", str(tmp_path / "runs" / "r1"), "--set", "export.opset=18"],
        env=WIDE,
    )
    assert result.exit_code == 2
    assert "differs from run r1's snapshot in: export" in result.output


def test_only_the_artifacts_section_may_differ_on_resume(cfg: str, tmp_path: Path) -> None:
    assert runner.invoke(app, ["run", cfg, "--run-id", "r1"]).exit_code == 0
    moved = ["--set", f"artifacts.cache_dir={tmp_path / 'other-cache'}"]
    result = runner.invoke(app, ["run", cfg, "--run", str(tmp_path / "runs" / "r1"), *moved])
    assert result.exit_code == 0, result.output


def test_a_failing_model_exits_6_and_says_how_to_resume(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", baked(tmp_path), "--run-id", "bad"], env=WIDE)
    assert result.exit_code == 6
    assert "ONNX validation failed" in result.output
    assert "did not complete" in result.output
    assert "resume with: trtship run" in result.output
    manifest = json.loads((tmp_path / "runs" / "bad" / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["stages"]["validate"]["status"] == "failed"
    log = (tmp_path / "runs" / "bad" / "logs" / "trtship.jsonl").read_text()
    assert "stage validate failed" in log


def test_reusing_a_run_id_is_refused(cfg: str) -> None:
    assert runner.invoke(app, ["run", cfg, "--run-id", "dup"]).exit_code == 0
    result = runner.invoke(app, ["run", cfg, "--run-id", "dup"], env=WIDE)
    assert result.exit_code == 11
    assert "already exists" in result.output


def test_run_id_cannot_escape_the_run_root(cfg: str, tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", cfg, "--run-id", "../escape"])
    assert result.exit_code == 2
    assert not (tmp_path / "escape").exists()


def test_resuming_something_that_is_not_a_run(cfg: str, tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", cfg, "--run", str(tmp_path)])
    assert result.exit_code == 11


# --------------------------------------------------------------------------- report


def test_report_summarizes_the_latest_run(cfg: str, tmp_path: Path) -> None:
    runner.invoke(app, ["run", cfg, "--run-id", "first"])
    runner.invoke(app, ["run", cfg, "--run-id", "second"])
    result = runner.invoke(
        app, ["report", "--root", str(tmp_path / "runs")], env={"COLUMNS": "200"}
    )
    assert result.exit_code == 0, result.output
    assert "Run second" in result.output
    for expected in ("Stages", "Artifacts", "validation onnx: passed", "onnx_optimized"):
        assert expected in result.output


def test_report_json_and_execution_order(cfg: str, tmp_path: Path) -> None:
    runner.invoke(app, ["run", cfg, "--run-id", "r1"])
    result = runner.invoke(app, ["report", str(tmp_path / "runs" / "r1"), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert [s["name"] for s in data["stages"]] == ["inspect", "export", "validate", "optimize"]
    assert data["status"] == "succeeded"
    assert data["integrity_problems"] == []
    assert {v["kind"] for v in data["validations"]} == {"onnx", "onnx_optimized"}
    assert data["model_sha256"]
    assert data["versions"]["torch"]


def test_report_flags_tampered_artifacts(cfg: str, tmp_path: Path) -> None:
    runner.invoke(app, ["run", cfg, "--run-id", "r1"])
    (tmp_path / "runs" / "r1" / "artifacts" / "tiny.onnx").write_bytes(b"tampered")
    data = json.loads(
        runner.invoke(app, ["report", str(tmp_path / "runs" / "r1"), "--json"]).stdout
    )
    assert any("tiny.onnx" in p for p in data["integrity_problems"])


def test_report_without_runs_exits_11(tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", "--root", str(tmp_path / "none")])
    assert result.exit_code == 11
    assert "no runs found" in result.output


# --------------------------------------------------------------------------- init / schema


def test_init_writes_a_valid_starter_config(tmp_path: Path) -> None:
    target = tmp_path / "trtship.yaml"
    result = runner.invoke(
        app, ["init", str(target), "--name", "resnet50", "--factory", "m.net:build"]
    )
    assert result.exit_code == 0, result.output
    config = load_config(target)
    assert isinstance(config, TrtshipConfig)
    assert config.model.name == "resnet50"
    assert config.model.factory == "m.net:build"
    assert config.tensorrt.profiles  # the template ships a matching profile
    assert config.warnings() == []
    assert runner.invoke(app, ["config", "validate", str(target)]).exit_code == 0


def test_init_refuses_to_overwrite_and_validates_its_arguments(tmp_path: Path) -> None:
    target = tmp_path / "t.yaml"
    target.write_text("keep")
    assert runner.invoke(app, ["init", str(target)]).exit_code == 11
    assert target.read_text() == "keep"
    assert runner.invoke(app, ["init", str(target), "--force"]).exit_code == 0
    assert target.read_text() != "keep"
    assert (
        runner.invoke(app, ["init", str(tmp_path / "a.yaml"), "--name", "bad name!"]).exit_code == 2
    )
    assert (
        runner.invoke(app, ["init", str(tmp_path / "b.yaml"), "--factory", "nocolon"]).exit_code
        == 2
    )


def test_config_schema_command_and_drift() -> None:
    result = runner.invoke(app, ["config", "schema"])
    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert schema["title"] == "trtship configuration"
    assert {"model", "tensorrt", "export"} <= set(schema["properties"])
    committed = (REPO / "configs" / "schemas" / "trtship.schema.json").read_text()
    assert committed == config_schema_json(), "run `make schema` and commit the result"


def test_config_schema_can_be_written_to_a_file(tmp_path: Path) -> None:
    target = tmp_path / "out" / "schema.json"
    assert runner.invoke(app, ["config", "schema", "-o", str(target)]).exit_code == 0
    assert json.loads(target.read_text())["title"] == "trtship configuration"
