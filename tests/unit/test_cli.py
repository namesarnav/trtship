from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from tests.helpers import fake_environment
from trtship import __version__
from trtship.cli.main import app
from trtship.utils import env
from trtship.utils.env import EnvironmentReport

runner = CliRunner()


@pytest.fixture
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "probe_all", lambda: fake_environment(gpu_ok=False))


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("version", "doctor", "config"):
        assert name in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_version_text_and_json() -> None:
    assert __version__ in runner.invoke(app, ["version"]).output
    data = json.loads(runner.invoke(app, ["version", "--json"]).stdout)
    assert data["trtship"] == __version__
    assert {"python", "platform"} <= set(data)


def test_doctor_without_gpu_exits_zero_and_marks_blocked_areas(no_gpu: None) -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "blocked" in result.output
    assert "TensorRT build" in result.output


def test_doctor_json_includes_readiness(no_gpu: None) -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["readiness"]["ONNX export and validation (CPU)"] is True
    assert data["readiness"]["TensorRT build, calibration, benchmark"] is False
    assert {c["name"] for c in data["capabilities"]} >= {"python", "tensorrt", "docker"}


def test_doctor_require_fails_with_exit_3(no_gpu: None) -> None:
    result = runner.invoke(app, ["doctor", "--require", "tensorrt"])
    assert result.exit_code == 3
    assert "tensorrt" in result.output


def test_doctor_require_passes_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "probe_all", lambda: fake_environment(gpu_ok=True))
    assert (
        runner.invoke(app, ["doctor", "--require", "tensorrt", "--require", "nvidia_gpu"]).exit_code
        == 0
    )


def test_doctor_fails_when_core_dependency_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "probe_all", lambda: fake_environment(gpu_ok=True, core_ok=False))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 3
    assert "python" in result.output


def test_doctor_unknown_require_is_usage_error(no_gpu: None) -> None:
    result = runner.invoke(app, ["doctor", "--require", "bogus"])
    assert result.exit_code == 2
    assert "unknown capability" in result.output


def test_doctor_output_keeps_bracketed_text_literal(no_gpu: None) -> None:
    result = runner.invoke(app, ["doctor"], env={"COLUMNS": "200"})
    assert "trtship[triton]" in result.output


def test_config_validate_ok(config_dict: dict[str, Any], write_config: Callable[..., Path]) -> None:
    path = write_config(config_dict, "c.yaml")
    result = runner.invoke(app, ["config", "validate", str(path)], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "valid" in result.output
    assert "tiny" in result.output


def test_config_validate_json_ok_and_warnings(
    config_dict: dict[str, Any], write_config: Callable[..., Path]
) -> None:
    config_dict["tensorrt"]["profiles"] = []
    data = json.loads(
        runner.invoke(
            app, ["config", "validate", str(write_config(config_dict, "c.yaml")), "--json"]
        ).stdout
    )
    assert data["valid"] is True
    assert len(data["config_sha256"]) == 64
    assert data["warnings"]


def test_config_validate_invalid_exits_2_with_locations(
    config_dict: dict[str, Any], write_config: Callable[..., Path]
) -> None:
    config_dict["export"] = {"opset": 3}
    result = runner.invoke(app, ["config", "validate", str(write_config(config_dict, "c.yaml"))])
    assert result.exit_code == 2
    assert "export.opset" in result.output


def test_config_validate_invalid_json(
    config_dict: dict[str, Any], write_config: Callable[..., Path]
) -> None:
    config_dict["bogus"] = 1
    result = runner.invoke(
        app, ["config", "validate", str(write_config(config_dict, "c.yaml")), "--json"]
    )
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert payload["details"]["errors"][0]["loc"] == "bogus"


def test_config_validate_missing_file_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(app, ["config", "validate", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 2
    assert "not found" in result.output


def test_config_validate_set_override(
    config_dict: dict[str, Any], write_config: Callable[..., Path]
) -> None:
    path = str(write_config(config_dict, "c.yaml"))
    assert (
        runner.invoke(app, ["config", "validate", path, "--set", "export.opset=18"]).exit_code == 0
    )
    bad = runner.invoke(app, ["config", "validate", path, "--set", "export.opset=2"])
    assert bad.exit_code == 2


def test_invalid_log_level_is_usage_error() -> None:
    result = runner.invoke(app, ["--log-level", "LOUD", "version"])
    assert result.exit_code == 2


def test_unexpected_exception_exits_70_with_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode() -> EnvironmentReport:
        raise RuntimeError("boom")

    monkeypatch.setattr(env, "probe_all", explode)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 70
    assert "bug in trtship" in result.output
    assert "RuntimeError" in result.output


def test_filesystem_errors_exit_11_without_a_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    def unwritable() -> EnvironmentReport:
        raise PermissionError(13, "Permission denied", "/runs")

    monkeypatch.setattr(env, "probe_all", unwritable)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 11
    assert "Permission denied: /runs" in result.output
    assert "Traceback" not in result.output
    assert "bug in trtship" not in result.output


def test_connection_errors_are_not_mistaken_for_filesystem_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refused() -> EnvironmentReport:
        raise ConnectionRefusedError("nobody home")

    monkeypatch.setattr(env, "probe_all", refused)
    assert runner.invoke(app, ["doctor"]).exit_code == 70


def test_yaml_is_not_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump([1, 2]))
    assert runner.invoke(app, ["config", "validate", str(path)]).exit_code == 2
