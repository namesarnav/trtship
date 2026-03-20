from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from trtship.cli.main import app

runner = CliRunner()
Writer = Callable[..., Path]


@pytest.fixture
def config_path(config_dict: dict[str, Any], write_config: Writer) -> str:
    return str(write_config(config_dict, "c.yaml"))


def test_inspect_prints_a_human_report(config_path: str) -> None:
    result = runner.invoke(app, ["inspect", config_path], env={"COLUMNS": "120"})
    assert result.exit_code == 0, result.output
    for expected in ("Parameters", "676", "Memory estimate", "Signature", "Architecture"):
        assert expected in result.output


def test_inspect_json_is_pure_json_on_stdout(config_path: str) -> None:
    result = runner.invoke(app, ["inspect", config_path, "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["parameters"]["total"] == 676
    assert data["signature"]["outputs"][0]["shape"] == ["batch", 4]
    assert data["memory"]["activation_bytes_upper_bound"] == (4 * 32 + 4 * 32 + 4 * 4) * 4


def test_inspect_output_file_and_overwrite_protection(config_path: str, tmp_path: Path) -> None:
    target = tmp_path / "out" / "report.json"
    first = runner.invoke(app, ["inspect", config_path, "-o", str(target)])
    assert first.exit_code == 0, first.output
    assert json.loads(target.read_text())["parameters"]["total"] == 676

    again = runner.invoke(app, ["inspect", config_path, "-o", str(target)])
    assert again.exit_code == 11
    assert "refusing to overwrite" in again.output
    assert json.loads(target.read_text())["parameters"]["total"] == 676  # untouched

    forced = runner.invoke(app, ["inspect", config_path, "-o", str(target), "--force"])
    assert forced.exit_code == 0


def test_inspect_depth_option(config_path: str) -> None:
    result = runner.invoke(app, ["inspect", config_path, "--depth", "0", "--json"])
    data = json.loads(result.stdout)
    assert data["architecture"]["children"] == []
    assert data["architecture"]["elided_modules"] == 3


def test_inspect_set_override_changes_the_model(config_path: str) -> None:
    result = runner.invoke(
        app,
        ["inspect", config_path, "--json", "--set", "model.factory_kwargs.out_features=10"],
    )
    assert json.loads(result.stdout)["signature"]["outputs"][0]["shape"] == ["batch", 10]


def test_inspect_model_failure_exits_4(config_dict: dict[str, Any], write_config: Writer) -> None:
    config_dict["model"]["factory"] = "trtship_fixtures.models:raises_on_build"
    result = runner.invoke(
        app, ["inspect", str(write_config(config_dict, "c.yaml"))], env={"COLUMNS": "200"}
    )
    assert result.exit_code == 4
    assert "bad hyperparameters" in result.output


def test_inspect_invalid_config_exits_2(config_dict: dict[str, Any], write_config: Writer) -> None:
    config_dict["bogus"] = 1
    assert runner.invoke(app, ["inspect", str(write_config(config_dict, "c.yaml"))]).exit_code == 2


def test_inspect_missing_config_exits_2(tmp_path: Path) -> None:
    assert runner.invoke(app, ["inspect", str(tmp_path / "nope.yaml")]).exit_code == 2
