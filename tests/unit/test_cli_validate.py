from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from tests.helpers import MLP_INPUT, MLP_PROFILE, build_config, export_to
from trtship.cli.main import app
from trtship.config import load_config

runner = CliRunner()
Writer = Callable[..., Path]
WIDE = {"COLUMNS": "200"}


@pytest.fixture
def good(config_dict: dict[str, Any], write_config: Writer, tmp_path: Path) -> tuple[str, Path]:
    """A config file and an ONNX model exported from it."""
    path = write_config(config_dict, "c.yaml")
    export_to(build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE), tmp_path / "m.onnx")
    return str(path), tmp_path / "m.onnx"


@pytest.fixture
def baked(tmp_path: Path) -> tuple[str, Path]:
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
    }
    config = tmp_path / "baked.yaml"
    config.write_text(yaml.safe_dump(data))
    export_to(load_config(config), tmp_path / "baked.onnx")
    return str(config), tmp_path / "baked.onnx"


def test_validate_onnx_passes(good: tuple[str, Path]) -> None:
    config, onnx_path = good
    result = runner.invoke(app, ["validate", "onnx", config, str(onnx_path)], env=WIDE)
    assert result.exit_code == 0, result.output
    assert "PASSED" in result.output
    assert "PyTorch vs ONNX Runtime" in result.output


def test_validate_onnx_json_is_pure_json(good: tuple[str, Path]) -> None:
    config, onnx_path = good
    result = runner.invoke(app, ["validate", "onnx", config, str(onnx_path), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["passed"] is True
    assert [p["label"] for p in data["points"]] == ["min", "opt", "max"]
    assert data["providers"] == ["CPUExecutionProvider"]


def test_validate_onnx_failure_exits_6_and_still_writes_the_report(
    baked: tuple[str, Path], tmp_path: Path
) -> None:
    config, onnx_path = baked
    report_path = tmp_path / "reports" / "onnx.json"
    result = runner.invoke(
        app, ["validate", "onnx", config, str(onnx_path), "-o", str(report_path)], env=WIDE
    )
    assert result.exit_code == 6
    assert "FAILED" in result.output
    assert "ONNX validation failed" in result.output
    saved = json.loads(report_path.read_text())
    assert saved["passed"] is False
    assert saved["failures"]


def test_validate_onnx_failure_json(baked: tuple[str, Path]) -> None:
    config, onnx_path = baked
    result = runner.invoke(app, ["validate", "onnx", config, str(onnx_path), "--json"])
    assert result.exit_code == 6
    assert json.loads(result.stdout)["passed"] is False


def test_validate_onnx_refuses_to_overwrite_the_report(
    good: tuple[str, Path], tmp_path: Path
) -> None:
    config, onnx_path = good
    report_path = tmp_path / "r.json"
    args = ["validate", "onnx", config, str(onnx_path), "-o", str(report_path)]
    assert runner.invoke(app, args).exit_code == 0
    again = runner.invoke(app, args)
    assert again.exit_code == 11
    assert runner.invoke(app, [*args, "--force"]).exit_code == 0


def test_validate_onnx_tolerance_override(good: tuple[str, Path]) -> None:
    config, onnx_path = good
    result = runner.invoke(
        app,
        [
            "validate",
            "onnx",
            config,
            str(onnx_path),
            "--json",
            "--set",
            "validation.onnx_tolerance.atol=0",
            "--set",
            "validation.onnx_tolerance.rtol=0",
            "--set",
            "validation.onnx_tolerance.cosine_min=1",
        ],
    )
    assert result.exit_code == 6
    assert json.loads(result.stdout)["tolerance"]["atol"] == 0


def test_validate_onnx_missing_and_corrupt_models_exit_6(
    good: tuple[str, Path], tmp_path: Path
) -> None:
    config, _ = good
    missing = runner.invoke(app, ["validate", "onnx", config, str(tmp_path / "nope.onnx")])
    assert missing.exit_code == 6
    junk = tmp_path / "junk.onnx"
    junk.write_bytes(b"junk")
    assert runner.invoke(app, ["validate", "onnx", config, str(junk)]).exit_code == 6


def test_validate_onnx_bad_config_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", "onnx", str(tmp_path / "no.yaml"), "m.onnx"])
    assert result.exit_code == 2


def test_validate_group_lists_onnx() -> None:
    result = runner.invoke(app, ["validate", "--help"])
    assert result.exit_code == 0
    assert "onnx" in result.output
