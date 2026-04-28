from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from tests.helpers import export_to
from trtship.cli.main import app
from trtship.config import load_config
from trtship.utils.hashing import sha256_file

runner = CliRunner()
WIDE = {"COLUMNS": "200"}
BAKED_PROFILE = {"x": {"min": [1, 6], "opt": [4, 6], "max": [8, 6]}}


@pytest.fixture
def baked(tmp_path: Path) -> tuple[str, Path]:
    """A config plus an exported model that has Constant nodes for the optimizer to extract."""
    data = {
        "model": {
            "name": "m",
            "kind": "module",
            "factory": "trtship_fixtures.models:baked_batch",
            "inputs": [{"name": "x", "shape": ["batch", 6]}],
        },
        "tensorrt": {"profiles": [{"inputs": BAKED_PROFILE}]},
    }
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(data))
    export_to(load_config(config), tmp_path / "orig.onnx")
    return str(config), tmp_path / "orig.onnx"


@pytest.fixture
def mlp(tmp_path: Path, config_dict: dict[str, Any]) -> tuple[str, Path]:
    config = tmp_path / "mlp.yaml"
    config.write_text(yaml.safe_dump(config_dict))
    export_to(load_config(config), tmp_path / "mlp.onnx")
    return str(config), tmp_path / "mlp.onnx"


def test_optimize_writes_a_validated_model(mlp: tuple[str, Path], tmp_path: Path) -> None:
    config, source = mlp
    target = tmp_path / "out" / "opt.onnx"
    before = sha256_file(source)
    result = runner.invoke(app, ["optimize", config, str(source), "-o", str(target)], env=WIDE)
    assert result.exit_code == 0, result.output
    assert "optimized" in result.output
    assert "extract_constants" in result.output
    assert "PASSED" in result.output
    assert target.is_file()
    assert sha256_file(source) == before  # the original is never touched


def test_optimize_json(mlp: tuple[str, Path], tmp_path: Path) -> None:
    config, source = mlp
    result = runner.invoke(
        app, ["optimize", config, str(source), "-o", str(tmp_path / "o.onnx"), "--json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["optimize"]["source_sha256"] == sha256_file(source)
    assert data["validation"]["passed"] is True
    assert data["optimize"]["passes"][0]["name"] == "extract_constants"


def test_optimize_no_validate_skips_validation(mlp: tuple[str, Path], tmp_path: Path) -> None:
    config, source = mlp
    result = runner.invoke(
        app,
        [
            "optimize",
            config,
            str(source),
            "-o",
            str(tmp_path / "o.onnx"),
            "--no-validate",
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["validation"] is None


def test_optimize_pass_selection(baked: tuple[str, Path], tmp_path: Path) -> None:
    config, source = baked
    result = runner.invoke(
        app,
        [
            "optimize",
            config,
            str(source),
            "-o",
            str(tmp_path / "o.onnx"),
            "--no-validate",
            "--pass",
            "extract_constants",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    passes = json.loads(result.stdout)["optimize"]["passes"]
    assert passes == [{"name": "extract_constants", "changes": 3}]


def test_optimize_takes_default_passes_from_the_config(
    baked: tuple[str, Path], tmp_path: Path
) -> None:
    config, source = baked
    result = runner.invoke(
        app,
        [
            "optimize",
            config,
            str(source),
            "-o",
            str(tmp_path / "o.onnx"),
            "--no-validate",
            "--json",
            "--set",
            "optimize.passes=[eliminate_identity]",
        ],
    )
    assert [p["name"] for p in json.loads(result.stdout)["optimize"]["passes"]] == [
        "eliminate_identity"
    ]


def test_unknown_pass_is_a_usage_error(baked: tuple[str, Path], tmp_path: Path) -> None:
    config, source = baked
    result = runner.invoke(
        app, ["optimize", config, str(source), "-o", str(tmp_path / "o.onnx"), "--pass", "bogus"]
    )
    assert result.exit_code == 2
    assert not (tmp_path / "o.onnx").exists()


def test_invalid_pass_in_config_is_rejected(baked: tuple[str, Path], tmp_path: Path) -> None:
    config, source = baked
    result = runner.invoke(
        app,
        [
            "optimize",
            config,
            str(source),
            "-o",
            str(tmp_path / "o.onnx"),
            "--set",
            "optimize.passes=[nope]",
        ],
    )
    assert result.exit_code == 2


def test_optimize_refuses_to_overwrite(mlp: tuple[str, Path], tmp_path: Path) -> None:
    config, source = mlp
    target = tmp_path / "o.onnx"
    target.write_bytes(b"keep")
    result = runner.invoke(app, ["optimize", config, str(source), "-o", str(target)])
    assert result.exit_code == 11
    assert target.read_bytes() == b"keep"


def test_a_model_that_fails_validation_exits_6_after_optimizing(
    baked: tuple[str, Path], tmp_path: Path
) -> None:
    """baked_batch is wrong at other batch sizes; optimizing does not fix it, and it is reported."""
    config, source = baked
    target = tmp_path / "o.onnx"
    result = runner.invoke(app, ["optimize", config, str(source), "-o", str(target)], env=WIDE)
    assert result.exit_code == 6
    assert "FAILED" in result.output
    assert target.is_file()


def test_optimize_requires_an_output_path(mlp: tuple[str, Path]) -> None:
    config, source = mlp
    assert runner.invoke(app, ["optimize", config, str(source)]).exit_code == 2
