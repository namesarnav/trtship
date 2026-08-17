"""The shipped ResNet-50 and BERT-style examples: configs load and the CPU stages run."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from trtship.artifacts import ArtifactStore, ArtifactType, RunDirectory
from trtship.cli.main import app
from trtship.config import load_config
from trtship.reporting import summarize_run

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "configs" / "examples"
runner = CliRunner()


def overrides(tmp_path: Path) -> list[str]:
    return [
        "--set",
        f"artifacts.root={tmp_path / 'runs'}",
        "--set",
        f"artifacts.cache_dir={tmp_path / 'cache'}",
    ]


def run_until(config: Path, stage: str, tmp_path: Path) -> RunDirectory:
    result = runner.invoke(
        app, ["run", str(config), "--run-id", "x", "--until", stage, *overrides(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    return RunDirectory.open(tmp_path / "runs" / "x")


@pytest.mark.parametrize("name", ["custom_model", "resnet50", "bert_style"])
def test_every_example_config_validates(name: str) -> None:
    result = runner.invoke(app, ["config", "validate", str(EXAMPLES / f"{name}.yaml")])
    assert result.exit_code == 0, result.output


# ------------------------------------------------------------------------------ bert_style


def test_bert_style_runs_to_optimize_with_two_dynamic_axes(tmp_path: Path) -> None:
    run = run_until(EXAMPLES / "bert_style.yaml", "optimize", tmp_path)
    summary = summarize_run(run)
    assert summary.status.value == "succeeded"
    assert summary.integrity_problems == []
    assert all(v.passed for v in summary.validations)

    onnx = ArtifactStore(run).require(ArtifactType.ONNX, needed_by="test")
    assert onnx.metadata["output_names"] == ["logits"]
    axes = onnx.metadata["dynamic_axes"]
    for name in ("input_ids", "attention_mask", "token_type_ids"):
        assert axes[name] == {"0": "batch", "1": "seq"}
    assert axes["logits"] == {"0": "batch"}


def test_bert_style_profile_covers_batch_and_sequence_for_every_input() -> None:
    config = load_config(EXAMPLES / "bert_style.yaml")
    (profile,) = config.tensorrt.profiles
    assert set(profile.inputs) == {"input_ids", "attention_mask", "token_type_ids"}
    for shape_range in profile.inputs.values():
        assert len(shape_range.min) == len(shape_range.opt) == len(shape_range.max) == 2


def test_bert_style_ignores_masked_positions() -> None:
    sys.path.insert(0, str(REPO / "examples"))
    try:
        build = importlib.import_module("bert_style.model").build
    finally:
        sys.path.remove(str(REPO / "examples"))

    model = build(seed=1).eval()
    ids = torch.randint(0, 1000, (2, 12))
    types = torch.zeros_like(ids)
    mask = torch.ones_like(ids)
    mask[:, 8:] = 0
    with torch.no_grad():
        baseline = model(ids, mask, types)
        changed = ids.clone()
        changed[:, 8:] = (changed[:, 8:] + 7) % 1000  # different tokens, but all masked out
        assert torch.allclose(model(changed, mask, types), baseline, atol=1e-5)
        changed[:, :8] = (changed[:, :8] + 7) % 1000  # now change tokens that are attended to
        assert not torch.allclose(model(changed, mask, types), baseline, atol=1e-5)


# ------------------------------------------------------------------------------ resnet50


def test_resnet50_exports_with_a_dynamic_batch_axis(tmp_path: Path) -> None:
    pytest.importorskip("torchvision")
    run = run_until(EXAMPLES / "resnet50.yaml", "export", tmp_path)
    onnx = ArtifactStore(run).require(ArtifactType.ONNX, needed_by="test")
    assert onnx.metadata["dynamic_axes"]["image"] == {"0": "batch"}
    report = json.loads(
        ArtifactStore(run)
        .absolute(ArtifactStore(run).require(ArtifactType.MODEL_REPORT, needed_by="test"))
        .read_text()
    )
    assert report["parameters"]["total"] == 25_557_032


def test_resnet50_int8_calibration_is_configured_but_ships_no_data() -> None:
    config = load_config(EXAMPLES / "resnet50.yaml")
    assert config.calibration is not None
    assert config.calibration.dataset.value == "images"
    assert config.calibration.path is not None
    assert not Path(config.calibration.path).exists()  # users supply real images; nothing shipped
    assert not config.calibration.allow_synthetic
