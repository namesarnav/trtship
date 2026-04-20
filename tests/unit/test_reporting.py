from __future__ import annotations

import io
from pathlib import Path

import torch
from rich.console import Console
from torch import nn

from tests.helpers import MLP_INPUT, MLP_PROFILE, build_config, export_to
from trtship.config import ModelConfig, OptimizationProfile
from trtship.models import load_model
from trtship.models.inspection import ModelReport, inspect_model
from trtship.onnx import OnnxValidationReport, validate_onnx
from trtship.reporting import (
    render_model_report,
    render_model_report_text,
    render_onnx_validation_text,
)

F = "trtship_fixtures.models"


def make_report(*, depth: int = 3) -> ModelReport:
    config = ModelConfig.model_validate(
        {
            "name": "tiny",
            "kind": "module",
            "factory": f"{F}:tiny_mlp",
            "inputs": [{"name": "x", "shape": ["batch", 16]}],
        },
        context={"base_dir": Path.cwd()},
    )
    profile = OptimizationProfile.model_validate(
        {"inputs": {"x": {"min": [1, 16], "opt": [4, 16], "max": [8, 16]}}}
    )
    return inspect_model(load_model(config), config, [profile], max_depth=depth)


def test_text_report_contains_the_key_facts() -> None:
    text = render_model_report_text(make_report())
    for expected in (
        "tiny (module)",
        "Sequential",
        "676",
        "trainable",
        "2.64 KiB",
        "<= 1.06 KiB",
        "[batch, 16]",  # brackets survive Rich markup handling
        "[batch, 4]",
        "0.weight",
        "Linear",
    ):
        assert expected in text, expected


def test_text_report_has_no_ansi_escapes() -> None:
    assert "\x1b" not in render_model_report_text(make_report())


def test_tree_shows_hierarchy_and_elision() -> None:
    text = render_model_report_text(make_report(depth=0))
    assert "nested modules" in text
    assert "├──" not in text  # children hidden at depth 0
    full = render_model_report_text(make_report(depth=3))
    assert "├──" in full or "└──" in full


def test_unmeasured_activations_and_notes_are_shown() -> None:
    report = make_report()
    unmeasured = report.model_copy(
        update={
            "memory": report.memory.model_copy(
                update={
                    "activation_bytes_upper_bound": None,
                    "activation_probe_sizes": None,
                    "notes": ["activation size could not be measured: boom [x]"],
                }
            )
        }
    )
    text = render_model_report_text(unmeasured)
    assert "not measured" in text
    assert "could not be measured: boom [x]" in text


def test_render_writes_to_the_given_console() -> None:
    buffer = io.StringIO()
    render_model_report(make_report(), Console(file=buffer, width=100, color_system=None))
    assert "Memory estimate" in buffer.getvalue()


def test_trainable_split_appears_for_frozen_layers() -> None:
    config = ModelConfig.model_validate(
        {
            "name": "tiny",
            "kind": "module",
            "factory": f"{F}:tiny_mlp",
            "inputs": [{"name": "x", "shape": ["batch", 16]}],
        },
        context={"base_dir": Path.cwd()},
    )
    model = load_model(config)
    assert isinstance(model.module, nn.Sequential)
    for param in model.module[0].parameters():
        param.requires_grad_(False)
    text = render_model_report_text(inspect_model(model, config))
    assert "132 trainable" in text
    assert isinstance(torch.__version__, str)


# --------------------------------------------------------------------------- ONNX validation


def _validation_report(tmp_path: Path, *, broken: bool) -> OnnxValidationReport:
    if broken:
        config = build_config(
            "baked_batch",
            [{"name": "x", "shape": ["batch", 6]}],
            profile={"x": {"min": [1, 6], "opt": [4, 6], "max": [8, 6]}},
        )
    else:
        config = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)
    model, signature = export_to(config, tmp_path / "m.onnx")
    return validate_onnx(tmp_path / "m.onnx", model, signature, config)


def test_passing_validation_report_text(tmp_path: Path) -> None:
    text = render_onnx_validation_text(_validation_report(tmp_path, broken=False))
    for expected in (
        "PASSED",
        "onnxruntime",
        "CPUExecutionProvider",
        "opset 17",
        "Gemm x2",
        "min {'batch': 1}",
        "max {'batch': 8}",
        "pass",
        "atol=0.0001",
    ):
        assert expected in text, expected
    assert "FAILED" not in text
    assert "Failures" not in text
    assert "\x1b" not in text


def test_failing_validation_report_lists_every_failure(tmp_path: Path) -> None:
    report = _validation_report(tmp_path, broken=True)
    text = render_onnx_validation_text(report)
    assert "FAILED" in text
    assert "Failures" in text
    assert "shape mismatch: reference [8, 1], candidate [4, 1]" in text
    assert "error" in text  # the min point could not even run
    assert "[batch': 1]" not in text  # brackets/markup were escaped, not eaten
