"""Real TensorRT engines validated against PyTorch on a real GPU. Skipped (with the reason) without
one; a skip is not a pass. Run on a GPU machine with:  uv sync --extra trt && pytest -m tensorrt -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import MLP_INPUT, MLP_PROFILE, build_config, export_to
from trtship.config import Precision
from trtship.models import infer_signature, load_model
from trtship.tensorrt import TensorRTExecutor, build_engine
from trtship.validation.engine import EngineUnderTest, validate_engines

pytestmark = [pytest.mark.gpu, pytest.mark.tensorrt]


@pytest.mark.parametrize("precision", [Precision.FP32, Precision.FP16])
def test_engine_matches_pytorch_within_the_precision_tolerance(
    tmp_path: Path, precision: Precision
) -> None:
    config = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)
    export_to(config, tmp_path / "m.onnx")
    model = load_model(config.model)
    signature = infer_signature(model, config.model, config.tensorrt.profiles)
    plan = tmp_path / f"{precision.value}.plan"
    build_engine(tmp_path / "m.onnx", signature, config, precision, plan)

    report = validate_engines(
        [EngineUnderTest(precision=precision, path=str(plan))],
        tmp_path / "m.onnx",
        model,
        signature,
        config,
        TensorRTExecutor,
    )
    assert report.passed, report.failures
    assert [p.label for p in report.results[0].points] == ["min", "opt", "max"]
