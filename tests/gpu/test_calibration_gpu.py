"""Real INT8 calibration on a real GPU. Skipped (with the reason) without one; a skip is not a pass.

Run on a GPU machine with:  uv sync --extra trt && pytest -m tensorrt -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.helpers import MLP_INPUT, MLP_PROFILE, build_config, export_to
from trtship.calibration import calibrate, make_cache_calibrator, read_cache_dir
from trtship.config import Precision, TrtshipConfig
from trtship.models import infer_signature, load_model
from trtship.tensorrt import build_engine, load_tensorrt

pytestmark = [pytest.mark.gpu, pytest.mark.tensorrt]


def test_calibrate_then_build_an_int8_engine(tmp_path: Path) -> None:
    np.save(
        tmp_path / "data.npy", np.random.default_rng(0).standard_normal((64, 16)).astype("float32")
    )
    data = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE).model_dump(mode="json")
    data["tensorrt"]["precisions"] = ["int8"]
    data["calibration"] = {
        "dataset": "numpy",
        "path": str(tmp_path / "data.npy"),
        "num_samples": 64,
        "batch_size": 8,
    }
    config = TrtshipConfig.model_validate(data)
    model = load_model(config.model)
    signature = infer_signature(model, config.model, config.tensorrt.profiles)
    export_to(config, tmp_path / "m.onnx")

    metadata = calibrate(
        tmp_path / "m.onnx",
        signature,
        config,
        tmp_path / "cal",
        model_weights_sha256=model.weights_sha256,
    )
    cache, _ = read_cache_dir(tmp_path / "cal")
    assert len(cache) > 0
    assert metadata.sample_count == 64
    assert metadata.representative

    trt = load_tensorrt(purpose="test")
    engine = build_engine(
        tmp_path / "m.onnx",
        signature,
        config,
        Precision.INT8,
        tmp_path / "int8.plan",
        calibrator=make_cache_calibrator(trt, metadata.method, cache),
    )
    assert engine.size_bytes > 0
    assert "INT8" in engine.builder_flags
