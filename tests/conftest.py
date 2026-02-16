"""Shared fixtures and GPU-marker enforcement."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from trtship.utils import env


def minimal_config_dict() -> dict[str, Any]:
    return {
        "model": {
            "name": "tiny",
            "kind": "module",
            "factory": "trtship_fixtures.models:tiny_mlp",
            "inputs": [{"name": "x", "dtype": "float32", "shape": ["batch", 16]}],
        },
        "tensorrt": {
            "precisions": ["fp32"],
            "profiles": [{"inputs": {"x": {"min": [1, 16], "opt": [4, 16], "max": [8, 16]}}}],
        },
    }


@pytest.fixture
def config_dict() -> dict[str, Any]:
    return minimal_config_dict()


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[[dict[str, Any], str], Path]:
    def _write(data: dict[str, Any], name: str = "config.yaml") -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return _write


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """GPU-marked tests are skipped with an explicit reason when the hardware is absent.

    They are never allowed to pass vacuously: a skip is reported as a skip.
    """
    gpu_ok, reason = _gpu_state()
    trt_ok, trt_reason = _tensorrt_state(gpu_ok, reason)
    for item in items:
        if "gpu" in item.keywords and not gpu_ok:
            item.add_marker(pytest.mark.skip(reason=f"no usable NVIDIA GPU: {reason}"))
        if "tensorrt" in item.keywords and not trt_ok:
            item.add_marker(pytest.mark.skip(reason=trt_reason))


def _gpu_state() -> tuple[bool, str]:
    _, capability = env.probe_nvidia_gpu()
    return capability.ok, capability.detail or capability.status.value


def _tensorrt_state(gpu_ok: bool, gpu_reason: str) -> tuple[bool, str]:
    if not gpu_ok:
        return False, f"TensorRT tests need a GPU: {gpu_reason}"
    capability = env.probe_tensorrt()
    return capability.ok, f"TensorRT unavailable: {capability.detail}"
