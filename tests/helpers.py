"""Helpers shared by tests that build configs and export models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trtship import __version__
from trtship.config import TrtshipConfig
from trtship.export import export_onnx
from trtship.models import LoadedModel, ModelSignature, infer_signature, load_model
from trtship.utils import env
from trtship.utils.env import Capability, CapabilityStatus, EnvironmentReport, GpuDevice
from trtship.utils.timeutil import utc_now

FIXTURES = "trtship_fixtures.models"
MLP_INPUT = {"name": "x", "shape": ["batch", 16]}
MLP_PROFILE = {"x": {"min": [1, 16], "opt": [4, 16], "max": [8, 16]}}
TOKEN_INPUTS = [
    {"name": "input_ids", "dtype": "int64", "shape": ["batch", "seq"], "value_range": [0, 100]},
    {"name": "attention_mask", "dtype": "int64", "shape": ["batch", "seq"], "value_range": [0, 2]},
]
TOKEN_PROFILE = {
    "input_ids": {"min": [1, 8], "opt": [4, 32], "max": [8, 64]},
    "attention_mask": {"min": [1, 8], "opt": [4, 32], "max": [8, 64]},
}


def build_config(
    factory: str,
    inputs: list[dict[str, Any]],
    *,
    profile: dict[str, Any] | None = None,
    export: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    **model_extra: Any,
) -> TrtshipConfig:
    data: dict[str, Any] = {
        "model": {
            "name": "m",
            "kind": "module",
            "factory": f"{FIXTURES}:{factory}",
            "inputs": inputs,
            **model_extra,
        },
        "export": export or {},
    }
    if profile:
        data["tensorrt"] = {"profiles": [{"inputs": profile}]}
    if validation:
        data["validation"] = validation
    return TrtshipConfig.model_validate(data)


def export_to(config: TrtshipConfig, path: Path) -> tuple[LoadedModel, ModelSignature]:
    """Load, infer the signature, and export ``config``'s model to ``path``."""
    model = load_model(config.model)
    signature = infer_signature(model, config.model, config.tensorrt.profiles, seed=config.seed)
    export_onnx(model, signature, config, path)
    return model, signature


def fake_environment(*, gpu_ok: bool, core_ok: bool = True) -> EnvironmentReport:
    """A hand-built environment report, so tests do not depend on this machine's hardware."""

    def cap(
        name: str, ok: bool, version: str | None = None, detail: str | None = None
    ) -> Capability:
        return Capability(
            name=name,
            status=CapabilityStatus.OK if ok else CapabilityStatus.MISSING,
            version=version if ok else None,
            detail=detail,
        )

    caps = [
        cap(env.PYTHON, core_ok, "3.12.0"),
        cap(env.TORCH, True, "2.0"),
        cap(env.TORCH_CUDA, gpu_ok, "12.4"),
        cap(env.NVIDIA_GPU, gpu_ok, "550", "gpu"),
        cap(env.CUDA_TOOLKIT, True, "12.4"),
        cap(env.TENSORRT, gpu_ok, "10.3", "TensorRT bindings not installed"),
        cap(env.ONNX, True, "1.16"),
        cap(env.ONNXRUNTIME, True, "1.18"),
        cap(env.TRITON_CLIENT, False, None, "not installed (pip install 'trtship[triton]')"),
        cap(env.DOCKER, True, "27"),
        cap(env.DOCKER_NVIDIA_RUNTIME, gpu_ok),
        cap(env.TRITON_SERVER, False),
    ]
    gpus = [GpuDevice(index=0, name="Test GPU", compute_capability="8.6")] if gpu_ok else []
    return EnvironmentReport(
        captured_at=utc_now(),
        trtship_version=__version__,
        python_version="3.12.0",
        platform="test",
        capabilities=caps,
        gpus=gpus,
    )
