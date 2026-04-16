"""Helpers shared by tests that build configs and export models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trtship.config import TrtshipConfig
from trtship.export import export_onnx
from trtship.models import LoadedModel, ModelSignature, infer_signature, load_model

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
