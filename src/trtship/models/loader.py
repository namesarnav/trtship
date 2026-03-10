"""Load a model described by :class:`ModelConfig` into a :class:`LoadedModel`.

Trust model:

* ``kind: module`` imports and calls the configured factory, i.e. it runs Python code from your
  environment by design. Only run configs you trust.
* ``kind: checkpoint`` and ``kind: torchscript`` read serialized files. Checkpoints load with
  ``weights_only=True`` unless ``trust_source`` is set; TorchScript archives can carry executable
  content and therefore always require ``trust_source``.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from trtship.config import ModelConfig, ModelKind
from trtship.errors import ModelError
from trtship.logging import get_logger
from trtship.models.outputs import flatten_outputs
from trtship.specs import TensorSpec
from trtship.utils import env

log = get_logger(__name__)

_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model")
_MAX_LISTED_KEYS = 10


@dataclass
class LoadedModel:
    """A model in eval mode plus everything needed to run and identify it."""

    module: nn.Module
    name: str
    kind: ModelKind
    inputs: tuple[TensorSpec, ...]
    declared_output_names: tuple[str, ...] | None
    device: torch.device
    source_path: Path | None
    weights_sha256: str
    torch_version: str

    def run(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass with named inputs, returning named outputs.

        Inputs are passed positionally in configured order, so the configured input order must
        match ``forward``'s parameter order.
        """
        missing = [spec.name for spec in self.inputs if spec.name not in inputs]
        if missing:
            raise ModelError(f"missing model inputs: {missing}")
        args = [inputs[spec.name].to(self.device) for spec in self.inputs]
        try:
            with torch.inference_mode():
                raw = self.module(*args)
        except Exception as exc:
            shapes = {spec.name: list(a.shape) for spec, a in zip(self.inputs, args, strict=True)}
            raise ModelError(
                f"model forward pass failed: {type(exc).__name__}: {exc}",
                hint="Check that the configured input shapes, dtypes and value_range are valid "
                "for this model, and that the input order matches forward().",
                details={"input_shapes": shapes, "exception": type(exc).__name__},
            ) from exc
        return flatten_outputs(raw, self.declared_output_names)


def hash_state_dict(state: Mapping[str, torch.Tensor]) -> str:
    """Identity of a set of weights, independent of the file format they came from.

    Hashes names, dtypes, shapes and raw bytes in sorted-name order, streaming tensor by tensor.
    """
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(f"{key}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        if tensor.numel():
            digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def resolve_device(device: str) -> torch.device:
    """Resolve a device string. CUDA without a usable GPU is an error, never a fallback."""
    resolved = torch.device(device)
    if resolved.type == "cuda":
        env.require(env.NVIDIA_GPU, purpose=f"running the model on {device}")
        env.require(env.TORCH_CUDA, purpose=f"running the model on {device}")
    elif resolved.type != "cpu":
        raise ModelError(f"unsupported device {device!r}", hint="Use 'cpu' or 'cuda[:N]'.")
    return resolved


def load_model(config: ModelConfig, *, device: str = "cpu") -> LoadedModel:
    """Load ``config`` onto ``device`` in eval mode."""
    target = resolve_device(device)
    loaders: dict[ModelKind, Callable[[ModelConfig, torch.device], nn.Module]] = {
        ModelKind.MODULE: _load_module,
        ModelKind.CHECKPOINT: _load_checkpoint,
        ModelKind.TORCHSCRIPT: _load_torchscript,
    }
    module = loaders[config.kind](config, target)
    module.eval()
    module.to(target)
    weights_sha256 = hash_state_dict(module.state_dict())
    log.info(
        "loaded model %s (%s) weights_sha256=%s",
        config.name,
        config.kind.value,
        weights_sha256[:12],
    )
    return LoadedModel(
        module=module,
        name=config.name,
        kind=config.kind,
        inputs=tuple(config.inputs),
        declared_output_names=tuple(config.output_names) if config.output_names else None,
        device=target,
        source_path=config.path,
        weights_sha256=weights_sha256,
        torch_version=str(torch.__version__),
    )


# --------------------------------------------------------------------------- kinds


def _import_factory(spec: str) -> Callable[..., Any]:
    module_name, _, attr_path = spec.partition(":")
    try:
        target: Any = importlib.import_module(module_name)
    except ImportError as exc:
        raise ModelError(
            f"cannot import factory module {module_name!r}: {exc}",
            hint="Make the module importable (install it, or add its directory to PYTHONPATH).",
        ) from exc
    for part in attr_path.split("."):
        try:
            target = getattr(target, part)
        except AttributeError:
            raise ModelError(
                f"factory {spec!r}: {module_name!r} has no attribute path {attr_path!r}"
            ) from None
    if not callable(target):
        raise ModelError(f"factory {spec!r} is not callable")
    return target  # type: ignore[no-any-return]


def _load_module(config: ModelConfig, device: torch.device) -> nn.Module:
    assert config.factory is not None  # guaranteed by ModelConfig validation
    return _build_from_factory(config.factory, config.factory_kwargs)


def _build_from_factory(factory_spec: str, kwargs: Mapping[str, Any]) -> nn.Module:
    factory = _import_factory(factory_spec)
    try:
        built = factory(**kwargs)
    except Exception as exc:
        raise ModelError(
            f"factory {factory_spec!r} raised {type(exc).__name__}: {exc}",
            details={"exception": type(exc).__name__},
        ) from exc
    if not isinstance(built, nn.Module):
        raise ModelError(
            f"factory {factory_spec!r} returned {type(built).__name__}, expected torch.nn.Module"
        )
    return built


def _read_path(config: ModelConfig) -> Path:
    assert config.path is not None  # guaranteed by ModelConfig validation
    if not config.path.is_file():
        raise ModelError(f"model file not found: {config.path}")
    return config.path


def _load_checkpoint(config: ModelConfig, device: torch.device) -> nn.Module:
    assert config.factory is not None
    path = _read_path(config)
    module = _build_from_factory(config.factory, config.factory_kwargs)
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=not config.trust_source)
    except Exception as exc:
        hint = (
            "The file is not a plain tensor checkpoint. Only set model.trust_source: true for "
            "files you trust, since loading them can execute arbitrary code."
            if not config.trust_source
            else None
        )
        raise ModelError(
            f"cannot load checkpoint {path}: {type(exc).__name__}: {exc}", hint=hint
        ) from exc
    state = _extract_state_dict(loaded)
    try:
        result = module.load_state_dict(state, strict=False)
    except RuntimeError as exc:  # shape mismatches surface here
        raise ModelError(f"checkpoint {path} does not fit the model: {exc}") from exc
    if result.missing_keys or result.unexpected_keys:
        raise ModelError(
            "checkpoint keys do not match the model",
            hint="Check that the factory builds the same architecture the checkpoint came from.",
            details={
                "missing_keys": _clip(result.missing_keys),
                "unexpected_keys": _clip(result.unexpected_keys),
            },
        )
    return module


def _extract_state_dict(loaded: Any) -> dict[str, torch.Tensor]:
    if isinstance(loaded, nn.Module):
        loaded = loaded.state_dict()
    if not isinstance(loaded, Mapping):
        raise ModelError(f"checkpoint holds a {type(loaded).__name__}, expected a state dict")
    for key in _STATE_DICT_KEYS:
        inner = loaded.get(key)
        if isinstance(inner, Mapping):
            loaded = inner
            break
    state = dict(loaded)
    bad = [k for k, v in state.items() if not isinstance(v, torch.Tensor)]
    if bad:
        raise ModelError(
            "checkpoint is not a plain state dict; some entries are not tensors",
            hint="Save with torch.save(model.state_dict(), path).",
            details={"non_tensor_keys": _clip(bad)},
        )
    if state and all(k.startswith("module.") for k in state):  # saved from DataParallel/DDP
        state = {k.removeprefix("module."): v for k, v in state.items()}
    return state


def _load_torchscript(config: ModelConfig, device: torch.device) -> nn.Module:
    path = _read_path(config)
    if not config.trust_source:
        raise ModelError(
            "loading a TorchScript archive requires model.trust_source: true",
            hint="TorchScript archives can carry executable content; only enable this for files "
            "you trust.",
        )
    log.warning("loading TorchScript archive %s with trust_source=true", path)
    try:
        archive: nn.Module = torch.jit.load(str(path), map_location=device)  # type: ignore[no-untyped-call]
        return archive
    except Exception as exc:
        raise ModelError(
            f"cannot load TorchScript archive {path}: {type(exc).__name__}: {exc}"
        ) from exc


def _clip(keys: Sequence[str]) -> list[str]:
    shown = list(keys[:_MAX_LISTED_KEYS])
    if len(keys) > _MAX_LISTED_KEYS:
        shown.append(f"... and {len(keys) - _MAX_LISTED_KEYS} more")
    return shown
