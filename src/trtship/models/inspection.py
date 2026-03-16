"""Model inspection: architecture, parameters, memory footprint, and I/O signature."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from trtship.config import ModelConfig, OptimizationProfile
from trtship.logging import get_logger
from trtship.models.inputs import make_inputs, resolve_symbol_sizes
from trtship.models.loader import LoadedModel
from trtship.models.signature import ModelSignature, infer_signature
from trtship.utils.timeutil import utc_now

log = get_logger(__name__)

REPORT_SCHEMA_VERSION = 1


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModuleNode(_Frozen):
    name: str
    type: str
    parameters: int  # includes all descendants
    trainable_parameters: int
    children: list[ModuleNode] = Field(default_factory=list)
    elided_modules: int = 0  # descendants hidden by the depth limit


class TensorInfo(_Frozen):
    name: str
    shape: list[int]
    dtype: str
    numel: int
    bytes: int
    requires_grad: bool


class ParameterSummary(_Frozen):
    total: int
    trainable: int
    frozen: int
    tensors: int
    by_dtype: dict[str, int]  # dtype -> parameter count
    buffer_tensors: int
    buffer_elements: int
    largest: list[TensorInfo]


class MemoryEstimate(_Frozen):
    parameter_bytes: int
    buffer_bytes: int
    weights_bytes: int
    # Sum of every leaf module's output size during one forward pass at `activation_probe_sizes`.
    # An upper bound for inference without buffer reuse; None when it could not be measured.
    activation_bytes_upper_bound: int | None
    activation_probe_sizes: dict[str, int] | None
    notes: list[str]


class ModelReport(_Frozen):
    schema_version: int = REPORT_SCHEMA_VERSION
    generated_at: datetime
    name: str
    kind: str
    source_path: str | None
    weights_sha256: str
    torch_version: str
    device: str
    root_type: str
    module_count: int
    parameters: ParameterSummary
    memory: MemoryEstimate
    architecture: ModuleNode
    signature: ModelSignature


def _tensor_info(name: str, tensor: torch.Tensor) -> TensorInfo:
    return TensorInfo(
        name=name,
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        numel=tensor.numel(),
        bytes=tensor.numel() * tensor.element_size(),
        requires_grad=tensor.requires_grad,
    )


def _count(module: nn.Module) -> tuple[int, int]:
    """(total, trainable) parameter elements; shared parameters are counted once."""
    params = list(module.parameters())
    return sum(p.numel() for p in params), sum(p.numel() for p in params if p.requires_grad)


def _build_tree(name: str, module: nn.Module, depth: int, max_depth: int) -> ModuleNode:
    total, trainable = _count(module)
    children: list[ModuleNode] = []
    elided = 0
    if depth < max_depth:
        for child_name, child in module.named_children():
            path = f"{name}.{child_name}" if name else child_name
            children.append(_build_tree(path, child, depth + 1, max_depth))
    else:
        elided = sum(1 for _ in module.modules()) - 1
    return ModuleNode(
        name=name or "<root>",
        type=type(module).__name__,
        parameters=total,
        trainable_parameters=trainable,
        children=children,
        elided_modules=elided,
    )


def _summarize_parameters(module: nn.Module, top: int) -> ParameterSummary:
    named = list(module.named_parameters())
    total = sum(p.numel() for _, p in named)
    trainable = sum(p.numel() for _, p in named if p.requires_grad)
    by_dtype: Counter[str] = Counter()
    for _, param in named:
        by_dtype[str(param.dtype)] += param.numel()
    buffers = list(module.buffers())
    largest = sorted(named, key=lambda item: item[1].numel(), reverse=True)[:top]
    return ParameterSummary(
        total=total,
        trainable=trainable,
        frozen=total - trainable,
        tensors=len(named),
        by_dtype=dict(by_dtype),
        buffer_tensors=len(buffers),
        buffer_elements=sum(b.numel() for b in buffers),
        largest=[_tensor_info(name, p) for name, p in largest],
    )


def _tensor_bytes(value: Any) -> int:
    """Total size of every tensor reachable in a (possibly nested) module output."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, tuple | list):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def _measure_activations(
    model: LoadedModel, sizes: dict[str, int], seed: int
) -> tuple[int | None, str | None]:
    """Sum leaf-module output sizes over one forward pass. Returns (bytes, note-if-unavailable)."""
    fired = 0
    total = 0

    def hook(_module: nn.Module, _args: Any, output: Any) -> None:
        nonlocal fired, total
        fired += 1
        total += _tensor_bytes(output)

    handles = []
    try:
        for module in model.module.modules():
            if next(module.children(), None) is None:  # leaf
                handles.append(module.register_forward_hook(hook))
        model.run(make_inputs(model.inputs, sizes, seed=seed, device=model.device))
    except Exception as exc:
        return None, f"activation size could not be measured: {type(exc).__name__}: {exc}"
    finally:
        for handle in handles:
            handle.remove()
    if fired == 0:
        return None, "activation size unavailable: no forward hooks fired (e.g. a scripted module)"
    return total, None


def inspect_model(
    model: LoadedModel,
    config: ModelConfig,
    profiles: Sequence[OptimizationProfile] = (),
    *,
    seed: int = 0,
    max_depth: int = 3,
    top: int = 10,
    signature: ModelSignature | None = None,
) -> ModelReport:
    """Inspect a loaded model. Runs real forward passes to derive shapes and activation sizes."""
    module = model.module
    signature = signature or infer_signature(model, config, profiles, seed=seed)
    parameters = _summarize_parameters(module, top)

    param_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in module.buffers())
    sizes = resolve_symbol_sizes(config, profiles, "opt")
    activation, note = _measure_activations(model, sizes, seed)
    memory = MemoryEstimate(
        parameter_bytes=param_bytes,
        buffer_bytes=buffer_bytes,
        weights_bytes=param_bytes + buffer_bytes,
        activation_bytes_upper_bound=activation,
        activation_probe_sizes=sizes if activation is not None else None,
        notes=[note] if note else [],
    )
    log.info(
        "inspected %s: %d parameters, %d modules",
        model.name,
        parameters.total,
        len(list(module.modules())),
    )
    return ModelReport(
        generated_at=utc_now(),
        name=model.name,
        kind=model.kind.value,
        source_path=str(model.source_path) if model.source_path else None,
        weights_sha256=model.weights_sha256,
        torch_version=model.torch_version,
        device=str(model.device),
        root_type=type(module).__name__,
        module_count=sum(1 for _ in module.modules()),
        parameters=parameters,
        memory=memory,
        architecture=_build_tree("", module, 0, max_depth),
        signature=signature,
    )
