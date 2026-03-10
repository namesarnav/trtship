"""Input/output signature inference.

Input specs are declared in config (an ``nn.Module`` cannot say what it accepts). Output specs are
derived by really running the model at two different assignments of the symbolic dimensions and
attributing each changing output dimension to the symbol whose size changed identically.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from pydantic import BaseModel, ConfigDict

from trtship.config import ModelConfig, OptimizationProfile
from trtship.errors import ModelError
from trtship.logging import get_logger
from trtship.models.dtypes import from_torch
from trtship.models.inputs import _PROBE_SIZES, make_inputs, resolve_symbol_sizes
from trtship.models.loader import LoadedModel
from trtship.specs import TensorSpec

log = get_logger(__name__)


class ModelSignature(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    # The two symbol-size assignments the outputs were derived from, kept for traceability.
    probe_sizes: list[dict[str, int]]


def _alternate_sizes(
    config: ModelConfig, profiles: Sequence[OptimizationProfile], base: Mapping[str, int]
) -> dict[str, int]:
    """A second size assignment differing from ``base`` in every symbol that can vary."""
    symbols = sorted({s for spec in config.inputs for s in spec.symbols})
    other_probe = {s: _PROBE_SIZES[(i + 1) % len(_PROBE_SIZES)] for i, s in enumerate(symbols)}
    smallest = resolve_symbol_sizes(config, profiles, "min", fallback=other_probe)
    largest = resolve_symbol_sizes(config, profiles, "max", fallback=other_probe)
    alternate: dict[str, int] = {}
    for symbol in symbols:
        if smallest[symbol] != base[symbol]:  # prefer the smaller shape: cheaper, less memory
            alternate[symbol] = smallest[symbol]
        elif largest[symbol] != base[symbol]:
            alternate[symbol] = largest[symbol]
        else:
            alternate[symbol] = base[symbol]  # pinned by the profile; cannot vary
    return alternate


def infer_signature(
    model: LoadedModel,
    config: ModelConfig,
    profiles: Sequence[OptimizationProfile] = (),
    *,
    seed: int = 0,
) -> ModelSignature:
    """Run the model and derive output specs (names, dtypes, static/symbolic shapes)."""
    sizes_a = resolve_symbol_sizes(config, profiles, "opt")
    sizes_b = _alternate_sizes(config, profiles, sizes_a)
    varies = sizes_a != sizes_b
    log.info("inferring signature with sizes %s%s", sizes_a, f" and {sizes_b}" if varies else "")

    out_a = _forward(model, sizes_a, seed)
    out_b = _forward(model, sizes_b, seed) if varies else out_a

    outputs: list[TensorSpec] = []
    for name, tensor_a in out_a.items():
        tensor_b = out_b[name]
        if tensor_a.dim() != tensor_b.dim():
            raise ModelError(
                f"output {name!r} changes rank ({tensor_a.dim()} vs {tensor_b.dim()}) with input "
                "size, which cannot be expressed as a fixed-rank tensor spec"
            )
        if tensor_a.dim() == 0:
            raise ModelError(
                f"output {name!r} is a scalar (0-d) tensor",
                hint="Return at least 1-d outputs (e.g. reshape to [1]); batched inference needs "
                "a dimension to index.",
            )
        dtype = from_torch(tensor_a.dtype)
        if tensor_b.dtype != tensor_a.dtype:
            raise ModelError(f"output {name!r} changes dtype with input size")
        shape = _attribute_dims(
            name, tuple(tensor_a.shape), tuple(tensor_b.shape), sizes_a, sizes_b
        )
        outputs.append(TensorSpec(name=name, dtype=dtype, shape=shape))
    return ModelSignature(
        inputs=list(config.inputs),
        outputs=outputs,
        probe_sizes=[dict(sizes_a), dict(sizes_b)] if varies else [dict(sizes_a)],
    )


def _forward(model: LoadedModel, sizes: Mapping[str, int], seed: int) -> dict[str, torch.Tensor]:
    try:
        inputs = make_inputs(model.inputs, sizes, seed=seed, device=model.device)
    except KeyError as exc:  # pragma: no cover - resolve_symbol_sizes covers every symbol
        raise ModelError(str(exc)) from exc
    try:
        return model.run(inputs)
    except ModelError as exc:
        exc.details.setdefault("symbol_sizes", dict(sizes))
        exc.hint = (
            (exc.hint or "")
            + f" (probe sizes used: {dict(sizes)}; declare tensorrt.profiles to choose valid sizes)"
        ).strip()
        raise


def _attribute_dims(
    name: str,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    sizes_a: Mapping[str, int],
    sizes_b: Mapping[str, int],
) -> list[int | str]:
    dims: list[int | str] = []
    for axis, (dim_a, dim_b) in enumerate(zip(shape_a, shape_b, strict=True)):
        if dim_a == dim_b:
            dims.append(dim_a)
            continue
        matches = sorted(s for s in sizes_a if sizes_a[s] == dim_a and sizes_b[s] == dim_b)
        # No single symbol explains the change (e.g. the dim is batch*seq): keep it dynamic
        # under a synthetic name rather than guessing.
        dims.append(matches[0] if matches else f"{name}_dim{axis}")
    return dims
