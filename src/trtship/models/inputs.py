"""Deterministic example inputs and symbolic-dimension size resolution."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Literal

import torch

from trtship.config import ModelConfig, OptimizationProfile
from trtship.errors import ConfigError, ModelError
from trtship.models.dtypes import to_torch
from trtship.specs import DType, TensorSpec

DEFAULT_INT_RANGE = (0, 2)
_INT_LIMITS = {
    DType.INT8: (-(2**7), 2**7),
    DType.UINT8: (0, 2**8),
    DType.INT32: (-(2**31), 2**31),
    DType.INT64: (-(2**63), 2**63),
}
# Distinct small primes: distinct sizes per symbol keep output-dim attribution unambiguous.
_PROBE_SIZES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)

Which = Literal["min", "opt", "max"]


def _generator(seed: int, name: str) -> torch.Generator:
    """Per-input generator: adding or reordering inputs never changes another input's data."""
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF)
    return generator


def make_input(spec: TensorSpec, sizes: Mapping[str, int], *, seed: int) -> torch.Tensor:
    """Generate one input tensor on CPU. Identical arguments always give identical data."""
    shape = spec.concrete_shape(sizes)
    generator = _generator(seed, spec.name)
    dtype = spec.dtype
    if dtype is DType.BOOL:
        return torch.rand(shape, generator=generator) < 0.5
    if dtype.is_floating:
        if spec.value_range is None:
            data = torch.randn(shape, generator=generator)
        else:
            low, high = spec.value_range
            data = low + (high - low) * torch.rand(shape, generator=generator)
        return data.to(to_torch(dtype))
    low_i, high_i = (
        (int(spec.value_range[0]), int(spec.value_range[1]))
        if spec.value_range is not None
        else DEFAULT_INT_RANGE
    )
    limit_low, limit_high = _INT_LIMITS[dtype]
    if low_i < limit_low or high_i > limit_high:
        raise ModelError(
            f"input {spec.name!r}: value_range [{low_i}, {high_i}) does not fit {dtype.value}",
            hint=f"{dtype.value} holds [{limit_low}, {limit_high}).",
        )
    return torch.randint(low_i, high_i, shape, generator=generator, dtype=torch.int64).to(
        to_torch(dtype)
    )


def make_inputs(
    specs: Sequence[TensorSpec],
    sizes: Mapping[str, int],
    *,
    seed: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Example inputs keyed by name. Data is generated on CPU, then moved, so it is identical
    across devices."""
    return {spec.name: make_input(spec, sizes, seed=seed).to(device) for spec in specs}


def symbol_ranges(
    model: ModelConfig, profiles: Sequence[OptimizationProfile]
) -> dict[str, tuple[int, int | None]]:
    """``(min, max)`` per symbolic dim from the first profile; ``(1, None)`` when unconstrained."""
    symbols = sorted({s for spec in model.inputs for s in spec.symbols})
    lows = resolve_symbol_sizes(model, profiles, "min")
    highs = resolve_symbol_sizes(model, profiles, "max")
    covered: set[str] = set()
    if profiles:
        for name in profiles[0].inputs:
            covered |= model.input(name).symbols
    return {s: (lows[s], highs[s]) if s in covered else (1, None) for s in symbols}


def resolve_symbol_sizes(
    model: ModelConfig,
    profiles: Sequence[OptimizationProfile],
    which: Which,
    *,
    fallback: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Concrete sizes for every symbolic dim.

    Sizes come from the first optimization profile's ``which`` shape. Symbols the profile does not
    constrain take ``fallback`` values, then distinct probe sizes. A symbol given different sizes by
    different inputs is a configuration error.
    """
    sizes: dict[str, int] = {}
    if profiles:
        for name, shape_range in profiles[0].inputs.items():
            spec = model.input(name)
            chosen = getattr(shape_range, which)
            for axis, symbol in spec.dynamic_axes.items():
                if sizes.setdefault(symbol, chosen[axis]) != chosen[axis]:
                    raise ConfigError(
                        f"symbolic dim {symbol!r} is given different {which} sizes "
                        f"({sizes[symbol]} and {chosen[axis]}) by different inputs in "
                        "tensorrt.profiles[0]",
                        hint="A symbol names one dimension; use distinct symbols for "
                        "independent dimensions.",
                    )
    symbols = sorted({s for spec in model.inputs for s in spec.symbols})
    for index, symbol in enumerate(symbols):
        if symbol in sizes:
            continue
        if fallback is not None and symbol in fallback:
            sizes[symbol] = fallback[symbol]
        else:
            sizes[symbol] = _PROBE_SIZES[index % len(_PROBE_SIZES)]
    return sizes
