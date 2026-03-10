"""Normalizing model outputs into a name -> tensor mapping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from trtship.errors import ModelError

DEFAULT_OUTPUT_NAME = "output"


def flatten_outputs(raw: Any, declared: Sequence[str] | None = None) -> dict[str, torch.Tensor]:
    """Turn a forward result (tensor, tuple/list of tensors, or dict of tensors) into named tensors.

    ``declared`` are the configured output names. For a single tensor the default name is
    ``output``; for a sequence, ``output_<i>``; for a dict, its keys (``None`` values, which some
    framework output objects contain for absent fields, are dropped).
    """
    if isinstance(raw, torch.Tensor):
        named: dict[str, torch.Tensor] = {DEFAULT_OUTPUT_NAME: raw}
        return _apply_declared(named, declared, positional=True)
    if isinstance(raw, Mapping):
        named = {str(k): v for k, v in raw.items() if v is not None}
        _require_tensors(named)
        if declared is not None:
            missing = [n for n in declared if n not in named]
            if missing or len(declared) != len(named):
                raise ModelError(
                    f"declared output_names {list(declared)} do not match the model's outputs "
                    f"{list(named)}",
                    hint="Remove model.output_names to use the model's own keys.",
                )
            return {n: named[n] for n in declared}
        return named
    if isinstance(raw, tuple | list):
        named = {f"{DEFAULT_OUTPUT_NAME}_{i}": v for i, v in enumerate(raw)}
        _require_tensors(named)
        return _apply_declared(named, declared, positional=True)
    raise ModelError(
        f"unsupported model output type {type(raw).__name__}",
        hint="forward() must return a tensor, a tuple/list of tensors, or a dict of tensors.",
    )


def _require_tensors(named: Mapping[str, Any]) -> None:
    bad = {k: type(v).__name__ for k, v in named.items() if not isinstance(v, torch.Tensor)}
    if bad:
        raise ModelError(
            f"model outputs must all be tensors; got {bad}",
            hint="Return only tensors from forward(), or wrap the model to drop other values.",
        )
    if not named:
        raise ModelError("model returned no tensor outputs")


def _apply_declared(
    named: dict[str, torch.Tensor], declared: Sequence[str] | None, *, positional: bool
) -> dict[str, torch.Tensor]:
    if declared is None:
        return named
    if not positional or len(declared) != len(named):
        raise ModelError(
            f"model.output_names has {len(declared)} names but the model returned "
            f"{len(named)} tensor(s)"
        )
    return dict(zip(declared, named.values(), strict=True))
