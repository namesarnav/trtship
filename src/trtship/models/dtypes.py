"""Mappings between trtship dtypes, torch dtypes, and numpy dtypes."""

from __future__ import annotations

from typing import Final

import numpy as np
import torch

from trtship.errors import ModelError
from trtship.specs import DType

_TORCH: Final[dict[DType, torch.dtype]] = {
    DType.FLOAT32: torch.float32,
    DType.FLOAT16: torch.float16,
    DType.INT64: torch.int64,
    DType.INT32: torch.int32,
    DType.INT8: torch.int8,
    DType.UINT8: torch.uint8,
    DType.BOOL: torch.bool,
}
_FROM_TORCH: Final[dict[torch.dtype, DType]] = {v: k for k, v in _TORCH.items()}
_NUMPY: Final[dict[DType, np.dtype[np.generic]]] = {
    DType.FLOAT32: np.dtype(np.float32),
    DType.FLOAT16: np.dtype(np.float16),
    DType.INT64: np.dtype(np.int64),
    DType.INT32: np.dtype(np.int32),
    DType.INT8: np.dtype(np.int8),
    DType.UINT8: np.dtype(np.uint8),
    DType.BOOL: np.dtype(np.bool_),
}


def to_torch(dtype: DType) -> torch.dtype:
    return _TORCH[dtype]


def to_numpy(dtype: DType) -> np.dtype[np.generic]:
    return _NUMPY[dtype]


def from_torch(dtype: torch.dtype) -> DType:
    try:
        return _FROM_TORCH[dtype]
    except KeyError:
        supported = ", ".join(sorted(str(t) for t in _FROM_TORCH))
        raise ModelError(
            f"unsupported tensor dtype {dtype}",
            hint=f"Cast the model's inputs/outputs to one of: {supported}.",
        ) from None
