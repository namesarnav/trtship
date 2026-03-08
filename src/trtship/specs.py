"""Tensor specifications shared by config, model inspection, export, validation, and Triton.

Deliberately free of torch/numpy imports so that importing configuration stays cheap.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

IDENTIFIER: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DType(StrEnum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT64 = "int64"
    INT32 = "int32"
    INT8 = "int8"
    UINT8 = "uint8"
    BOOL = "bool"

    @property
    def is_floating(self) -> bool:
        return self in (DType.FLOAT32, DType.FLOAT16)

    @property
    def is_integer(self) -> bool:
        return self in (DType.INT64, DType.INT32, DType.INT8, DType.UINT8)


class TensorSpec(BaseModel):
    """A named tensor: dtype and shape. String dims are symbolic (dynamic); ints are static.

    ``value_range`` bounds the values of randomly generated example data for inputs: uniform
    floats in ``[lo, hi)``, or integers in ``[lo, hi)``. It is what makes integer inputs such as
    token ids safe to fabricate (an out-of-vocabulary id would crash an embedding lookup).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    name: str
    dtype: DType = DType.FLOAT32
    shape: list[StrictInt | str] = Field(min_length=1)
    value_range: tuple[float, float] | None = None

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not IDENTIFIER.match(value):
            raise ValueError("must be an identifier (letters, digits, underscore)")
        return value

    @field_validator("shape")
    @classmethod
    def _valid_shape(cls, value: list[int | str]) -> list[int | str]:
        for position, dim in enumerate(value):
            if isinstance(dim, int) and dim < 1:
                raise ValueError(f"dim {position} must be >= 1, got {dim}")
            if isinstance(dim, str) and not IDENTIFIER.match(dim):
                raise ValueError(f"dim {position}: symbolic name {dim!r} is not an identifier")
        return value

    @model_validator(mode="after")
    def _valid_range(self) -> TensorSpec:
        if self.value_range is not None:
            low, high = self.value_range
            if not low < high:
                raise ValueError(
                    f"value_range must satisfy low < high, got {list(self.value_range)}"
                )
            if self.dtype is DType.BOOL:
                raise ValueError("value_range does not apply to bool tensors")
            if self.dtype.is_integer and not (float(low).is_integer() and float(high).is_integer()):
                raise ValueError("value_range bounds must be whole numbers for integer dtypes")
        return self

    @property
    def dynamic_axes(self) -> dict[int, str]:
        return {i: d for i, d in enumerate(self.shape) if isinstance(d, str)}

    @property
    def is_dynamic(self) -> bool:
        return bool(self.dynamic_axes)

    @property
    def symbols(self) -> frozenset[str]:
        return frozenset(self.dynamic_axes.values())

    def concrete_shape(self, sizes: Mapping[str, int]) -> tuple[int, ...]:
        """Resolve symbolic dims with ``sizes``; every symbol must be present."""
        resolved: list[int] = []
        for dim in self.shape:
            if isinstance(dim, int):
                resolved.append(dim)
            elif dim in sizes:
                resolved.append(sizes[dim])
            else:
                raise KeyError(f"no size given for symbolic dim {dim!r} of tensor {self.name!r}")
        return tuple(resolved)
