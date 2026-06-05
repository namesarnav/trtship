"""Describing a built engine: its I/O tensors, dtypes, shapes, and profile ranges.

:class:`EngineInfo` is the value object Triton configuration generation consumes, so a
``config.pbtxt`` is derived from the real engine and never hardcoded. Extraction supports both the
TensorRT >= 8.5 tensor API (the only one in TensorRT 10) and the older bindings API.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from trtship.errors import EngineBuildError
from trtship.specs import DType

# TensorRT DataType member name -> trtship dtype.
_DTYPE_BY_NAME: dict[str, DType] = {
    "FLOAT": DType.FLOAT32,
    "HALF": DType.FLOAT16,
    "INT8": DType.INT8,
    "INT32": DType.INT32,
    "INT64": DType.INT64,
    "BOOL": DType.BOOL,
    "UINT8": DType.UINT8,
}


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProfileRange(_Frozen):
    min: list[int]
    opt: list[int]
    max: list[int]


class TensorBinding(_Frozen):
    name: str
    mode: Literal["input", "output"]
    dtype: DType
    shape: list[int]  # -1 marks a dynamic dimension
    profiles: list[ProfileRange] = Field(default_factory=list)  # inputs only, one per profile

    @property
    def is_dynamic(self) -> bool:
        return any(d < 0 for d in self.shape)


class EngineInfo(_Frozen):
    tensorrt_version: str
    num_optimization_profiles: int
    tensors: list[TensorBinding]

    @property
    def inputs(self) -> list[TensorBinding]:
        return [t for t in self.tensors if t.mode == "input"]

    @property
    def outputs(self) -> list[TensorBinding]:
        return [t for t in self.tensors if t.mode == "output"]

    def max_batch_size(self) -> int:
        """Largest first-axis size every input supports, or 0 if the first axis is not dynamic.

        Triton's ``max_batch_size`` is the largest value of the leading (batch) dimension, so it is
        only meaningful when every input has a dynamic leading dimension.
        """
        limits = []
        for tensor in self.inputs:
            if not tensor.shape or tensor.shape[0] >= 0 or not tensor.profiles:
                return 0
            limits.append(max(profile.max[0] for profile in tensor.profiles))
        return min(limits) if limits else 0


def convert_dtype(trt: Any, dtype: Any) -> DType:
    """Map a TensorRT ``DataType`` to a trtship dtype."""
    for name, mapped in _DTYPE_BY_NAME.items():
        member = getattr(trt.DataType, name, None)
        if member is not None and dtype == member:
            return mapped
    raise EngineBuildError(
        f"the engine uses a tensor dtype trtship does not support: {dtype}",
        hint="Cast the model's inputs and outputs to float32, float16, int32, int64, int8, "
        "uint8 or bool.",
    )


def _ranges_from_tensor_api(engine: Any, name: str) -> list[ProfileRange]:
    ranges = []
    for profile in range(int(engine.num_optimization_profiles)):
        low, opt, high = engine.get_tensor_profile_shape(name, profile)
        ranges.append(ProfileRange(min=list(low), opt=list(opt), max=list(high)))
    return ranges


def describe_engine(trt: Any, engine: Any) -> EngineInfo:
    """Read the I/O description of a deserialized ``ICudaEngine``."""
    tensors: list[TensorBinding] = []
    profiles = int(engine.num_optimization_profiles)
    if hasattr(engine, "num_io_tensors"):
        for index in range(int(engine.num_io_tensors)):
            name = engine.get_tensor_name(index)
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = [int(d) for d in engine.get_tensor_shape(name)]
            ranges = (
                _ranges_from_tensor_api(engine, name)
                if is_input and any(d < 0 for d in shape)
                else []
            )
            tensors.append(
                TensorBinding(
                    name=name,
                    mode="input" if is_input else "output",
                    dtype=convert_dtype(trt, engine.get_tensor_dtype(name)),
                    shape=shape,
                    profiles=ranges,
                )
            )
    else:  # bindings API (TensorRT < 8.5)
        for index in range(int(engine.num_bindings)):
            is_input = bool(engine.binding_is_input(index))
            shape = [int(d) for d in engine.get_binding_shape(index)]
            ranges = []
            if is_input and any(d < 0 for d in shape):
                for profile in range(profiles):
                    low, opt, high = engine.get_profile_shape(profile, index)
                    ranges.append(ProfileRange(min=list(low), opt=list(opt), max=list(high)))
            tensors.append(
                TensorBinding(
                    name=engine.get_binding_name(index),
                    mode="input" if is_input else "output",
                    dtype=convert_dtype(trt, engine.get_binding_dtype(index)),
                    shape=shape,
                    profiles=ranges,
                )
            )
    return EngineInfo(
        tensorrt_version=str(trt.__version__),
        num_optimization_profiles=profiles,
        tensors=tensors,
    )
