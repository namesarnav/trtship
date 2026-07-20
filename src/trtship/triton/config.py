"""Generate a Triton ``config.pbtxt`` from a built engine's real metadata.

Nothing about the model is hardcoded: tensor names, data types, and shapes come from the
:class:`EngineInfo` read from the engine itself, and the batching limit comes from the optimization
profile the engine was built with (unless the configuration sets a lower one).

Triton's convention: when ``max_batch_size > 0`` every tensor's leading axis is the batch axis and
is *omitted* from ``dims``; when it is 0, ``dims`` lists every axis and ``-1`` marks a variable one.
"""

from __future__ import annotations

from typing import Final

from trtship.config import TritonConfig
from trtship.errors import ConfigError, TritonError
from trtship.specs import DType
from trtship.tensorrt import EngineInfo, TensorBinding

_TRITON_DTYPES: Final[dict[DType, str]] = {
    DType.FLOAT32: "TYPE_FP32",
    DType.FLOAT16: "TYPE_FP16",
    DType.INT64: "TYPE_INT64",
    DType.INT32: "TYPE_INT32",
    DType.INT8: "TYPE_INT8",
    DType.UINT8: "TYPE_UINT8",
    DType.BOOL: "TYPE_BOOL",
}


def derive_max_batch_size(info: EngineInfo, settings: TritonConfig) -> int:
    """The ``max_batch_size`` to declare.

    The engine's profile bounds it (0 if the engine has no dynamic leading axis on every input).
    ``triton.max_batch_size`` may lower it or set 0 to disable batching, but cannot exceed it.
    """
    supported = info.max_batch_size()
    requested = settings.max_batch_size
    if requested is None:
        return supported
    if requested > supported:
        raise ConfigError(
            f"triton.max_batch_size is {requested} but the engine supports at most {supported}",
            hint="Rebuild with a larger `max` in tensorrt.profiles, or lower "
            "triton.max_batch_size.",
        )
    return requested


def _dims(binding: TensorBinding, max_batch_size: int) -> list[int]:
    if max_batch_size == 0:
        return list(binding.shape)
    if not binding.shape or binding.shape[0] != -1:
        raise TritonError(
            f"tensor {binding.name!r} has shape {binding.shape}, but Triton batching needs a "
            "dynamic leading (batch) axis on every tensor",
            hint="Set triton.max_batch_size: 0 to serve this engine without Triton batching.",
        )
    return list(binding.shape[1:])


def _tensor_block(binding: TensorBinding, max_batch_size: int) -> str:
    dims = ", ".join(str(d) for d in _dims(binding, max_batch_size))
    return (
        "  {\n"
        f'    name: "{binding.name}"\n'
        f"    data_type: {_TRITON_DTYPES[binding.dtype]}\n"
        f"    dims: [ {dims} ]\n"
        "  }"
    )


def render_config_pbtxt(model_name: str, info: EngineInfo, settings: TritonConfig) -> str:
    """The text of ``config.pbtxt`` for ``info`` served with ``settings``."""
    if not info.inputs or not info.outputs:
        raise TritonError("the engine reports no inputs or no outputs")
    max_batch = derive_max_batch_size(info, settings)
    batching = settings.dynamic_batching
    if batching is not None:
        if max_batch == 0:
            raise ConfigError(
                "triton.dynamic_batching needs batching, but max_batch_size resolves to 0",
                hint="Build with a dynamic batch dimension (a profile) and leave "
                "triton.max_batch_size unset, or remove triton.dynamic_batching.",
            )
        too_large = [b for b in batching.preferred_batch_sizes if b > max_batch]
        if too_large:
            raise ConfigError(
                f"triton.dynamic_batching.preferred_batch_sizes {too_large} exceed "
                f"max_batch_size {max_batch}"
            )

    lines = [
        f'name: "{model_name}"',
        'platform: "tensorrt_plan"',
        'default_model_filename: "model.plan"',
        f"max_batch_size: {max_batch}",
        "input [",
        ",\n".join(_tensor_block(b, max_batch) for b in info.inputs),
        "]",
        "output [",
        ",\n".join(_tensor_block(b, max_batch) for b in info.outputs),
        "]",
        "instance_group [",
        "  {",
        f"    count: {settings.instance_count}",
        "    kind: KIND_GPU",
        f"    gpus: [ {', '.join(str(g) for g in settings.instance_gpus)} ]",
        "  }",
        "]",
    ]
    if batching is not None:
        lines.append("dynamic_batching {")
        if batching.preferred_batch_sizes:
            sizes = ", ".join(str(b) for b in batching.preferred_batch_sizes)
            lines.append(f"  preferred_batch_size: [ {sizes} ]")
        lines.append(f"  max_queue_delay_microseconds: {batching.max_queue_delay_us}")
        lines.append("}")
    return "\n".join(lines) + "\n"
