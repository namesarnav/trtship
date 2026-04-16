"""Static analysis of an ONNX graph: the official checker, shape inference, and statistics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final

import onnx
import onnx.shape_inference
from pydantic import BaseModel, ConfigDict

from trtship.errors import ValidationFailedError
from trtship.logging import get_logger

log = get_logger(__name__)

_STANDARD_DOMAINS: Final = frozenset({"", "ai.onnx", "ai.onnx.ml", "ai.onnx.training"})


class GraphReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    opset: int
    ir_version: int
    producer: str
    node_count: int  # top-level nodes; nodes inside If/Loop/Scan bodies are not counted
    op_counts: dict[str, int]  # op type -> count, most frequent first
    initializer_count: int
    initializer_bytes: int
    opset_domains: dict[str, int]  # domain -> version ('' is the default ONNX domain)
    custom_domains: list[str]
    inputs: list[str]
    outputs: list[str]
    outputs_without_shape: list[str]  # after strict shape inference
    warnings: list[str]


def _initializer_bytes(proto: onnx.ModelProto) -> int:
    total = 0
    for init in proto.graph.initializer:
        elements = 1
        for dim in init.dims:
            elements *= int(dim)
        total += elements * onnx.helper.tensor_dtype_to_np_dtype(init.data_type).itemsize
    return total


def analyze_graph(model_path: Path) -> GraphReport:
    """Run the ONNX checker (full check) and strict shape inference, and summarize the graph.

    Raises :class:`ValidationFailedError` if the checker or shape inference rejects the model.
    """
    try:
        proto = onnx.load(str(model_path))
    except Exception as exc:
        raise ValidationFailedError(
            f"cannot read ONNX model {model_path}: {type(exc).__name__}: {exc}",
            hint="The file is missing, truncated, or not an ONNX protobuf.",
        ) from exc
    try:
        onnx.checker.check_model(proto, full_check=True)
    except (onnx.checker.ValidationError, onnx.shape_inference.InferenceError) as exc:
        raise ValidationFailedError(
            f"the ONNX checker rejected {model_path.name}: {exc}",
            details={"stage": "checker", "message": str(exc)},
        ) from exc
    try:
        inferred = onnx.shape_inference.infer_shapes(proto, check_type=True, strict_mode=True)
    except Exception as exc:
        raise ValidationFailedError(
            f"ONNX shape inference failed for {model_path.name}: {exc}",
            details={"stage": "shape_inference", "message": str(exc)},
        ) from exc

    graph = proto.graph
    initializer_names = {i.name for i in graph.initializer}
    opsets = {entry.domain: int(entry.version) for entry in proto.opset_import}
    custom = sorted(d for d in opsets if d not in _STANDARD_DOMAINS)
    counts = Counter(node.op_type for node in graph.node)
    no_shape = [
        out.name for out in inferred.graph.output if not out.type.tensor_type.HasField("shape")
    ]
    warnings = [
        f"operator domain {domain!r} is not standard ONNX and is unlikely to be supported by "
        "TensorRT"
        for domain in custom
    ]
    if no_shape:
        warnings.append(f"shape inference could not determine the shape of outputs {no_shape}")
    return GraphReport(
        opset=opsets.get("", opsets.get("ai.onnx", 0)),
        ir_version=int(proto.ir_version),
        producer=f"{proto.producer_name} {proto.producer_version}".strip(),
        node_count=len(graph.node),
        op_counts=dict(counts.most_common()),
        initializer_count=len(graph.initializer),
        initializer_bytes=_initializer_bytes(proto),
        opset_domains=opsets,
        custom_domains=custom,
        inputs=[i.name for i in graph.input if i.name not in initializer_names],
        outputs=[o.name for o in graph.output],
        outputs_without_shape=no_shape,
        warnings=warnings,
    )
