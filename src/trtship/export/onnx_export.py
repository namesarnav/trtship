"""PyTorch -> ONNX export with signature verification.

Two exporters are supported, selected by ``export.dynamo``:

* TorchScript-tracing (default): no extra dependency; dynamic axes are declared per tensor.
  Deprecated upstream.
* ``torch.export``-based (``dynamo: true``): needs the ``onnxscript`` package
  (``uv sync --extra dynamo``); dynamic dimensions are declared with ``torch.export.Dim`` ranges
  taken from the first TensorRT optimization profile.

Whatever the exporter, the resulting graph is checked against the inferred
:class:`ModelSignature`: input/output names and order, dtypes, ranks, static dimensions, and that
symbolic dimensions were declared dynamic. This is a *structural* check of what the graph declares.
It cannot see a dimension that tracing turned into a constant inside the graph body while the
declared shapes still say "dynamic" (for example ``int(x.shape[0])`` in ``forward``); only running
the graph at more than one shape can, which is the job of the ONNX validation stage.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import secrets
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Final, Literal

import onnx
import torch
from pydantic import BaseModel, ConfigDict

from trtship.config import TrtshipConfig
from trtship.errors import ArtifactConflictError, EnvironmentUnavailableError, ExportError
from trtship.logging import get_logger
from trtship.models import (
    LoadedModel,
    ModelSignature,
    make_inputs,
    resolve_symbol_sizes,
    symbol_ranges,
)
from trtship.specs import DType, TensorSpec
from trtship.utils.fs import publish_new
from trtship.utils.hashing import sha256_file

log = get_logger(__name__)

METADATA_SCHEMA_VERSION: Final = 1
_MAX_WARNING_CHARS = 400
_ONNX_DTYPES: Final[dict[int, DType]] = {
    onnx.TensorProto.FLOAT: DType.FLOAT32,
    onnx.TensorProto.FLOAT16: DType.FLOAT16,
    onnx.TensorProto.INT64: DType.INT64,
    onnx.TensorProto.INT32: DType.INT32,
    onnx.TensorProto.INT8: DType.INT8,
    onnx.TensorProto.UINT8: DType.UINT8,
    onnx.TensorProto.BOOL: DType.BOOL,
}


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExportMetadata(_Frozen):
    schema_version: int = METADATA_SCHEMA_VERSION
    exporter: Literal["torchscript", "dynamo"]
    opset: int
    ir_version: int
    producer: str
    model_name: str
    weights_sha256: str
    torch_version: str
    onnx_version: str
    input_names: list[str]
    output_names: list[str]
    # tensor name -> {axis: symbolic dimension name}; only dynamic axes appear.
    dynamic_axes: dict[str, dict[int, str]]
    trace_sizes: dict[str, int]  # concrete sizes the model was traced at
    constant_folding: bool
    warnings: list[str]


class ExportResult(_Frozen):
    path: str
    sha256: str
    size_bytes: int
    metadata: ExportMetadata


# --------------------------------------------------------------------------- public API


def export_onnx(
    model: LoadedModel,
    signature: ModelSignature,
    config: TrtshipConfig,
    output_path: Path,
) -> ExportResult:
    """Export ``model`` to ``output_path`` (which must not exist) and verify the result."""
    if output_path.exists():
        raise ArtifactConflictError(
            f"refusing to overwrite existing artifact: {output_path}",
            hint="Artifacts are immutable. Choose a new output path.",
        )
    settings = config.export
    exporter: Literal["torchscript", "dynamo"] = "dynamo" if settings.dynamo else "torchscript"
    if settings.dynamo and importlib.util.find_spec("onnxscript") is None:
        raise EnvironmentUnavailableError(
            "export.dynamo is true but the 'onnxscript' package is not installed",
            hint="Install it with `uv sync --extra dynamo`, or set export.dynamo: false.",
        )

    sizes = resolve_symbol_sizes(config.model, config.tensorrt.profiles, "opt")
    example = make_inputs(model.inputs, sizes, seed=config.seed, device=model.device)
    args = tuple(example[spec.name] for spec in model.inputs)
    input_names = [spec.name for spec in signature.inputs]
    output_names = [spec.name for spec in signature.outputs]
    dynamic_axes = _dynamic_axes(signature)
    ranges = symbol_ranges(config.model, config.tensorrt.profiles)
    pinned = {s: lo for s, (lo, hi) in ranges.items() if hi is not None and lo == hi}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f".{output_path.name}.{secrets.token_hex(4)}.tmp")
    log.info("exporting %s to ONNX (opset %d, %s exporter)", model.name, settings.opset, exporter)
    try:
        with warnings.catch_warnings(record=True) as caught, _quiet() as chatter:
            warnings.simplefilter("always")
            try:
                if settings.dynamo:
                    _export_dynamo(
                        model, args, tmp, input_names, output_names, ranges, pinned, config
                    )
                else:
                    _export_torchscript(
                        model, args, tmp, input_names, output_names, dynamic_axes, config
                    )
            except Exception as exc:
                raise _translate(exc, exporter, settings.opset, sizes) from exc
        if chatter.getvalue().strip():
            log.debug("exporter output:\n%s", chatter.getvalue().strip())

        proto = onnx.load(str(tmp))
        problems = check_onnx_signature(proto, signature, pinned)
        if problems:
            raise ExportError(
                "the exported graph does not match the model signature",
                hint="The exporter dropped or changed a declared input/output property. Check "
                "the mismatches listed in the details; a dynamic axis that the exporter "
                "specialized to a constant usually points at shape-dependent Python code.",
                details={"problems": problems, "exporter": exporter},
            )
        metadata = ExportMetadata(
            exporter=exporter,
            opset=_default_opset(proto),
            ir_version=proto.ir_version,
            producer=f"{proto.producer_name} {proto.producer_version}".strip(),
            model_name=model.name,
            weights_sha256=model.weights_sha256,
            torch_version=model.torch_version,
            onnx_version=onnx.__version__,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            trace_sizes=dict(sizes),
            constant_folding=settings.constant_folding,
            warnings=_format_warnings(caught),
        )
        _embed_metadata(proto, metadata, tmp)
        sha256, size = sha256_file(tmp), tmp.stat().st_size
        publish_new(tmp, output_path)
    finally:
        tmp.unlink(missing_ok=True)
    log.info("exported %s (%d bytes, sha256 %s)", output_path, size, sha256[:12])
    return ExportResult(path=str(output_path), sha256=sha256, size_bytes=size, metadata=metadata)


def check_onnx_signature(
    proto: onnx.ModelProto,
    signature: ModelSignature,
    pinned: Mapping[str, int] | None = None,
) -> list[str]:
    """Compare an ONNX graph's I/O with a signature; returns human-readable mismatches.

    Names, order, dtype, rank and static dims must match. A symbolic dim must be dynamic in the
    graph (unless the symbol is ``pinned`` to a single size); input dims must additionally keep
    their symbol name.
    """
    pinned = pinned or {}
    graph = proto.graph
    initializers = {init.name for init in graph.initializer}
    actual_inputs = [vi for vi in graph.input if vi.name not in initializers]
    problems: list[str] = []
    for kind, expected, actual in (
        ("input", signature.inputs, actual_inputs),
        ("output", signature.outputs, list(graph.output)),
    ):
        if [s.name for s in expected] != [vi.name for vi in actual]:
            problems.append(
                f"{kind} names/order differ: expected {[s.name for s in expected]}, "
                f"graph has {[vi.name for vi in actual]}"
            )
            continue
        for spec, value_info in zip(expected, actual, strict=True):
            problems.extend(_check_tensor(kind, spec, value_info, pinned))
    return problems


# --------------------------------------------------------------------------- exporters


def _export_torchscript(
    model: LoadedModel,
    args: tuple[torch.Tensor, ...],
    tmp: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    config: TrtshipConfig,
) -> None:
    with torch.no_grad():
        torch.onnx.export(
            model.module,
            args,
            str(tmp),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes or None,
            opset_version=config.export.opset,
            do_constant_folding=config.export.constant_folding,
            dynamo=False,
        )


def _export_dynamo(
    model: LoadedModel,
    args: tuple[torch.Tensor, ...],
    tmp: Path,
    input_names: list[str],
    output_names: list[str],
    ranges: Mapping[str, tuple[int, int | None]],
    pinned: Mapping[str, int],
    config: TrtshipConfig,
) -> None:
    # One Dim per symbol, shared across inputs, so equal symbols stay equal in the graph.
    dims = {
        symbol: torch.export.Dim(symbol, min=low, max=high)
        for symbol, (low, high) in ranges.items()
        if symbol not in pinned
    }
    per_input: list[dict[int, Any] | None] = []
    for spec in model.inputs:
        axes = {axis: dims[sym] for axis, sym in spec.dynamic_axes.items() if sym in dims}
        per_input.append(axes or None)
    dynamic_shapes = tuple(per_input) if any(d is not None for d in per_input) else None
    with torch.no_grad():
        program = torch.onnx.export(
            model.module,
            args,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            opset_version=config.export.opset,
            optimize=config.export.constant_folding,
            dynamo=True,
        )
    if program is None:  # pragma: no cover - torch returns a program when no path is given
        raise ExportError("the torch.export-based exporter returned no program")
    program.save(str(tmp), external_data=False)


# --------------------------------------------------------------------------- helpers


def _dynamic_axes(signature: ModelSignature) -> dict[str, dict[int, str]]:
    axes: dict[str, dict[int, str]] = {}
    for spec in (*signature.inputs, *signature.outputs):
        if spec.dynamic_axes:
            axes[spec.name] = dict(spec.dynamic_axes)
    return axes


@contextlib.contextmanager
def _quiet() -> Iterator[io.StringIO]:
    """Capture the exporters' progress chatter (they write to stdout/stderr)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield buffer


def _format_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    seen: dict[str, None] = {}
    for item in caught:
        text = f"{item.category.__name__}: {' '.join(str(item.message).split())}"
        seen.setdefault(text[:_MAX_WARNING_CHARS], None)
    return list(seen)


def _translate(exc: Exception, exporter: str, opset: int, sizes: Mapping[str, int]) -> ExportError:
    name = type(exc).__name__
    details: dict[str, Any] = {
        "exporter": exporter,
        "opset": opset,
        "exception": name,
        "trace_sizes": dict(sizes),
    }
    hint = "Check the traceback details above and the model's ONNX compatibility."
    if name == "UnsupportedOperatorError":
        match = re.search(r"'([^']+)'", str(exc))
        if match:
            details["operator"] = match.group(1)
        hint = (
            "The exporter has no ONNX mapping for this operator at this opset. Try a higher "
            "export.opset, switch export.dynamo, or replace the operator in the model."
        )
    elif "onstraint" in str(exc) or "pecializ" in str(exc):
        hint = (
            "torch.export could not honor the dynamic dimensions. Check the shape ranges in "
            "tensorrt.profiles (an `opt` of 1 is specialized as a constant)."
        )
    return ExportError(f"ONNX export failed: {name}: {exc}", hint=hint, details=details)


def _default_opset(proto: onnx.ModelProto) -> int:
    for entry in proto.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return int(entry.version)
    raise ExportError("exported model declares no default-domain opset")


def _embed_metadata(proto: onnx.ModelProto, metadata: ExportMetadata, path: Path) -> None:
    props = {p.key: p.value for p in proto.metadata_props}
    props.update(
        {
            "trtship.model_name": metadata.model_name,
            "trtship.weights_sha256": metadata.weights_sha256,
            "trtship.exporter": metadata.exporter,
            "trtship.torch_version": metadata.torch_version,
        }
    )
    onnx.helper.set_model_props(proto, props)
    try:
        onnx.save(proto, str(path))
    except ValueError as exc:  # protobuf's 2 GiB limit
        raise ExportError(
            f"cannot serialize the ONNX model: {exc}",
            hint="Models over 2 GiB need external weight data, which trtship does not support yet.",
        ) from exc


def _dims(value_info: onnx.ValueInfoProto) -> list[int | str | None] | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    result: list[int | str | None] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param") and dim.dim_param:
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def _check_tensor(
    kind: str, spec: TensorSpec, value_info: onnx.ValueInfoProto, pinned: Mapping[str, int]
) -> list[str]:
    label = f"{kind} {spec.name!r}"
    actual = _dims(value_info)
    if actual is None:
        return [f"{label}: the graph declares no shape"]
    problems: list[str] = []
    elem_type = value_info.type.tensor_type.elem_type
    if _ONNX_DTYPES.get(elem_type) is not spec.dtype:
        problems.append(f"{label}: dtype is ONNX type {elem_type}, expected {spec.dtype.value}")
    if len(actual) != len(spec.shape):
        return [*problems, f"{label}: rank {len(actual)} in the graph, expected {len(spec.shape)}"]
    for axis, (want, got) in enumerate(zip(spec.shape, actual, strict=True)):
        if isinstance(want, int):
            if got != want:
                problems.append(f"{label} axis {axis}: expected static {want}, graph has {got!r}")
        elif want in pinned and got == pinned[want]:
            continue  # the profile pins this symbol to one size, so a constant is correct
        elif isinstance(got, int):
            problems.append(
                f"{label} axis {axis}: expected dynamic {want!r} but the graph fixed it to {got}"
            )
        elif kind == "input" and got != want:
            problems.append(f"{label} axis {axis}: expected symbol {want!r}, graph has {got!r}")
    return problems
