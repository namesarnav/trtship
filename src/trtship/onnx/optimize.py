"""Safe, non-destructive ONNX graph optimization.

Only passes that provably preserve behavior are provided, and each stays correct in the presence
of control-flow subgraphs (``If``/``Loop``/``Scan``), whose bodies may reference tensors of the
enclosing graph by name. Heavier optimization (fusion, layout, precision) is TensorRT's job.

The source model is never modified: the optimized model is written to a new path (which must not
exist) and identified by its own hash. The interface (input/output names, dtypes, shapes) must be
unchanged and the result must pass the ONNX checker. Numerical acceptance against PyTorch is done
by running the validation stage on the optimized model.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Final

import onnx
import onnx.numpy_helper
import onnx.shape_inference
from pydantic import BaseModel, ConfigDict

from trtship.config import OptimizePass
from trtship.errors import ArtifactConflictError, ConfigError, ExportError
from trtship.export import check_onnx_signature
from trtship.logging import get_logger
from trtship.models import ModelSignature
from trtship.onnx.graph import GraphReport, analyze_graph
from trtship.utils.fs import publish_new
from trtship.utils.hashing import sha256_file

log = get_logger(__name__)

RESULT_SCHEMA_VERSION: Final = 1
_STANDARD_DOMAINS: Final = ("", "ai.onnx")
_MIN_IR_FOR_INITIALIZER_OUTPUTS: Final = 4

# Canonical execution order. Selecting a subset keeps this order regardless of how it was listed.
PASS_ORDER: Final[tuple[str, ...]] = (
    "extract_constants",
    "eliminate_identity",
    "deduplicate_initializers",
    "eliminate_dead_nodes",
    "eliminate_unused_initializers",
    "infer_shapes",
)


class PassResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    changes: int


class OptimizeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = RESULT_SCHEMA_VERSION
    source_path: str
    source_sha256: str
    source_size_bytes: int
    path: str
    sha256: str
    size_bytes: int
    passes: list[PassResult]
    # True if a pass that alters the graph (not just its shape metadata) changed something.
    changed: bool
    before: GraphReport
    after: GraphReport


# --------------------------------------------------------------------------- graph helpers


def _subgraphs(node: onnx.NodeProto) -> Iterator[onnx.GraphProto]:
    for attribute in node.attribute:
        if attribute.type == onnx.AttributeProto.GRAPH:
            yield attribute.g
        elif attribute.type == onnx.AttributeProto.GRAPHS:
            yield from attribute.graphs


def _walk_nodes(graph: onnx.GraphProto) -> Iterator[onnx.NodeProto]:
    """Every node of ``graph`` and of all nested subgraphs."""
    for node in graph.node:
        yield node
        for sub in _subgraphs(node):
            yield from _walk_nodes(sub)


def _names_used_inside(node: onnx.NodeProto) -> set[str]:
    """Names a node's subgraphs read; an over-approximation of their free variables."""
    used: set[str] = set()
    for sub in _subgraphs(node):
        for inner in _walk_nodes(sub):
            used.update(n for n in inner.input if n)
        used.update(out.name for out in sub.output)
    return used


def _used_names(graph: onnx.GraphProto) -> set[str]:
    used = {n for node in _walk_nodes(graph) for n in node.input if n}
    used.update(out.name for out in graph.output)
    for node in _walk_nodes(graph):
        for sub in _subgraphs(node):
            used.update(out.name for out in sub.output)
    return used


def _rename_uses(graph: onnx.GraphProto, mapping: dict[str, str], *, top: bool = True) -> None:
    """Rewrite every read of a name in ``mapping`` (including inside subgraphs)."""
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name in mapping:
                node.input[index] = mapping[name]
        for sub in _subgraphs(node):
            _rename_uses(sub, mapping, top=False)
    if not top:  # a nested graph's outputs may name tensors of an enclosing scope
        for out in graph.output:
            if out.name in mapping:
                out.name = mapping[out.name]


def _is_standard(node: onnx.NodeProto) -> bool:
    return node.domain in _STANDARD_DOMAINS


# --------------------------------------------------------------------------- passes


def extract_constants(model: onnx.ModelProto) -> int:
    """Turn ``Constant`` nodes holding a tensor into initializers."""
    graph = model.graph
    if model.ir_version < _MIN_IR_FOR_INITIALIZER_OUTPUTS:
        return 0  # older IR versions require initializers to also be listed as inputs
    protected = {out.name for out in graph.output}
    changed = 0
    for node in list(graph.node):
        if node.op_type != "Constant" or not _is_standard(node) or node.output[0] in protected:
            continue
        attributes = {a.name: a for a in node.attribute}
        value = attributes.get("value")
        if (
            set(attributes) != {"value"}
            or value is None
            or value.type != onnx.AttributeProto.TENSOR
        ):
            continue
        tensor = onnx.TensorProto()
        tensor.CopyFrom(value.t)
        tensor.name = node.output[0]
        graph.initializer.append(tensor)
        graph.node.remove(node)
        changed += 1
    return changed


def eliminate_identity(model: onnx.ModelProto) -> int:
    """Remove ``Identity`` nodes whose output is not a graph output, rewiring their readers."""
    graph = model.graph
    protected = {out.name for out in graph.output}
    changed = 0
    while True:
        target = next(
            (
                node
                for node in graph.node
                if node.op_type == "Identity"
                and _is_standard(node)
                and node.output[0] not in protected
            ),
            None,
        )
        if target is None:
            return changed
        _rename_uses(graph, {target.output[0]: target.input[0]})
        graph.node.remove(target)
        changed += 1


def deduplicate_initializers(model: onnx.ModelProto) -> int:
    """Merge initializers with identical dtype, shape, and contents."""
    graph = model.graph
    fixed = {i.name for i in graph.input} | {o.name for o in graph.output}
    seen: dict[tuple[int, tuple[int, ...], bytes], str] = {}
    mapping: dict[str, str] = {}
    for init in graph.initializer:
        if init.name in fixed or init.data_location == onnx.TensorProto.EXTERNAL:
            continue
        try:
            payload = onnx.numpy_helper.to_array(init).tobytes()
        except Exception as exc:  # element type numpy cannot represent: leave it alone
            log.debug("not deduplicating initializer %s: %s", init.name, exc)
            continue
        key = (init.data_type, tuple(init.dims), payload)
        if key in seen:
            mapping[init.name] = seen[key]
        else:
            seen[key] = init.name
    if not mapping:
        return 0
    _rename_uses(graph, mapping)
    kept = [i for i in graph.initializer if i.name not in mapping]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    return len(mapping)


def eliminate_dead_nodes(model: onnx.ModelProto) -> int:
    """Remove nodes that do not contribute to any graph output."""
    graph = model.graph
    needed = {out.name for out in graph.output}
    keep: list[bool] = [False] * len(graph.node)
    for index in range(len(graph.node) - 1, -1, -1):
        node = graph.node[index]
        if any(name in needed for name in node.output):
            keep[index] = True
            needed.update(n for n in node.input if n)
            needed.update(_names_used_inside(node))
    dead = keep.count(False)
    if dead:
        survivors = [node for node, alive in zip(graph.node, keep, strict=True) if alive]
        copies = []
        for node in survivors:
            clone = onnx.NodeProto()
            clone.CopyFrom(node)
            copies.append(clone)
        del graph.node[:]
        graph.node.extend(copies)
    return dead


def eliminate_unused_initializers(model: onnx.ModelProto) -> int:
    """Remove initializers that nothing reads (graph inputs and outputs are kept)."""
    graph = model.graph
    used = _used_names(graph) | {i.name for i in graph.input}
    unused = [init for init in graph.initializer if init.name not in used]
    if unused:
        kept = [init for init in graph.initializer if init.name in used]
        del graph.initializer[:]
        graph.initializer.extend(kept)
    return len(unused)


def infer_shapes(model: onnx.ModelProto) -> int:
    """Populate ``value_info`` for intermediate tensors. Counts tensors newly described."""
    before = len(model.graph.value_info)
    inferred = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=False)
    del model.graph.value_info[:]
    model.graph.value_info.extend(inferred.graph.value_info)
    return max(len(model.graph.value_info) - before, 0)


_PASSES: Final[dict[str, Callable[[onnx.ModelProto], int]]] = {
    "extract_constants": extract_constants,
    "eliminate_identity": eliminate_identity,
    "deduplicate_initializers": deduplicate_initializers,
    "eliminate_dead_nodes": eliminate_dead_nodes,
    "eliminate_unused_initializers": eliminate_unused_initializers,
    "infer_shapes": infer_shapes,
}


def _select(passes: Sequence[OptimizePass] | Sequence[str] | None) -> list[str]:
    if passes is None:
        return list(PASS_ORDER)
    unknown = [p for p in passes if p not in _PASSES]
    if unknown:
        raise ConfigError(
            f"unknown optimization pass(es): {', '.join(unknown)}",
            hint=f"Available passes: {', '.join(PASS_ORDER)}.",
        )
    return [name for name in PASS_ORDER if name in passes]


# --------------------------------------------------------------------------- entry point


def optimize_onnx(
    source: Path,
    dest: Path,
    signature: ModelSignature,
    passes: Sequence[OptimizePass] | Sequence[str] | None = None,
) -> OptimizeResult:
    """Write an optimized copy of ``source`` to ``dest`` (which must not exist).

    Raises :class:`ExportError` if an optimization changed the model's interface (a bug in a
    pass, never silently accepted) and :class:`ValidationFailedError` if the result fails the ONNX
    checker.
    """
    selected = _select(passes)
    if dest.exists():
        raise ArtifactConflictError(
            f"refusing to overwrite existing artifact: {dest}",
            hint="Artifacts are immutable. Choose a new output path.",
        )
    if dest.resolve() == source.resolve():
        raise ArtifactConflictError("the optimized model must not overwrite its source")
    source_sha = sha256_file(source)
    before = analyze_graph(source)
    proto = onnx.load(str(source))

    results = [PassResult(name=name, changes=_PASSES[name](proto)) for name in selected]
    log.info("optimization passes: %s", {r.name: r.changes for r in results})

    props = {p.key: p.value for p in proto.metadata_props}
    props["trtship.optimized_from_sha256"] = source_sha
    props["trtship.optimize_passes"] = ",".join(selected)
    onnx.helper.set_model_props(proto, props)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{secrets.token_hex(4)}.tmp")
    try:
        onnx.save(proto, str(tmp))
        problems = check_onnx_signature(onnx.load(str(tmp)), signature)
        if problems:
            raise ExportError(
                "optimization changed the model's interface",
                hint="This is a bug in an optimization pass; the source model is untouched.",
                details={"problems": problems, "passes": selected},
            )
        after = analyze_graph(tmp)
        sha, size = sha256_file(tmp), tmp.stat().st_size
        publish_new(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return OptimizeResult(
        source_path=str(source),
        source_sha256=source_sha,
        source_size_bytes=source.stat().st_size,
        path=str(dest),
        sha256=sha,
        size_bytes=size,
        passes=results,
        changed=any(r.changes for r in results if r.name != "infer_shapes"),
        before=before,
        after=after,
    )


__all__ = [
    "PASS_ORDER",
    "OptimizeResult",
    "PassResult",
    "deduplicate_initializers",
    "eliminate_dead_nodes",
    "eliminate_identity",
    "eliminate_unused_initializers",
    "extract_constants",
    "infer_shapes",
    "optimize_onnx",
]
