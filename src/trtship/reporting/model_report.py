"""Human-readable rendering of a :class:`ModelReport`."""

from __future__ import annotations

import io

from rich.console import Console, Group
from rich.markup import escape
from rich.table import Table
from rich.tree import Tree

from trtship.models.inspection import ModelReport, ModuleNode
from trtship.specs import TensorSpec
from trtship.utils.units import format_bytes, format_count


def _shape(spec: TensorSpec) -> str:
    return "[" + ", ".join(str(d) for d in spec.shape) + "]"


def _node_label(node: ModuleNode) -> str:
    label = f"[bold]{escape(node.name.rsplit('.', 1)[-1])}[/] ({escape(node.type)})"
    if node.parameters:
        label += f"  {format_count(node.parameters)} params"
        if node.trainable_parameters != node.parameters:
            label += f", {format_count(node.trainable_parameters)} trainable"
    if node.elided_modules:
        label += f"  [dim]+{node.elided_modules} nested modules[/]"
    return label


def _add_children(tree: Tree, node: ModuleNode) -> None:
    for child in node.children:
        _add_children(tree.add(_node_label(child)), child)


def _count_text(value: int) -> str:
    humanized = format_count(value)
    return f"{value:,}" if humanized == str(value) else f"{value:,} ({humanized})"


def _kv(rows: list[tuple[str, str]], title: str) -> Table:
    table = Table(title=title, title_justify="left", show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold")
    table.add_column(overflow="fold")
    for key, value in rows:
        table.add_row(key, escape(value))
    return table


def render_model_report(report: ModelReport, console: Console) -> None:
    p, m = report.parameters, report.memory
    console.print(
        _kv(
            [
                ("name", f"{report.name} ({report.kind})"),
                ("root module", report.root_type),
                ("modules", str(report.module_count)),
                ("device", report.device),
                ("torch", report.torch_version),
                ("weights sha256", report.weights_sha256),
                ("source", report.source_path or "-"),
            ],
            "Model",
        )
    )
    console.print()
    dtypes = ", ".join(f"{d}: {format_count(n)}" for d, n in p.by_dtype.items()) or "-"
    console.print(
        _kv(
            [
                ("total", _count_text(p.total)),
                ("trainable", f"{p.trainable:,}"),
                ("frozen", f"{p.frozen:,}"),
                ("parameter tensors", str(p.tensors)),
                ("by dtype", dtypes),
                ("buffers", f"{p.buffer_tensors} tensors, {p.buffer_elements:,} elements"),
            ],
            "Parameters",
        )
    )
    console.print()
    activation = (
        f"<= {format_bytes(m.activation_bytes_upper_bound)} at {m.activation_probe_sizes}"
        if m.activation_bytes_upper_bound is not None
        else "not measured"
    )
    rows = [
        (
            "weights",
            f"{format_bytes(m.weights_bytes)} (parameters {format_bytes(m.parameter_bytes)}, "
            f"buffers {format_bytes(m.buffer_bytes)})",
        ),
        ("activations", activation),
    ]
    console.print(_kv(rows, "Memory estimate"))
    for note in m.notes:
        console.print(f"[yellow]note:[/] {escape(note)}")
    console.print()

    io_table = Table(title="Signature", title_justify="left")
    for column in ("Direction", "Name", "Dtype", "Shape"):
        io_table.add_column(column)
    for spec in report.signature.inputs:
        io_table.add_row("input", spec.name, spec.dtype.value, escape(_shape(spec)))
    for spec in report.signature.outputs:
        io_table.add_row("output", spec.name, spec.dtype.value, escape(_shape(spec)))
    console.print(io_table)
    console.print()

    root = Tree(_node_label(report.architecture))
    _add_children(root, report.architecture)
    largest = Table(title="Largest parameter tensors", title_justify="left")
    for column in ("Name", "Shape", "Dtype", "Elements", "Size"):
        largest.add_column(column)
    for tensor in p.largest:
        largest.add_row(
            tensor.name,
            escape(str(tensor.shape)),
            tensor.dtype,
            f"{tensor.numel:,}",
            format_bytes(tensor.bytes),
        )
    console.print(Group("[bold]Architecture[/]", root))
    console.print()
    console.print(largest)


def render_model_report_text(report: ModelReport, *, width: int = 110) -> str:
    """Plain text (no ANSI) rendering, suitable for saving next to the JSON report."""
    buffer = io.StringIO()
    render_model_report(
        report, Console(file=buffer, width=width, color_system=None, legacy_windows=False)
    )
    return buffer.getvalue()
