"""Human-readable rendering of an :class:`OnnxValidationReport`."""

from __future__ import annotations

import io

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from trtship.onnx.validate import OnnxValidationReport
from trtship.utils.units import format_bytes


def _sci(value: float | None) -> str:
    return "-" if value is None else f"{value:.2e}"


def _fixed(value: float | None, digits: int = 6) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render_onnx_validation(report: OnnxValidationReport, console: Console) -> None:
    verdict = "[bold green]PASSED[/]" if report.passed else "[bold red]FAILED[/]"
    console.print(f"ONNX validation {verdict}  {escape(report.onnx_path)}")
    g = report.graph
    top_ops = ", ".join(f"{op} x{n}" for op, n in list(g.op_counts.items())[:6])
    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column(style="bold")
    summary.add_column(overflow="fold")
    for key, value in (
        ("model", f"{report.model_name}  (weights {report.weights_sha256[:12]})"),
        ("onnx sha256", report.onnx_sha256),
        ("size", f"{format_bytes(report.onnx_size_bytes)} ({g.initializer_count} initializers)"),
        ("graph", f"opset {g.opset}, IR {g.ir_version}, {g.node_count} nodes"),
        ("ops", top_ops or "-"),
        ("runtime", f"onnxruntime {report.ort_version} {report.providers}"),
        ("seed / time", f"{report.seed} / {report.duration_s}s"),
        (
            "tolerance",
            f"atol={report.tolerance.atol:g} rtol={report.tolerance.rtol:g} "
            f"cosine>={report.tolerance.cosine_min:g}",
        ),
    ):
        summary.add_row(key, escape(value))
    console.print(summary)
    for warning in g.warnings:
        console.print(f"[yellow]warning:[/] {escape(warning)}")
    console.print()

    table = Table(title="PyTorch vs ONNX Runtime", title_justify="left")
    for column in (
        "Shape point",
        "Output",
        "Shape",
        "Max abs",
        "Mean abs",
        "Max rel",
        "Cosine",
        "Top-1",
    ):
        table.add_column(column)
    table.add_column("Result")
    for point in report.points:
        label = f"{point.label} {escape(str(point.sizes))}"
        if point.error:
            table.add_row(label, "-", "-", "-", "-", "-", "-", "-", "[red]error[/]")
            continue
        for index, out in enumerate(point.outputs):
            cosine = (
                out.cosine_min_per_sample
                if out.cosine_min_per_sample is not None
                else out.cosine_similarity
            )
            table.add_row(
                label if index == 0 else "",
                out.name,
                escape(str(out.candidate_shape)),
                _sci(out.max_abs_error),
                _sci(out.mean_abs_error),
                _sci(out.max_rel_error),
                _fixed(cosine),
                _fixed(out.top1_agreement, 4),
                "[green]pass[/]" if out.passed else "[red]FAIL[/]",
            )
    console.print(table)
    if report.failures:
        console.print("\n[bold red]Failures[/]")
        for failure in report.failures:
            console.print(f"  - {escape(failure)}", highlight=False)


def render_onnx_validation_text(report: OnnxValidationReport, *, width: int = 120) -> str:
    buffer = io.StringIO()
    render_onnx_validation(
        report, Console(file=buffer, width=width, color_system=None, legacy_windows=False)
    )
    return buffer.getvalue()
