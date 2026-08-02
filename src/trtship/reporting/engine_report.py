"""Human-readable rendering of an :class:`EngineValidationReport`."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from trtship.validation.engine import EngineValidationReport


def _sci(value: float | None) -> str:
    return "-" if value is None else f"{value:.2e}"


def _fixed(value: float | None, digits: int = 5) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render_engine_validation(report: EngineValidationReport, console: Console) -> None:
    verdict = "[bold green]PASSED[/]" if report.passed else "[bold red]FAILED[/]"
    via = "" if report.backend == "tensorrt" else f" via {report.backend}"
    console.print(f"Engine validation{via} {verdict}  model {escape(report.model_name)}")
    console.print(
        f"reference: PyTorch; also compared with onnxruntime {report.ort_version} on the CPU; "
        f"seed {report.seed}; {report.duration_s}s"
    )
    for result in report.results:
        tol = result.tolerance
        console.print(
            f"\n[bold]{result.precision.value}[/]  {escape(result.engine_path)}\n"
            f"  tolerance vs PyTorch: atol={tol.atol:g} rtol={tol.rtol:g} "
            f"cosine>={tol.cosine_min:g}"
            + (f" top1>={tol.top1_agreement_min:g}" if tol.top1_agreement_min is not None else "")
        )
        table = Table(title_justify="left")
        for column in (
            "Shape point",
            "Output",
            "Max abs",
            "Max rel",
            "Cosine",
            "Top-1",
            "vs ONNX max abs",
            "Result",
        ):
            table.add_column(column)
        for point in result.points:
            label = f"{point.label} {escape(str(point.sizes))}"
            if point.error:
                table.add_row(label, "-", "-", "-", "-", "-", "-", "[red]error[/]")
                continue
            onnx_by_name = {c.name: c for c in point.vs_onnx}
            for index, comparison in enumerate(point.vs_pytorch):
                cosine = comparison.cosine_min_per_sample or comparison.cosine_similarity
                other = onnx_by_name.get(comparison.name)
                table.add_row(
                    label if index == 0 else "",
                    comparison.name,
                    _sci(comparison.max_abs_error),
                    _sci(comparison.max_rel_error),
                    _fixed(cosine, 6),
                    _fixed(comparison.top1_agreement, 4),
                    _sci(other.max_abs_error) if other else "-",
                    "[green]pass[/]" if comparison.passed else "[red]FAIL[/]",
                )
        console.print(table)
    if report.failures:
        console.print("\n[bold red]Failures[/]")
        for failure in report.failures:
            console.print(f"  - {escape(failure)}", highlight=False)
