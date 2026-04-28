"""`trtship optimize`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from trtship.cli.render import console, emit_json
from trtship.config import load_config
from trtship.models import infer_signature, load_model
from trtship.onnx import OptimizeResult, optimize_onnx, validate_onnx
from trtship.reporting.validation_report import render_onnx_validation
from trtship.utils.units import format_bytes


def _render(result: OptimizeResult) -> None:
    console.print(f"[green]optimized[/] {result.path}")
    table = Table(show_header=True, box=None, pad_edge=False)
    for column in ("", "source", "optimized"):
        table.add_column(column, justify="right" if column else "left")
    table.add_row("size", format_bytes(result.source_size_bytes), format_bytes(result.size_bytes))
    table.add_row("nodes", str(result.before.node_count), str(result.after.node_count))
    table.add_row(
        "initializers",
        f"{result.before.initializer_count} ({format_bytes(result.before.initializer_bytes)})",
        f"{result.after.initializer_count} ({format_bytes(result.after.initializer_bytes)})",
    )
    console.print(table)
    console.print("passes:")
    for item in result.passes:
        console.print(f"  {item.name:32} {item.changes} change(s)")
    if not result.changed:
        console.print("[dim]no pass changed the graph[/]")
    console.print(f"sha256 {result.sha256}")


def optimize_command(
    config_path: Annotated[Path, typer.Argument(metavar="CONFIG", help="trtship YAML config.")],
    source: Annotated[Path, typer.Argument(metavar="MODEL.onnx", help="ONNX model to optimize.")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to write the optimized model.")
    ],
    passes: Annotated[
        list[str] | None,
        typer.Option("--pass", help="Run only this pass (repeatable). Default: config or all."),
    ] = None,
    validate: Annotated[
        bool, typer.Option(help="Validate the optimized model against PyTorch.")
    ] = True,
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="Override a value: section.key=value.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print results as JSON.")] = False,
) -> None:
    """Write an optimized copy of an ONNX model. The source file is never modified."""
    config = load_config(config_path, overrides=overrides or [])
    model = load_model(config.model)
    signature = infer_signature(model, config.model, config.tensorrt.profiles, seed=config.seed)
    selected = passes if passes else config.optimize.passes
    result = optimize_onnx(source, output, signature, selected)
    report = validate_onnx(output, model, signature, config) if validate else None

    if json_output:
        emit_json(
            {
                "optimize": result.model_dump(mode="json"),
                "validation": report.model_dump(mode="json") if report else None,
            }
        )
    else:
        _render(result)
        if report is not None:
            console.print()
            render_onnx_validation(report, console)
    if report is not None:
        report.raise_for_failure()
