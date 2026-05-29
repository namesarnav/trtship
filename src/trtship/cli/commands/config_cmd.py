"""`trtship config ...`."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from trtship.cli.guard import handle_errors
from trtship.cli.render import console, emit_json
from trtship.config import TrtshipConfig, config_hash, config_schema_json, load_config
from trtship.errors import ConfigError

config_app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)


@config_app.command("validate")
@handle_errors
def validate(
    path: Annotated[Path, typer.Argument(help="Path to a trtship YAML config.")],
    overrides: Annotated[
        list[str] | None,
        typer.Option("--set", help="Override a value: section.key=value (repeatable)."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Validate a config file. Exits 2 when it is invalid."""
    try:
        config = load_config(path, overrides=overrides or [])
    except ConfigError as exc:
        if json_output:
            emit_json({"valid": False, "path": str(path), **exc.to_dict()})
            raise typer.Exit(exc.exit_code) from exc
        raise
    warnings = config.warnings()
    if json_output:
        emit_json(
            {
                "valid": True,
                "path": str(path),
                "config_sha256": config_hash(config),
                "warnings": warnings,
            }
        )
        return
    _render_summary(path, config)
    for warning in warnings:
        console.print(f"[yellow]warning:[/] {escape(warning)}", highlight=False)


def _render_summary(path: Path, config: TrtshipConfig) -> None:
    console.print(f"[green]valid[/] {escape(str(path))}")
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold")
    table.add_column(overflow="fold")
    table.add_row("model", escape(f"{config.model.name} ({config.model.kind.value})"))
    for spec in config.model.inputs:
        table.add_row("  input", escape(f"{spec.name}: {spec.dtype.value} {list(spec.shape)}"))
    table.add_row("precisions", ", ".join(p.value for p in config.tensorrt.precisions))
    table.add_row("opset", str(config.export.opset))
    table.add_row("run root", str(config.artifacts.root))
    table.add_row("config sha256", config_hash(config))
    console.print(table)


@config_app.command("schema")
@handle_errors
def schema(
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the schema here instead.")
    ] = None,
) -> None:
    """Print the JSON Schema of the configuration (useful for editor validation)."""
    text = config_schema_json()
    if output is None:
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        console.print(f"wrote {output}")
