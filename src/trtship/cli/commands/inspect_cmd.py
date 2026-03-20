"""`trtship inspect`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from trtship.cli.render import console, emit_json
from trtship.config import load_config
from trtship.errors import ArtifactError
from trtship.models import load_model
from trtship.models.inspection import inspect_model
from trtship.reporting import render_model_report
from trtship.utils.fs import atomic_write_json


def inspect_command(
    config_path: Annotated[Path, typer.Argument(metavar="CONFIG", help="trtship YAML config.")],
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="Override a value: section.key=value.")
    ] = None,
    depth: Annotated[int, typer.Option(min=0, help="Module-tree depth to show.")] = 3,
    top: Annotated[int, typer.Option(min=1, help="Number of largest tensors to list.")] = 10,
    json_output: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Also write the JSON report to this file.")
    ] = None,
    force: Annotated[bool, typer.Option(help="Overwrite --output if it exists.")] = False,
) -> None:
    """Load the model and report its architecture, parameters, memory, and I/O signature."""
    config = load_config(config_path, overrides=overrides or [])
    if output is not None and output.exists() and not force:
        raise ArtifactError(
            f"refusing to overwrite existing file: {output}", hint="Pass --force to overwrite it."
        )
    model = load_model(config.model)
    report = inspect_model(
        model,
        config.model,
        config.tensorrt.profiles,
        seed=config.seed,
        max_depth=depth,
        top=top,
    )
    payload = report.model_dump(mode="json")
    if output is not None:
        atomic_write_json(output, payload)
    if json_output:
        emit_json(payload)
    else:
        render_model_report(report, console)
        if output is not None:
            console.print(f"\nwrote {output}")
