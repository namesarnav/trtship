"""`trtship report`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from trtship.artifacts import RunDirectory
from trtship.cli.render import console, emit_json
from trtship.reporting import render_run_summary, summarize_run


def report_command(
    run_dir: Annotated[
        Path | None, typer.Argument(metavar="RUN", help="Run directory (default: the latest run).")
    ] = None,
    root: Annotated[Path, typer.Option(help="Where to look for the latest run.")] = Path("runs"),
    json_output: Annotated[bool, typer.Option("--json", help="Print the summary as JSON.")] = False,
) -> None:
    """Summarize a run: stages, artifacts, validation results, and integrity."""
    run = RunDirectory.open(run_dir) if run_dir is not None else RunDirectory.latest(root)
    summary = summarize_run(run)
    if json_output:
        emit_json(summary.model_dump(mode="json"))
    else:
        render_run_summary(summary, console)
