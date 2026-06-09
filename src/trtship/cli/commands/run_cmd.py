"""`trtship run`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from trtship.artifacts import RunDirectory, StageStatus
from trtship.cli.render import console, emit_json, err_console
from trtship.config import TrtshipConfig, load_config
from trtship.errors import ConfigError, TrtshipError
from trtship.logging import attach_log_file, detach_log_file
from trtship.pipeline import (
    Pipeline,
    StageOutcome,
    order_stages,
    plan_stages,
    preflight_stages,
    select_stages,
)
from trtship.pipeline.stages import default_stages
from trtship.reporting import render_run_summary, summarize_run
from trtship.utils import env

_STATUS_LABEL = {
    StageStatus.SUCCEEDED: "[green]done[/]",
    StageStatus.CACHED: "[cyan]cached[/]",
    StageStatus.SKIPPED: "[dim]skipped[/]",
}


def _print_progress(outcome: StageOutcome) -> None:
    label = _STATUS_LABEL.get(outcome.status, outcome.status.value)
    detail = outcome.reason or ", ".join(f"{k}={v}" for k, v in list(outcome.metrics.items())[:3])
    console.print(f"  {outcome.name:<16} {label:<18} {outcome.duration_s:>7.2f}s  {detail}")
    for warning in outcome.warnings[:3]:
        console.print(f"    [yellow]warning:[/] {warning}", highlight=False, markup=True)


def _check_resumable(config: TrtshipConfig, run: RunDirectory) -> None:
    """A run may only be continued with the configuration it was created with."""
    snapshot = load_config(run.config_path)
    differing = sorted(
        section
        for section in TrtshipConfig.model_fields
        if section != "artifacts" and getattr(snapshot, section) != getattr(config, section)
    )
    if differing:
        raise ConfigError(
            f"the configuration differs from run {run.run_id}'s snapshot "
            f"in: {', '.join(differing)}",
            hint=f"Start a new run, or pass the snapshot itself: {run.config_path}",
        )


def run_command(
    config_path: Annotated[Path, typer.Argument(metavar="CONFIG", help="trtship YAML config.")],
    run_dir: Annotated[
        Path | None, typer.Option("--run", help="Continue an existing run directory.")
    ] = None,
    run_id: Annotated[
        str | None, typer.Option(help="Id for a new run (default: date + token).")
    ] = None,
    from_stage: Annotated[
        str | None, typer.Option("--from", help="Start at this stage (needs --run).")
    ] = None,
    only: Annotated[str | None, typer.Option(help="Run just this stage (needs --run).")] = None,
    until: Annotated[str | None, typer.Option(help="Stop after this stage.")] = None,
    dry_run: Annotated[bool, typer.Option(help="Show the plan without running anything.")] = False,
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="Override a value: section.key=value.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
) -> None:
    """Run the pipeline. Stages that are already up to date in a run are skipped."""
    config = load_config(config_path, overrides=overrides or [])
    stages = default_stages()
    ordered = order_stages(stages)
    target = only or from_stage
    if target is not None and run_dir is None and target != ordered[0].name:
        raise ConfigError(
            f"--from/--only {target} needs artifacts from earlier stages",
            hint="Pass --run RUN_DIR to continue an existing run.",
        )
    plan = plan_stages(stages, config, from_stage=from_stage, only=only, until=until)
    if dry_run:
        table = Table(title="Plan", title_justify="left")
        for column in ("Stage", "Action", "Reason"):
            table.add_column(column)
        for item in plan:
            table.add_row(item.name, item.action, item.reason or "")
        console.print(table)
        return

    environment = env.probe_all(repo_dir=config_path.resolve().parent)
    # Check capabilities before creating a run directory, so an impossible request leaves no litter.
    selected = select_stages(ordered, from_stage=from_stage, only=only, until=until)
    preflight_stages(selected, config, environment)
    if run_dir is not None:
        run = RunDirectory.open(run_dir)
        _check_resumable(config, run)
    else:
        run = RunDirectory.create(config.artifacts.root, config, environment, run_id=run_id)
    if not json_output:
        console.print(f"run {run.run_id}  ({run.path})")

    handler = attach_log_file(run.logs_dir / "trtship.jsonl")
    pipeline = Pipeline(
        config, run, stages, environment, progress=None if json_output else _print_progress
    )
    try:
        result = pipeline.execute(from_stage=from_stage, only=only, until=until)
    except TrtshipError:
        err_console.print(f"[bold]run {run.run_id} did not complete[/]; details: {run.path}")
        err_console.print(f"resume with: trtship run {config_path} --run {run.path}")
        raise
    finally:
        detach_log_file(handler)

    if json_output:
        emit_json(result.model_dump(mode="json"))
    else:
        console.print()
        render_run_summary(summarize_run(run), console)
