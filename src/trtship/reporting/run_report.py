"""Summary of a run directory: what ran, what it produced, and how it went."""

from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from trtship.artifacts import (
    ArtifactStore,
    ArtifactType,
    RunDirectory,
    RunStatus,
    StageRecord,
    StageStatus,
)
from trtship.utils.env import EnvironmentReport
from trtship.utils.units import format_bytes

SUMMARY_SCHEMA_VERSION = 1


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StageSummary(_Frozen):
    name: str
    status: StageStatus
    duration_s: float | None
    artifacts: list[str]
    metrics: dict[str, Any]
    error: dict[str, Any] | None = None


class ArtifactSummary(_Frozen):
    id: str
    type: ArtifactType
    path: str
    size_bytes: int
    stage: str
    sha256: str


class ValidationSummary(_Frozen):
    kind: str
    artifact: str
    passed: bool
    failures: list[str] = Field(default_factory=list)


class RunSummary(_Frozen):
    schema_version: int = SUMMARY_SCHEMA_VERSION
    run_id: str
    path: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    trtship_version: str
    config_sha256: str
    model_sha256: str | None
    seed: int
    git_commit: str | None
    git_dirty: bool | None
    versions: dict[str, str | None]
    gpus: list[str]
    stages: list[StageSummary]
    artifacts: list[ArtifactSummary]
    validations: list[ValidationSummary]
    integrity_problems: list[str]


def summarize_run(run: RunDirectory) -> RunSummary:
    manifest = run.read_manifest()
    store = ArtifactStore(run)
    environment = EnvironmentReport.model_validate_json(run.environment_path.read_text("utf-8"))

    def ran_at(item: tuple[str, StageRecord]) -> datetime:
        record = item[1]
        return record.started_at or record.finished_at or manifest.created_at

    stages = []
    # The manifest is stored with sorted keys, so recover execution order from the timestamps.
    for name, stage_record in sorted(manifest.stages.items(), key=ran_at):
        duration = (
            round((stage_record.finished_at - stage_record.started_at).total_seconds(), 3)
            if stage_record.started_at and stage_record.finished_at
            else None
        )
        stages.append(
            StageSummary(
                name=name,
                status=stage_record.status,
                duration_s=duration,
                artifacts=stage_record.artifact_ids,
                metrics=stage_record.metrics,
                error=stage_record.error,
            )
        )

    validations = []
    for report_record in store.records(ArtifactType.VALIDATION_REPORT):
        try:
            report = json.loads(store.absolute(report_record).read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        validations.append(
            ValidationSummary(
                kind=str(report_record.metadata.get("kind", "unknown")),
                artifact=report_record.id,
                passed=bool(report.get("passed", False)),
                failures=list(report.get("failures", []))[:5],
            )
        )

    return RunSummary(
        run_id=manifest.run_id,
        path=str(run.path),
        status=manifest.status,
        created_at=manifest.created_at,
        updated_at=manifest.updated_at,
        trtship_version=manifest.trtship_version,
        config_sha256=manifest.config_sha256,
        model_sha256=manifest.model_sha256,
        seed=manifest.seed,
        git_commit=environment.git.commit if environment.git else None,
        git_dirty=environment.git.dirty if environment.git else None,
        versions=environment.versions(),
        gpus=[
            f"{g.name} (sm_{(g.compute_capability or '?').replace('.', '')})"
            for g in environment.gpus
        ],
        stages=stages,
        artifacts=[
            ArtifactSummary(
                id=r.id,
                type=r.type,
                path=r.path,
                size_bytes=r.size_bytes,
                stage=r.stage,
                sha256=r.sha256,
            )
            for r in store.records()
        ],
        validations=validations,
        integrity_problems=store.verify_all(),
    )


_STATUS_STYLE = {
    StageStatus.SUCCEEDED: "[green]succeeded[/]",
    StageStatus.CACHED: "[cyan]cached[/]",
    StageStatus.SKIPPED: "[dim]skipped[/]",
    StageStatus.FAILED: "[bold red]failed[/]",
    StageStatus.RUNNING: "[yellow]running[/]",
    StageStatus.PENDING: "[dim]pending[/]",
}


def _render_header(summary: RunSummary, console: Console) -> None:
    status = {
        RunStatus.SUCCEEDED: "[bold green]succeeded[/]",
        RunStatus.FAILED: "[bold red]failed[/]",
    }.get(summary.status, summary.status.value)
    console.print(f"Run [bold]{escape(summary.run_id)}[/]  {status}")
    info = Table(show_header=False, box=None, pad_edge=False)
    info.add_column(style="bold")
    info.add_column(overflow="fold")
    commit = (summary.git_commit or "-")[:12] + (" (dirty)" if summary.git_dirty else "")
    versions = ", ".join(f"{k} {v}" for k, v in summary.versions.items() if k in _VERSIONED)
    for key, value in (
        ("path", summary.path),
        ("created", summary.created_at.isoformat(timespec="seconds")),
        ("trtship", summary.trtship_version),
        ("git", commit),
        ("model sha256", summary.model_sha256 or "-"),
        ("config sha256", summary.config_sha256),
        ("seed", str(summary.seed)),
        ("gpus", ", ".join(summary.gpus) or "none detected"),
        ("versions", versions),
    ):
        info.add_row(key, escape(value))
    console.print(info)
    console.print()


def _render_stages(summary: RunSummary, console: Console) -> None:
    table = Table(title="Stages", title_justify="left")
    for column in ("Stage", "Status", "Time", "Detail"):
        table.add_column(column, overflow="fold")
    for stage in summary.stages:
        detail = ""
        if stage.error:
            detail = f"{stage.error.get('error')}: {stage.error.get('message')}"
        elif stage.metrics:
            detail = ", ".join(f"{k}={v}" for k, v in list(stage.metrics.items())[:4])
        seconds = "-" if stage.duration_s is None else f"{stage.duration_s:.2f}s"
        table.add_row(stage.name, _STATUS_STYLE[stage.status], seconds, escape(detail))
    console.print(table)


def _render_validations(summary: RunSummary, console: Console) -> None:
    console.print()
    for validation in summary.validations:
        mark = "[green]passed[/]" if validation.passed else "[bold red]FAILED[/]"
        console.print(f"validation {validation.kind}: {mark}")
        for failure in validation.failures:
            console.print(f"  - {escape(failure)}", highlight=False)


def _render_artifacts(summary: RunSummary, console: Console) -> None:
    console.print()
    table = Table(title="Artifacts", title_justify="left")
    for column in ("Type", "Path", "Size", "Stage", "Id"):
        table.add_column(column, overflow="fold")
    for item in summary.artifacts:
        table.add_row(
            item.type.value, item.path, format_bytes(item.size_bytes), item.stage, item.id
        )
    console.print(table)


def render_run_summary(summary: RunSummary, console: Console) -> None:
    _render_header(summary, console)
    _render_stages(summary, console)
    if summary.validations:
        _render_validations(summary, console)
    if summary.artifacts:
        _render_artifacts(summary, console)
    if summary.integrity_problems:
        console.print("\n[bold red]Integrity problems[/]")
        for problem in summary.integrity_problems:
            console.print(f"  - {escape(problem)}", highlight=False)


_VERSIONED = frozenset(
    {
        "python",
        "torch",
        "onnx",
        "onnxruntime",
        "tensorrt",
        "tritonclient",
        "cuda_toolkit",
        "nvidia_gpu",
    }
)


def render_run_summary_text(summary: RunSummary, *, width: int = 120) -> str:
    buffer = io.StringIO()
    render_run_summary(
        summary, Console(file=buffer, width=width, color_system=None, legacy_windows=False)
    )
    return buffer.getvalue()
