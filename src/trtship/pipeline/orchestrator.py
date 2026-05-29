"""Pipeline orchestration: ordering, selection, preflight, caching, execution, and recovery."""

from __future__ import annotations

import secrets
import shutil
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from trtship.artifacts import (
    ArtifactCache,
    ArtifactStore,
    CachedArtifact,
    RunDirectory,
    RunStatus,
    StageStatus,
)
from trtship.config import TrtshipConfig
from trtship.errors import ConfigError, EnvironmentUnavailableError, TrtshipError
from trtship.logging import get_logger, log_context
from trtship.pipeline.stage import Resources, Stage, StageContext
from trtship.utils.env import EnvironmentReport
from trtship.utils.timeutil import utc_now

log = get_logger(__name__)


class PlannedStage(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    action: Literal["run", "skip"]
    reason: str | None = None


class StageOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    status: StageStatus
    duration_s: float
    artifacts: list[str] = Field(default_factory=list)
    reason: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    cache_key: str | None = None


class PipelineResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    run_path: str
    status: RunStatus
    stages: list[StageOutcome]


def order_stages(stages: Sequence[Stage]) -> list[Stage]:
    """Topologically order stages by the artifacts they produce and consume.

    A stage runs after every other stage that produces something it requires or uses. Ties keep the
    order of ``stages``.
    """
    names = [s.name for s in stages]
    if len(set(names)) != len(names):
        raise ConfigError(
            f"duplicate stage names: {sorted(n for n in names if names.count(n) > 1)}"
        )
    after: dict[str, set[str]] = {s.name: set() for s in stages}
    for consumer in stages:
        for artifact_type in (*consumer.requires, *consumer.uses):
            after[consumer.name].update(
                p.name for p in stages if p is not consumer and artifact_type in p.produces
            )
    ordered: list[Stage] = []
    done: set[str] = set()
    remaining = list(stages)
    while remaining:
        ready = next((s for s in remaining if after[s.name] <= done), None)
        if ready is None:
            cycle = [s.name for s in remaining]
            raise ConfigError(f"stage dependencies form a cycle among {cycle}")
        ordered.append(ready)
        done.add(ready.name)
        remaining.remove(ready)
    return ordered


def select_stages(
    ordered: Sequence[Stage],
    *,
    from_stage: str | None = None,
    only: str | None = None,
    until: str | None = None,
) -> list[Stage]:
    names = [s.name for s in ordered]
    for label, value in (("--from", from_stage), ("--only", only), ("--until", until)):
        if value is not None and value not in names:
            raise ConfigError(
                f"unknown stage {value!r} for {label}", hint=f"Stages: {', '.join(names)}."
            )
    if only is not None:
        if from_stage is not None or until is not None:
            raise ConfigError("--only cannot be combined with --from or --until")
        return [ordered[names.index(only)]]
    start = names.index(from_stage) if from_stage is not None else 0
    end = names.index(until) if until is not None else len(names) - 1
    if start > end:
        raise ConfigError(f"--from {from_stage} comes after --until {until} in the pipeline order")
    return list(ordered[start : end + 1])


def plan_stages(
    stages: Sequence[Stage],
    config: TrtshipConfig,
    *,
    from_stage: str | None = None,
    only: str | None = None,
    until: str | None = None,
) -> list[PlannedStage]:
    """What a run would do, without creating one."""
    selected = select_stages(order_stages(stages), from_stage=from_stage, only=only, until=until)
    plan = []
    for stage in selected:
        reason = stage.skip_reason(config)
        plan.append(
            PlannedStage(name=stage.name, action="skip" if reason else "run", reason=reason)
        )
    return plan


class Pipeline:
    def __init__(
        self,
        config: TrtshipConfig,
        run: RunDirectory,
        stages: Sequence[Stage],
        environment: EnvironmentReport,
        *,
        progress: Callable[[StageOutcome], None] | None = None,
    ) -> None:
        self.config = config
        self.run = run
        self.store = ArtifactStore(run)
        self.environment = environment
        self.ordered = order_stages(stages)
        self.resources = Resources(config, run)
        self.cache = (
            ArtifactCache(config.artifacts.cache_dir) if config.artifacts.reuse_cache else None
        )
        self.progress = progress

    # ------------------------------------------------------------------ planning

    def plan(
        self, *, from_stage: str | None = None, only: str | None = None, until: str | None = None
    ) -> list[PlannedStage]:
        return plan_stages(self.ordered, self.config, from_stage=from_stage, only=only, until=until)

    def _preflight(self, selected: Sequence[Stage]) -> None:
        """Fail before doing any work if a selected stage needs a capability this machine lacks."""
        problems: list[str] = []
        runnable: list[str] = []
        for stage in selected:
            if stage.skip_reason(self.config):
                continue
            missing = [
                f"{name} is {cap.status.value}" + (f" ({cap.detail})" if cap.detail else "")
                for name in stage.requires_capabilities
                if not (cap := self.environment.capability(name)).ok
            ]
            if missing:
                problems.append(f"stage {stage.name!r}: {'; '.join(missing)}")
            elif not problems:
                runnable.append(stage.name)
        if problems:
            hint = "Fix the environment (see `trtship doctor`)"
            if runnable:
                hint += (
                    f", or run only the stages this machine supports with --until {runnable[-1]}"
                )
            raise EnvironmentUnavailableError(
                "cannot run the selected stages on this machine:\n  " + "\n  ".join(problems),
                hint=hint + ".",
                details={"problems": problems},
            )

    # ------------------------------------------------------------------ execution

    def _tools(self, stage: Stage) -> dict[str, str | None]:
        tools: dict[str, str | None] = dict(self.environment.versions())
        if stage.requires_capabilities:
            tools["gpus"] = ";".join(
                f"{g.name}/{g.compute_capability}/{g.driver_version}" for g in self.environment.gpus
            )
        return tools

    def execute(
        self, *, from_stage: str | None = None, only: str | None = None, until: str | None = None
    ) -> PipelineResult:
        selected = select_stages(self.ordered, from_stage=from_stage, only=only, until=until)
        self._preflight(selected)
        self.run.update_manifest(lambda m: setattr(m, "status", RunStatus.RUNNING))
        outcomes: list[StageOutcome] = []
        try:
            for stage in selected:
                with log_context(run_id=self.run.run_id, stage=stage.name):
                    outcome = self._execute_stage(stage)
                outcomes.append(outcome)
                if self.progress:
                    self.progress(outcome)
        except BaseException:
            self.run.update_manifest(lambda m: setattr(m, "status", RunStatus.FAILED))
            raise
        self.run.update_manifest(lambda m: setattr(m, "status", RunStatus.SUCCEEDED))
        return PipelineResult(
            run_id=self.run.run_id,
            run_path=str(self.run.path),
            status=RunStatus.SUCCEEDED,
            stages=outcomes,
        )

    def _context(self, stage: Stage) -> StageContext:
        scratch = self.run.path / ".work" / f"{stage.name}-{secrets.token_hex(4)}"
        scratch.mkdir(parents=True)
        return StageContext(
            config=self.config,
            run=self.run,
            store=self.store,
            environment=self.environment,
            resources=self.resources,
            stage=stage.name,
            scratch=scratch,
        )

    def _execute_stage(self, stage: Stage) -> StageOutcome:
        started = time.perf_counter()
        reason = stage.skip_reason(self.config)
        if reason:
            self.run.update_stage(
                stage.name, status=StageStatus.SKIPPED, metrics={"reason": reason}
            )
            log.info("skipping %s: %s", stage.name, reason)
            return StageOutcome(
                name=stage.name, status=StageStatus.SKIPPED, duration_s=0.0, reason=reason
            )

        ctx = self._context(stage)
        try:
            for artifact_type in stage.requires:
                ctx.input(artifact_type)
            for artifact_type in stage.uses:
                ctx.input(artifact_type, optional=True)
            key = stage.cache_key(ctx, self._tools(stage))

            up_to_date = self._up_to_date(stage, key)
            if up_to_date is not None:
                return self._finish(
                    stage, StageStatus.SKIPPED, started, key, up_to_date, reason="up to date"
                )

            if stage.cacheable and self.cache is not None:
                adopted = self.cache.adopt(
                    key, stage.name, self.store, parents=[r.id for r in ctx.inputs_used]
                )
                if adopted is not None:
                    ids = [r.id for r in adopted]
                    return self._finish(
                        stage, StageStatus.CACHED, started, key, ids, reason="reused from cache"
                    )

            self.run.update_stage(
                stage.name,
                status=StageStatus.RUNNING,
                started_at=utc_now(),
                cache_key=key,
                error=None,
            )
            log.info("running stage %s", stage.name)
            result = stage.run(ctx)
            ids = [r.id for r in ctx.published]
            if stage.cacheable and self.cache is not None and ctx.published:
                self.cache.store(
                    key,
                    stage.name,
                    [CachedArtifact(r, self.store.absolute(r)) for r in ctx.published],
                )
            return self._finish(
                stage,
                StageStatus.SUCCEEDED,
                started,
                key,
                ids,
                metrics=result.metrics,
                warnings=result.warnings,
            )
        except KeyboardInterrupt:
            self._fail(stage, {"error": "KeyboardInterrupt", "message": "interrupted"})
            raise
        except TrtshipError as exc:
            self._fail(stage, exc.to_dict())
            raise
        except Exception as exc:
            self._fail(stage, {"error": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            shutil.rmtree(ctx.scratch, ignore_errors=True)
            work = self.run.path / ".work"
            if work.is_dir() and not any(work.iterdir()):
                work.rmdir()

    def _up_to_date(self, stage: Stage, key: str) -> list[str] | None:
        """Artifact ids of a previous successful execution with the same key, if still intact."""
        if not stage.cacheable:
            return None
        record = self.run.read_manifest().stages.get(stage.name)
        if record is None or record.cache_key != key:
            return None
        if record.status not in (StageStatus.SUCCEEDED, StageStatus.CACHED, StageStatus.SKIPPED):
            return None
        if not record.artifact_ids:
            return None
        try:
            for artifact_id in record.artifact_ids:
                self.store.verify(self.store.get(artifact_id))
        except TrtshipError:
            return None
        return list(record.artifact_ids)

    def _finish(
        self,
        stage: Stage,
        status: StageStatus,
        started: float,
        key: str,
        artifact_ids: list[str],
        *,
        reason: str | None = None,
        metrics: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> StageOutcome:
        duration = round(time.perf_counter() - started, 3)
        recorded = {**(metrics or {}), **({"reason": reason} if reason else {})}
        fields: dict[str, Any] = {
            "status": status,
            "finished_at": utc_now(),
            "cache_key": key,
            "artifact_ids": artifact_ids,
            "error": None,
        }
        if status is StageStatus.SUCCEEDED or status is StageStatus.CACHED:
            fields["metrics"] = recorded
        self.run.update_stage(stage.name, **fields)
        return StageOutcome(
            name=stage.name,
            status=status,
            duration_s=duration,
            artifacts=artifact_ids,
            reason=reason,
            metrics=metrics or {},
            warnings=warnings or [],
            cache_key=key,
        )

    def _fail(self, stage: Stage, error: dict[str, Any]) -> None:
        self.run.update_stage(
            stage.name, status=StageStatus.FAILED, finished_at=utc_now(), error=error
        )
        log.error("stage %s failed: %s", stage.name, error.get("message"))
