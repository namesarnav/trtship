"""``package``: assemble a Triton model repository from a built engine. Needs no GPU."""

from __future__ import annotations

from typing import Any, ClassVar

from trtship.artifacts import ArtifactRecord, ArtifactType
from trtship.config import Precision, TrtshipConfig
from trtship.errors import ArtifactError, TritonError, ValidationFailedError
from trtship.pipeline.stage import Stage, StageContext, StageResult
from trtship.pipeline.stages.validate_engine_stage import latest_engine_per_precision
from trtship.tensorrt import EngineInfo
from trtship.triton import build_repository


def select_engine(
    engines: dict[Precision, ArtifactRecord], wanted: Precision | None
) -> ArtifactRecord:
    """The engine to package: ``triton.precision``, or the only one built."""
    if not engines:
        raise ArtifactError("the run has no engine to package", hint="Run the build stage.")
    if wanted is not None:
        if wanted not in engines:
            built = ", ".join(sorted(p.value for p in engines))
            raise TritonError(
                f"triton.precision is {wanted.value} but the run built: {built}",
                hint="Build that precision, or change triton.precision.",
            )
        return engines[wanted]
    if len(engines) > 1:
        built = ", ".join(sorted(p.value for p in engines))
        raise TritonError(
            f"the run built several engines ({built}) and triton.precision is not set",
            hint="Set triton.precision to the one to serve.",
        )
    return next(iter(engines.values()))


class PackageStage(Stage):
    name: ClassVar[str] = "package"
    requires: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.ENGINE,)
    uses: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.VALIDATION_REPORT,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.TRITON_REPOSITORY,)

    cacheable: ClassVar[bool] = False  # a copy of the engine plus a small text file

    def config_slice(self, config: TrtshipConfig) -> Any:
        return {"triton": config.triton.model_dump(mode="json"), "model": config.model.name}

    def run(self, ctx: StageContext) -> StageResult:
        engines = latest_engine_per_precision(ctx.store.records(ArtifactType.ENGINE))
        record = select_engine(engines, ctx.config.triton.precision)
        ctx.store.verify(record)
        ctx.inputs_used.append(record)
        precision = Precision(record.metadata["precision"])
        warnings = self._check_validation(ctx, precision)

        info = EngineInfo.model_validate(record.metadata["build"]["engine"])
        result = build_repository(
            ctx.scratch / "model_repository",
            ctx.config.model.name,
            ctx.store.absolute(record),
            info,
            ctx.config.triton,
        )
        published = ctx.publish(
            ctx.scratch / "model_repository",
            ArtifactType.TRITON_REPOSITORY,
            "model_repository",
            metadata={
                "model_name": result.model_name,
                "version": result.version,
                "precision": precision.value,
                "max_batch_size": result.max_batch_size,
                "engine": record.id,
            },
        )
        return StageResult(
            metrics={
                "repository": published.path,
                "model": result.model_name,
                "precision": precision.value,
                "max_batch_size": result.max_batch_size,
            },
            warnings=warnings,
        )

    @staticmethod
    def _check_validation(ctx: StageContext, precision: Precision) -> list[str]:
        """Refuse to package an engine that failed validation; warn if it was never validated."""
        reports = [
            r
            for r in ctx.store.records(ArtifactType.VALIDATION_REPORT)
            if r.metadata.get("kind") == "engine"
        ]
        if not reports:
            return [
                "the engine has not been validated against PyTorch (no engine validation report)"
            ]
        newest = max(reports, key=lambda r: r.created_at)
        ctx.store.verify(newest)
        ctx.inputs_used.append(newest)
        if not newest.metadata.get("passed", False):
            raise ValidationFailedError(
                "the engine failed numerical validation; refusing to package it",
                hint="Fix the accuracy problem (see the validation report) and rebuild.",
            )
        if precision.value not in newest.metadata.get("precisions", []):
            return [f"the {precision.value} engine was not covered by the engine validation report"]
        return []
