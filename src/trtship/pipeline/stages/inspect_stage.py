"""``inspect``: parameters, memory estimate, and I/O signature of the model."""

from __future__ import annotations

from typing import Any, ClassVar

from trtship.artifacts import ArtifactType
from trtship.config import TrtshipConfig
from trtship.models.inspection import inspect_model
from trtship.pipeline.stage import Stage, StageContext, StageResult
from trtship.pipeline.stages._common import model_slice, profiles_slice, write_report


class InspectStage(Stage):
    name: ClassVar[str] = "inspect"
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.MODEL_REPORT,)
    depends_on_model: ClassVar[bool] = True

    def config_slice(self, config: TrtshipConfig) -> Any:
        return {"model": model_slice(config), "profiles": profiles_slice(config)}

    def run(self, ctx: StageContext) -> StageResult:
        model = ctx.resources.model
        report = inspect_model(
            model,
            ctx.config.model,
            ctx.config.tensorrt.profiles,
            seed=ctx.config.seed,
            signature=ctx.resources.signature,
        )
        path = write_report(ctx.scratch / "model_report.json", report.model_dump(mode="json"))
        ctx.publish(
            path,
            ArtifactType.MODEL_REPORT,
            metadata={
                "parameters": report.parameters.total,
                "weights_sha256": report.weights_sha256,
            },
        )
        return StageResult(
            metrics={
                "parameters": report.parameters.total,
                "trainable_parameters": report.parameters.trainable,
                "weights_bytes": report.memory.weights_bytes,
            },
            warnings=list(report.memory.notes),
        )
