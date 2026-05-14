"""``export``: PyTorch -> ONNX."""

from __future__ import annotations

from typing import Any, ClassVar

from trtship.artifacts import ArtifactType
from trtship.config import TrtshipConfig
from trtship.export import export_onnx
from trtship.pipeline.stage import Stage, StageContext, StageResult
from trtship.pipeline.stages._common import model_slice, profiles_slice


class ExportStage(Stage):
    name: ClassVar[str] = "export"
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.ONNX,)
    depends_on_model: ClassVar[bool] = True

    def config_slice(self, config: TrtshipConfig) -> Any:
        return {
            "model": model_slice(config),
            "export": config.export.model_dump(mode="json"),
            "profiles": profiles_slice(config),
        }

    def run(self, ctx: StageContext) -> StageResult:
        model = ctx.resources.model
        result = export_onnx(
            model, ctx.resources.signature, ctx.config, ctx.scratch / f"{model.name}.onnx"
        )
        ctx.publish(
            ctx.scratch / f"{model.name}.onnx",
            ArtifactType.ONNX,
            metadata=result.metadata.model_dump(mode="json"),
        )
        return StageResult(
            metrics={
                "opset": result.metadata.opset,
                "exporter": result.metadata.exporter,
                "size_bytes": result.size_bytes,
            },
            warnings=list(result.metadata.warnings),
        )
