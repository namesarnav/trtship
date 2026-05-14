"""``validate``: check the exported ONNX graph and compare it with PyTorch."""

from __future__ import annotations

from typing import Any, ClassVar

from trtship.artifacts import ArtifactType
from trtship.config import TrtshipConfig
from trtship.onnx import validate_onnx
from trtship.pipeline.stage import Stage, StageContext, StageResult
from trtship.pipeline.stages._common import model_slice, profiles_slice, write_report


class ValidateStage(Stage):
    name: ClassVar[str] = "validate"
    requires: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.ONNX,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.VALIDATION_REPORT,)
    depends_on_model: ClassVar[bool] = True

    def config_slice(self, config: TrtshipConfig) -> Any:
        return {
            "model": model_slice(config),
            "validation": config.validation.model_dump(mode="json"),
            "profiles": profiles_slice(config),
        }

    def run(self, ctx: StageContext) -> StageResult:
        onnx_record = ctx.input(ArtifactType.ONNX)
        assert onnx_record is not None
        report = validate_onnx(
            ctx.store.absolute(onnx_record),
            ctx.resources.model,
            ctx.resources.signature,
            ctx.config,
        )
        path = write_report(ctx.scratch / "onnx_validation.json", report.model_dump(mode="json"))
        ctx.publish(
            path,
            ArtifactType.VALIDATION_REPORT,
            metadata={"subject": onnx_record.id, "kind": "onnx", "passed": report.passed},
        )
        report.raise_for_failure()  # the report is already published as evidence
        worst = max((o.max_abs_error or 0.0 for p in report.points for o in p.outputs), default=0.0)
        return StageResult(
            metrics={
                "passed": report.passed,
                "shape_points": len(report.points),
                "max_abs_error": worst,
            },
            warnings=list(report.graph.warnings),
        )
