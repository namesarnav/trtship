"""``optimize``: write a cleaned-up copy of the ONNX model and validate it."""

from __future__ import annotations

from typing import Any, ClassVar

from trtship.artifacts import ArtifactType
from trtship.config import TrtshipConfig
from trtship.onnx import optimize_onnx, validate_onnx
from trtship.pipeline.stage import Stage, StageContext, StageResult
from trtship.pipeline.stages._common import model_slice, profiles_slice, write_report


class OptimizeStage(Stage):
    name: ClassVar[str] = "optimize"
    requires: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.ONNX,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (
        ArtifactType.ONNX_OPTIMIZED,
        ArtifactType.VALIDATION_REPORT,
    )
    depends_on_model: ClassVar[bool] = True

    def skip_reason(self, config: TrtshipConfig) -> str | None:
        return None if config.optimize.enabled else "optimize.enabled is false"

    def config_slice(self, config: TrtshipConfig) -> Any:
        return {
            "model": model_slice(config),
            "optimize": config.optimize.model_dump(mode="json"),
            "validation": config.validation.model_dump(mode="json"),
            "profiles": profiles_slice(config),
        }

    def run(self, ctx: StageContext) -> StageResult:
        onnx_record = ctx.input(ArtifactType.ONNX)
        assert onnx_record is not None
        name = ctx.resources.model.name
        candidate = ctx.scratch / f"{name}.optimized.onnx"
        result = optimize_onnx(
            ctx.store.absolute(onnx_record),
            candidate,
            ctx.resources.signature,
            ctx.config.optimize.passes,
        )
        report = validate_onnx(candidate, ctx.resources.model, ctx.resources.signature, ctx.config)
        report_path = write_report(
            ctx.scratch / "onnx_optimized_validation.json", report.model_dump(mode="json")
        )
        ctx.publish(
            report_path,
            ArtifactType.VALIDATION_REPORT,
            metadata={
                "subject": "optimized onnx",
                "kind": "onnx_optimized",
                "passed": report.passed,
            },
        )
        # A failing optimized model is never registered: its file stays in scratch and is removed.
        report.raise_for_failure()
        ctx.publish(
            candidate,
            ArtifactType.ONNX_OPTIMIZED,
            metadata={"optimize": result.model_dump(mode="json"), "validation_passed": True},
        )
        return StageResult(
            metrics={
                "changed": result.changed,
                "nodes_before": result.before.node_count,
                "nodes_after": result.after.node_count,
                "passes": {p.name: p.changes for p in result.passes},
            }
        )
