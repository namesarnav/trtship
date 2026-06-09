"""``build``: TensorRT engines, one per configured precision."""

from __future__ import annotations

from typing import Any, ClassVar

from trtship.artifacts import ArtifactType
from trtship.config import Precision, TrtshipConfig
from trtship.errors import EngineBuildError
from trtship.pipeline.stage import Stage, StageContext, StageResult
from trtship.tensorrt import build_engine
from trtship.utils import env


class BuildStage(Stage):
    name: ClassVar[str] = "build"
    requires: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.ONNX,)
    uses: ClassVar[tuple[ArtifactType, ...]] = (
        ArtifactType.ONNX_OPTIMIZED,
        ArtifactType.CALIBRATION_CACHE,
    )
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.ENGINE,)
    requires_capabilities: ClassVar[tuple[str, ...]] = (env.NVIDIA_GPU, env.TENSORRT)

    def config_slice(self, config: TrtshipConfig) -> Any:
        return {
            "tensorrt": config.tensorrt.model_dump(mode="json", exclude={"timing_cache_path"}),
            "inputs": [spec.model_dump(mode="json") for spec in config.model.inputs],
        }

    def run(self, ctx: StageContext) -> StageResult:
        optimized = ctx.input(ArtifactType.ONNX_OPTIMIZED, optional=True)
        source = optimized or ctx.input(ArtifactType.ONNX)
        assert source is not None
        onnx_path = ctx.store.absolute(source)
        name = ctx.resources.model.name
        metrics: dict[str, Any] = {"source": source.id}
        warnings: list[str] = []

        for precision in ctx.config.tensorrt.precisions:
            if precision is Precision.INT8:
                raise EngineBuildError(
                    "INT8 engines need a calibration cache, and the calibrate stage is not "
                    "implemented yet",
                    hint="Remove int8 from tensorrt.precisions for now.",
                )
            plan = ctx.scratch / f"{name}.{precision.value}.plan"
            result = build_engine(onnx_path, ctx.resources.signature, ctx.config, precision, plan)
            ctx.publish(
                plan,
                ArtifactType.ENGINE,
                metadata={
                    "precision": precision.value,
                    "source_onnx": source.id,
                    "build": result.model_dump(mode="json"),
                },
            )
            metrics[precision.value] = {
                "size_bytes": result.size_bytes,
                "build_time_s": result.build_time_s,
            }
            warnings.extend(f"{precision.value}: {w}" for w in result.warnings)
        return StageResult(metrics=metrics, warnings=warnings)
