"""``build``: TensorRT engines, one per configured precision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from trtship.artifacts import ArtifactType
from trtship.calibration import make_cache_calibrator, read_cache_dir
from trtship.config import Precision, TrtshipConfig
from trtship.errors import EngineBuildError
from trtship.pipeline.stage import Stage, StageContext, StageResult
from trtship.tensorrt import build_engine, load_tensorrt
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
            calibration = self._calibration(ctx, precision, source.sha256)
            warnings.extend(calibration.notes)
            plan = ctx.scratch / f"{name}.{precision.value}.plan"
            result = build_engine(
                onnx_path,
                ctx.resources.signature,
                ctx.config,
                precision,
                plan,
                calibrator=calibration.calibrator,
            )
            ctx.publish(
                plan,
                ArtifactType.ENGINE,
                metadata={
                    "precision": precision.value,
                    "source_onnx": source.id,
                    "calibration_cache": calibration.cache_id,
                    "build": result.model_dump(mode="json"),
                },
            )
            metrics[precision.value] = {
                "size_bytes": result.size_bytes,
                "build_time_s": result.build_time_s,
            }
            warnings.extend(f"{precision.value}: {w}" for w in result.warnings)
        return StageResult(metrics=metrics, warnings=warnings)

    def _calibration(
        self, ctx: StageContext, precision: Precision, onnx_sha256: str
    ) -> _Calibration:
        """A cache-serving calibrator for INT8 builds; nothing for other precisions."""
        if precision is not Precision.INT8:
            return _Calibration(None, None, [])
        record = ctx.input(ArtifactType.CALIBRATION_CACHE, optional=True)
        if record is None:
            raise EngineBuildError(
                "an INT8 engine needs a calibration cache, but the run has none",
                hint="Run the calibrate stage (it needs a `calibration` config section).",
            )
        cache, metadata = read_cache_dir(ctx.store.absolute(record))
        notes = []
        if metadata.source_onnx_sha256 != onnx_sha256:
            notes.append(
                "the calibration cache was computed from a different ONNX file than the one "
                "being built; TensorRT may recalibrate or reject some scales"
            )
        trt = load_tensorrt(purpose="building an INT8 TensorRT engine")
        return _Calibration(make_cache_calibrator(trt, metadata.method, cache), record.id, notes)


@dataclass(frozen=True)
class _Calibration:
    calibrator: Any | None
    cache_id: str | None
    notes: list[str]
