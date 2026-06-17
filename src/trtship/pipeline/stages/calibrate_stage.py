"""``calibrate``: INT8 calibration, producing a calibration cache artifact."""

from __future__ import annotations

from typing import Any, ClassVar

from trtship.artifacts import ArtifactType
from trtship.calibration import calibrate, prepare
from trtship.config import Precision, TrtshipConfig
from trtship.pipeline.stage import Stage, StageContext, StageResult
from trtship.utils import env
from trtship.utils.hashing import sha256_json


class CalibrateStage(Stage):
    name: ClassVar[str] = "calibrate"
    requires: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.ONNX,)
    uses: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.ONNX_OPTIMIZED,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.CALIBRATION_CACHE,)
    requires_capabilities: ClassVar[tuple[str, ...]] = (
        env.NVIDIA_GPU,
        env.TENSORRT,
        env.TORCH_CUDA,
    )
    depends_on_model: ClassVar[bool] = True

    def skip_reason(self, config: TrtshipConfig) -> str | None:
        if Precision.INT8 not in config.tensorrt.precisions:
            return "int8 is not in tensorrt.precisions"
        return None

    def config_slice(self, config: TrtshipConfig) -> Any:
        assert config.calibration is not None
        return {
            "calibration": config.calibration.model_dump(
                mode="json", exclude={"path", "cache_path"}
            ),
            "tensorrt": {
                "profiles": [p.model_dump(mode="json") for p in config.tensorrt.profiles],
                "workspace_mb": config.tensorrt.workspace_mb,
                "optimization_level": config.tensorrt.optimization_level,
            },
            "inputs": [spec.model_dump(mode="json") for spec in config.model.inputs],
        }

    def cache_key(self, ctx: StageContext, tools: dict[str, str | None]) -> str:
        # The data itself is part of the identity, not just where it lives.
        fingerprint = prepare(ctx.config).identity.fingerprint
        return sha256_json({"base": super().cache_key(ctx, tools), "dataset": fingerprint})

    def run(self, ctx: StageContext) -> StageResult:
        optimized = ctx.input(ArtifactType.ONNX_OPTIMIZED, optional=True)
        source = optimized or ctx.input(ArtifactType.ONNX)
        assert source is not None
        model = ctx.resources.model
        target = ctx.scratch / f"{model.name}.calibration"
        metadata = calibrate(
            ctx.store.absolute(source),
            ctx.resources.signature,
            ctx.config,
            target,
            model_weights_sha256=model.weights_sha256,
        )
        ctx.publish(
            target,
            ArtifactType.CALIBRATION_CACHE,
            metadata={
                "method": metadata.method,
                "dataset": metadata.dataset.model_dump(mode="json"),
                "sample_count": metadata.sample_count,
                "batch_size": metadata.batch_size,
                "source_onnx": source.id,
                "representative": metadata.representative,
            },
        )
        warnings = []
        if not metadata.representative:
            warnings.append(
                "calibration used synthetic data, which is not representative; the INT8 scales "
                "and the accuracy of the INT8 engine are unreliable"
            )
        return StageResult(
            metrics={
                "samples": metadata.sample_count,
                "batches": metadata.num_batches,
                "dataset": metadata.dataset.name,
                "method": metadata.method,
            },
            warnings=warnings,
        )
