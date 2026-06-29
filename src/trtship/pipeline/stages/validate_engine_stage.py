"""``validate_engine``: PyTorch vs ONNX Runtime vs TensorRT, per precision."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from trtship.artifacts import ArtifactRecord, ArtifactType
from trtship.config import Precision, TrtshipConfig
from trtship.errors import ArtifactError
from trtship.pipeline.stage import Stage, StageContext, StageResult
from trtship.pipeline.stages._common import model_slice, profiles_slice, write_report
from trtship.tensorrt import TensorRTExecutor
from trtship.utils import env
from trtship.utils.hashing import sha256_json
from trtship.validation.engine import EngineExecutor, EngineUnderTest, validate_engines


def latest_engine_per_precision(records: list[ArtifactRecord]) -> dict[Precision, ArtifactRecord]:
    """The newest engine artifact for each precision (retries may have left older ones)."""
    latest: dict[Precision, ArtifactRecord] = {}
    for record in sorted(records, key=lambda r: r.created_at):
        latest[Precision(record.metadata["precision"])] = record
    return latest


def _open_engine(path: Path) -> EngineExecutor:
    return TensorRTExecutor(path)


class ValidateEngineStage(Stage):
    name: ClassVar[str] = "validate_engine"
    requires: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.ENGINE, ArtifactType.ONNX)
    uses: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.ONNX_OPTIMIZED,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.VALIDATION_REPORT,)
    requires_capabilities: ClassVar[tuple[str, ...]] = (
        env.NVIDIA_GPU,
        env.TENSORRT,
        env.TORCH_CUDA,
    )
    depends_on_model: ClassVar[bool] = True

    def config_slice(self, config: TrtshipConfig) -> Any:
        return {
            "model": model_slice(config),
            "validation": config.validation.model_dump(mode="json"),
            "profiles": profiles_slice(config),
        }

    def cache_key(self, ctx: StageContext, tools: dict[str, str | None]) -> str:
        # Every precision's engine is an input, not only the newest artifact.
        engines = latest_engine_per_precision(ctx.store.records(ArtifactType.ENGINE))
        hashes = {p.value: r.sha256 for p, r in engines.items()}
        return sha256_json({"base": super().cache_key(ctx, tools), "engines": hashes})

    def run(self, ctx: StageContext) -> StageResult:
        engines = latest_engine_per_precision(ctx.store.records(ArtifactType.ENGINE))
        if not engines:
            raise ArtifactError("the run has no engine to validate", hint="Run the build stage.")
        for record in engines.values():
            ctx.inputs_used.append(record)  # provenance: every engine is an input
        source_id = next(iter(engines.values())).metadata["source_onnx"]
        source = ctx.store.get(source_id)
        ctx.store.verify(source)
        report = validate_engines(
            [
                EngineUnderTest(precision=p, path=str(ctx.store.absolute(r)))
                for p, r in engines.items()
            ],
            ctx.store.absolute(source),
            ctx.resources.model,
            ctx.resources.signature,
            ctx.config,
            _open_engine,
        )
        path = write_report(ctx.scratch / "engine_validation.json", report.model_dump(mode="json"))
        ctx.publish(
            path,
            ArtifactType.VALIDATION_REPORT,
            metadata={
                "kind": "engine",
                "passed": report.passed,
                "precisions": [r.precision.value for r in report.results],
            },
        )
        report.raise_for_failure()
        return StageResult(
            metrics={
                r.precision.value: {
                    "passed": r.passed,
                    "max_abs_error": max(
                        (c.max_abs_error or 0.0 for p in r.points for c in p.vs_pytorch),
                        default=0.0,
                    ),
                }
                for r in report.results
            }
        )
