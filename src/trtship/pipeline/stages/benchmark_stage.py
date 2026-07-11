"""``benchmark``: measure the built engines (and, optionally, an ONNX Runtime CPU baseline)."""

from __future__ import annotations

from typing import Any, ClassVar

from trtship.artifacts import ArtifactType
from trtship.benchmark import benchmark_engines, benchmark_onnx, torch_gpu_used_mb
from trtship.config import TrtshipConfig
from trtship.errors import ArtifactError
from trtship.pipeline.stage import Stage, StageContext, StageResult
from trtship.pipeline.stages._common import model_slice, profiles_slice, write_report
from trtship.pipeline.stages.validate_engine_stage import latest_engine_per_precision
from trtship.tensorrt import TensorRTExecutor
from trtship.utils import env


class BenchmarkStage(Stage):
    name: ClassVar[str] = "benchmark"
    requires: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.ENGINE, ArtifactType.ONNX)
    uses: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.ONNX_OPTIMIZED,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.BENCHMARK_REPORT,)
    requires_capabilities: ClassVar[tuple[str, ...]] = (
        env.NVIDIA_GPU,
        env.TENSORRT,
        env.TORCH_CUDA,
    )
    depends_on_model: ClassVar[bool] = True
    cacheable: ClassVar[bool] = False  # a measurement of this machine, right now

    def config_slice(self, config: TrtshipConfig) -> Any:
        return {
            "model": model_slice(config),
            "benchmark": config.benchmark.model_dump(mode="json"),
            "profiles": profiles_slice(config),
        }

    def run(self, ctx: StageContext) -> StageResult:
        engines = latest_engine_per_precision(ctx.store.records(ArtifactType.ENGINE))
        if not engines:
            raise ArtifactError("the run has no engine to benchmark", hint="Run the build stage.")
        for record in engines.values():
            ctx.inputs_used.append(record)
        source = ctx.store.get(next(iter(engines.values())).metadata["source_onnx"])
        ctx.store.verify(source)
        model = ctx.resources.model
        metrics: dict[str, Any] = {}
        warnings: list[str] = []

        if ctx.config.benchmark.include_onnx_baseline:
            baseline = benchmark_onnx(
                ctx.store.absolute(source), model, ctx.config, ctx.environment
            )
            self._publish(ctx, "benchmark_onnx_cpu.json", baseline, "onnx")
            metrics["onnxruntime-cpu"] = _headline(baseline)

        device = ctx.environment.gpus[0].name if ctx.environment.gpus else "cuda:0"
        report = benchmark_engines(
            [(p, ctx.store.absolute(r)) for p, r in engines.items()],
            model,
            ctx.config,
            ctx.environment,
            executor_factory=TensorRTExecutor,
            gpu_used_mb=lambda: torch_gpu_used_mb(ctx.config.tensorrt.device_index),
            device_label=device,
        )
        self._publish(ctx, "benchmark_engines.json", report, "engines")
        metrics["tensorrt"] = _headline(report)
        warnings.extend(f"not measured: {s}" for s in report.skipped)
        return StageResult(metrics=metrics, warnings=warnings)

    @staticmethod
    def _publish(ctx: StageContext, name: str, report: Any, kind: str) -> None:
        path = write_report(ctx.scratch / name, report.model_dump(mode="json"))
        ctx.publish(path, ArtifactType.BENCHMARK_REPORT, metadata={"kind": kind})


def _headline(report: Any) -> list[dict[str, Any]]:
    """A compact per-measurement summary for the run manifest."""
    return [
        {
            "precision": m.precision,
            "batch": m.batch_size,
            "concurrency": m.concurrency,
            "e2e_p50_ms": round(m.phases["end_to_end"].p50_ms, 4),
            "samples_per_s": round(m.throughput_samples_per_s, 2),
        }
        for m in report.measurements
    ]
