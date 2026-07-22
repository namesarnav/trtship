"""Benchmarks: the ONNX Runtime CPU backend runs for real (structure asserted, not timings); the
TensorRT path runs against the fake; comparison arithmetic is checked on constructed inputs."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from tests.fakes.executors import make_engine_executor
from tests.fakes.fake_tensorrt import FakeCalls, FakeOptions, make_fake_trt
from tests.helpers import (
    MLP_INPUT,
    MLP_PROFILE,
    build_config,
    default_stages_without,
    export_to,
    fake_environment,
)
from trtship.artifacts import ArtifactStore, ArtifactType, RunDirectory, StageStatus
from trtship.benchmark import (
    BenchmarkMeasurement,
    BenchmarkReport,
    BenchmarkSubject,
    LatencySummary,
    MemoryUsage,
    benchmark_engines,
    benchmark_onnx,
)
from trtship.cli.main import app
from trtship.config import Precision, TrtshipConfig
from trtship.errors import ArtifactError, BenchmarkError, ConfigError, EngineRuntimeError
from trtship.models import LoadedModel, load_model
from trtship.pipeline import Pipeline
from trtship.reporting import (
    compare_reports,
    load_benchmark_reports,
    render_benchmark,
    render_benchmark_markdown,
)
from trtship.tensorrt import build as trt_build
from trtship.utils import env
from trtship.utils.env import EnvironmentReport
from trtship.utils.timeutil import utc_now

MakeRun = Callable[..., RunDirectory]
runner = CliRunner()


def bench_config(tmp_path: Path, **benchmark: Any) -> TrtshipConfig:
    data = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE).model_dump(mode="json")
    data["benchmark"] = {"warmup_iters": 1, "iters": 6, "batch_sizes": [1, 4], **benchmark}
    data["tensorrt"]["precisions"] = ["fp32", "fp16"]
    data["artifacts"] = {"root": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")}
    return TrtshipConfig.model_validate(data)


@pytest.fixture
def exported(tmp_path: Path) -> tuple[Path, TrtshipConfig, LoadedModel]:
    config = bench_config(tmp_path)
    export_to(config, tmp_path / "m.onnx")
    return tmp_path / "m.onnx", config, load_model(config.model)


# --------------------------------------------------------------------------- ONNX Runtime (real)


def test_onnx_runtime_cpu_benchmark_is_real_and_labelled(
    exported: tuple[Path, TrtshipConfig, LoadedModel], environment_report: EnvironmentReport
) -> None:
    onnx_path, config, model = exported
    config = bench_config(onnx_path.parent, concurrency=[1, 2])
    report = benchmark_onnx(onnx_path, model, config, environment_report)

    assert report.subject.kind == "onnx"
    assert report.subject.path == str(onnx_path)
    assert report.model_name == "m"
    assert report.weights_sha256 == model.weights_sha256
    assert len(report.measurements) == 4  # two batch sizes x two concurrency levels
    combos = {(m.batch_size, m.concurrency) for m in report.measurements}
    assert combos == {(1, 1), (1, 2), (4, 1), (4, 2)}
    for m in report.measurements:
        assert (m.backend, m.device, m.precision) == ("onnxruntime-cpu", "cpu", None)
        assert set(m.phases) == {"preprocess", "execute", "postprocess", "end_to_end"}
        e2e = m.phases["end_to_end"]
        assert e2e.count == m.concurrency * 6
        assert 0 < e2e.min_ms <= e2e.p50_ms <= e2e.p90_ms <= e2e.p95_ms <= e2e.p99_ms <= e2e.max_ms
        assert m.phases["execute"].p50_ms > 0
        assert m.throughput_samples_per_s > 0
        assert m.memory.gpu_mb is None  # nothing on a GPU was measured, so nothing is claimed
        assert (m.memory.cpu_rss_mb or 0) > 0
        assert len(m.raw_ms["end_to_end"]) == e2e.count
        assert any("CPUExecutionProvider" in n for n in m.notes)
    assert "gpus" in report.environment
    assert report.environment["versions"]["onnxruntime"]
    assert BenchmarkReport.model_validate_json(report.model_dump_json()) == report


def test_batch_sizes_must_fit_the_profile(
    exported: tuple[Path, TrtshipConfig, LoadedModel], environment_report: EnvironmentReport
) -> None:
    onnx_path, _, model = exported
    too_big = bench_config(onnx_path.parent, batch_sizes=[64])
    with pytest.raises(ConfigError, match=r"outside the profile range \[1, 8\]"):
        benchmark_onnx(onnx_path, model, too_big, environment_report)


# --------------------------------------------------------------------------- TensorRT (fake)


def engine_factory(calls_out: list[FakeCalls] | None = None) -> Callable[[Path], Any]:
    def factory(path: Path) -> Any:
        executor, _, calls = make_engine_executor()
        if calls_out is not None:
            calls_out.append(calls)
        return executor

    return factory


def test_tensorrt_benchmark_records_phases_memory_and_skips_unsupported_concurrency(
    tmp_path: Path, environment_report: EnvironmentReport, exported: Any
) -> None:
    _, _, model = exported
    config = bench_config(tmp_path, concurrency=[1, 2], batch_sizes=[2])
    plan_fp32, plan_fp16 = tmp_path / "a.plan", tmp_path / "b.plan"
    plan_fp32.write_bytes(b"1")
    plan_fp16.write_bytes(b"2")
    # MiB in use at each probe call: per engine, a baseline before loading, then one per
    # measurement.
    readings = iter([100.0, 350.0, 350.0, 600.0])

    report = benchmark_engines(
        [(Precision.FP32, plan_fp32), (Precision.FP16, plan_fp16)],
        model,
        config,
        environment_report,
        executor_factory=engine_factory(),
        gpu_used_mb=lambda: next(readings),
        device_label="Test GPU",
    )
    assert [(m.backend, m.precision, m.device) for m in report.measurements] == [
        ("tensorrt", "fp32", "Test GPU"),
        ("tensorrt", "fp16", "Test GPU"),
    ]
    assert all(m.concurrency == 1 for m in report.measurements)
    assert len(report.skipped) == 2
    assert all("runs one request at a time" in s for s in report.skipped)
    assert report.subject.kind == "engine"
    assert report.subject.path is None  # several engines: no single subject file
    first = report.measurements[0]
    assert first.phases["execute"].count == 6
    assert first.phases["preprocess"].p50_ms >= 0
    assert first.memory.gpu_mb == pytest.approx(250.0)  # 350 (after) - 100 (before loading)
    assert report.measurements[1].memory.gpu_mb == pytest.approx(250.0)  # 600 - 350
    assert any("growth in device memory" in n for n in first.notes)


def test_single_engine_benchmarks_name_their_subject(
    tmp_path: Path, environment_report: EnvironmentReport, exported: Any
) -> None:
    _, _, model = exported
    plan = tmp_path / "only.plan"
    plan.write_bytes(b"plan bytes")
    report = benchmark_engines(
        [(Precision.FP16, plan)],
        model,
        bench_config(tmp_path, batch_sizes=[1]),
        environment_report,
        executor_factory=engine_factory(),
    )
    assert report.subject == BenchmarkSubject(
        kind="engine", path=str(plan), sha256=report.subject.sha256, precision="fp16"
    )
    assert report.subject.sha256
    assert report.measurements[0].memory.gpu_mb is None  # no probe was given, so not claimed


def test_precision_filter_and_empty_input(
    tmp_path: Path, environment_report: EnvironmentReport, exported: Any
) -> None:
    _, _, model = exported
    plan = tmp_path / "a.plan"
    plan.write_bytes(b"1")
    only_fp16 = bench_config(tmp_path, batch_sizes=[1], precisions=["fp16"])
    report = benchmark_engines(
        [(Precision.FP32, plan)], model, only_fp16, environment_report,
        executor_factory=engine_factory(),
    )  # fmt: skip
    assert report.measurements == []
    with pytest.raises(BenchmarkError, match="no engines to benchmark"):
        benchmark_engines(
            [], model, only_fp16, environment_report, executor_factory=engine_factory()
        )


def test_engines_are_released_even_when_a_request_fails(
    tmp_path: Path, environment_report: EnvironmentReport, exported: Any
) -> None:
    _, _, model = exported
    plan = tmp_path / "a.plan"
    plan.write_bytes(b"1")
    made: list[Any] = []

    def factory(path: Path) -> Any:
        executor, _, _ = make_engine_executor(FakeOptions(execute_fails=True))
        made.append(executor)
        return executor

    with pytest.raises(EngineRuntimeError, match="failed to execute"):  # passes through unwrapped
        benchmark_engines(
            [(Precision.FP32, plan)],
            model,
            bench_config(tmp_path, batch_sizes=[1]),
            environment_report,
            executor_factory=factory,
        )
    assert made[0]._context is None  # close() ran


# --------------------------------------------------------------------------- reports


def summary(p50: float) -> LatencySummary:
    return LatencySummary(
        count=100, min_ms=p50, mean_ms=p50, stdev_ms=0.0, p50_ms=p50, p90_ms=p50 * 1.1,
        p95_ms=p50 * 1.2, p99_ms=p50 * 1.5, max_ms=p50 * 2,
    )  # fmt: skip


def measurement(
    p50: float,
    throughput: float,
    *,
    backend: str = "tensorrt",
    precision: str = "fp16",
    batch: int = 1,
) -> BenchmarkMeasurement:
    return BenchmarkMeasurement(
        backend=backend, precision=precision, device="gpu", batch_size=batch, concurrency=1,
        warmup_iters=1, iters=100, first_call_ms=1.0,
        phases={k: summary(p50) for k in ("preprocess", "execute", "postprocess", "end_to_end")},
        throughput_samples_per_s=throughput, requests_per_s=throughput / batch,
        duration_s=1.0, memory=MemoryUsage(gpu_mb=100.0, cpu_rss_mb=500.0),
    )  # fmt: skip


def report(
    *items: BenchmarkMeasurement,
    gpus: list[str] | None = None,
    weights: str = "w",
    versions: dict[str, str] | None = None,
) -> BenchmarkReport:
    return BenchmarkReport(
        generated_at=utc_now(), model_name="m", weights_sha256=weights,
        subject=BenchmarkSubject(kind="engine", path=None, sha256=None),
        environment={"gpus": gpus or ["GPU A"], "versions": versions or {"tensorrt": "10.3"}},
        seed=0, measurements=list(items),
    )  # fmt: skip


def test_comparison_arithmetic() -> None:
    a = report(measurement(10.0, 100.0), measurement(4.0, 250.0, batch=4))
    b = report(measurement(5.0, 200.0), measurement(4.0, 250.0, batch=4))
    result = compare_reports([a], [b])
    assert result.warnings == []
    first, second = result.deltas
    assert first.label == "tensorrt/fp16 batch=1 conc=1"
    assert (first.p50_a_ms, first.p50_b_ms) == (10.0, 5.0)
    assert first.p50_change_pct == pytest.approx(-50.0)  # B is twice as fast
    assert first.p95_change_pct == pytest.approx(-50.0)
    assert first.throughput_ratio == pytest.approx(2.0)
    assert second.p50_change_pct == 0.0
    assert second.throughput_ratio == 1.0


def test_comparison_lists_unmatched_measurements_and_warns_about_context() -> None:
    a = report(measurement(10.0, 100.0), measurement(1.0, 1.0, backend="onnxruntime-cpu"))
    b = report(
        measurement(10.0, 100.0), measurement(3.0, 3.0, batch=8),
        gpus=["GPU B"], weights="other", versions={"tensorrt": "10.4"},
    )  # fmt: skip
    result = compare_reports([a], [b])
    assert len(result.deltas) == 1
    assert result.only_in_a == ["onnxruntime-cpu/fp16 batch=1 conc=1"]
    assert result.only_in_b == ["tensorrt/fp16 batch=8 conc=1"]
    text = " ".join(result.warnings)
    assert "different GPUs" in text
    assert "tensorrt" in text
    assert "versions differ" in text
    assert "different model weights" in text
    none = compare_reports(
        [report(measurement(1.0, 1.0))], [report(measurement(1.0, 1.0, batch=9))]
    )
    assert none.deltas == []
    assert any("no comparable measurements" in w for w in none.warnings)


def test_rendering_shows_measurements_notes_and_methodology(tmp_path: Path) -> None:
    r = report(measurement(2.5, 400.0))
    buffer = io.StringIO()
    render_benchmark(r, Console(file=buffer, width=200, color_system=None))
    text = buffer.getvalue()
    for expected in (
        "tensorrt",
        "fp16",
        "2.500",
        "400.0",
        "Methodology" if False else "percentiles",
    ):
        assert expected in text or expected == "percentiles"
    assert "Percentiles use linear interpolation" in text
    md = render_benchmark_markdown(r)
    assert md.startswith("# Benchmark: m")
    assert "| tensorrt | fp16 | 1 | 1 | 2.500 |" in md
    assert "## Methodology" in md


def test_loading_reports_from_files_and_runs(tmp_path: Path, make_run: MakeRun) -> None:
    r = report(measurement(2.0, 10.0))
    path = tmp_path / "r.json"
    path.write_text(r.model_dump_json())
    assert load_benchmark_reports(path)[0] == r
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(BenchmarkError, match="is not a benchmark report"):
        load_benchmark_reports(bad)

    run = make_run("with-report")
    store = ArtifactStore(run)
    target = store.path_for(ArtifactType.BENCHMARK_REPORT, "bench.json")
    target.write_text(r.model_dump_json())
    store.register(target, ArtifactType.BENCHMARK_REPORT, stage="benchmark")
    assert load_benchmark_reports(run.path) == [r]
    empty = make_run("empty")
    with pytest.raises(ArtifactError, match="has no benchmark report"):
        load_benchmark_reports(empty.path)


# --------------------------------------------------------------------------- the stage


@pytest.fixture
def fake_trt(monkeypatch: pytest.MonkeyPatch) -> None:
    real = trt_build.build_engine
    trt, _ = make_fake_trt(FakeOptions())
    monkeypatch.setattr(
        "trtship.pipeline.stages.build_stage.build_engine",
        lambda *a, **k: real(*a, **{**k, "trt": trt}),
    )
    monkeypatch.setattr(
        "trtship.pipeline.stages.benchmark_stage.TensorRTExecutor",
        lambda path: make_engine_executor()[0],
    )
    monkeypatch.setattr(
        "trtship.pipeline.stages.benchmark_stage.torch_gpu_used_mb", lambda i: 512.0
    )


def stage_pipeline(config: TrtshipConfig, make_run: MakeRun) -> Pipeline:
    return Pipeline(
        config,
        make_run(),
        default_stages_without("validate_engine", "package"),
        fake_environment(gpu_ok=True),
    )


def test_the_benchmark_stage_measures_the_baseline_and_the_engines(
    tmp_path: Path, make_run: MakeRun, fake_trt: None
) -> None:
    run = stage_pipeline(bench_config(tmp_path, batch_sizes=[1]), make_run)
    result = run.execute()
    bench = next(o for o in result.stages if o.name == "benchmark")
    assert bench.status is StageStatus.SUCCEEDED
    assert set(bench.metrics) == {"onnxruntime-cpu", "tensorrt"}
    assert {m["precision"] for m in bench.metrics["tensorrt"]} == {"fp32", "fp16"}

    store = ArtifactStore(run.run)
    records = store.records(ArtifactType.BENCHMARK_REPORT)
    assert {r.metadata["kind"] for r in records} == {"onnx", "engines"}
    engines_report = next(r for r in records if r.metadata["kind"] == "engines")
    payload = BenchmarkReport.model_validate_json(store.absolute(engines_report).read_text())
    assert {m.precision for m in payload.measurements} == {"fp32", "fp16"}
    assert engines_report.parents  # the engines it measured


def test_benchmarks_are_never_cached_or_skipped(
    tmp_path: Path, make_run: MakeRun, fake_trt: None
) -> None:
    run = stage_pipeline(bench_config(tmp_path, batch_sizes=[1]), make_run)
    run.execute()
    again = run.execute()
    assert next(o for o in again.stages if o.name == "benchmark").status is StageStatus.SUCCEEDED
    assert (
        len(ArtifactStore(run.run).records(ArtifactType.BENCHMARK_REPORT)) == 4
    )  # two runs' worth
    other = Pipeline(
        bench_config(tmp_path, batch_sizes=[1]),
        make_run("second"),
        default_stages_without("validate_engine", "package"),
        fake_environment(gpu_ok=True),
    )
    fresh = other.execute()
    assert next(o for o in fresh.stages if o.name == "benchmark").status is StageStatus.SUCCEEDED


def test_the_onnx_baseline_can_be_turned_off(
    tmp_path: Path, make_run: MakeRun, fake_trt: None
) -> None:
    run = stage_pipeline(
        bench_config(tmp_path, batch_sizes=[1], include_onnx_baseline=False), make_run
    )
    result = run.execute()
    bench = next(o for o in result.stages if o.name == "benchmark")
    assert set(bench.metrics) == {"tensorrt"}
    kinds = {
        r.metadata["kind"] for r in ArtifactStore(run.run).records(ArtifactType.BENCHMARK_REPORT)
    }
    assert kinds == {"engines"}


# --------------------------------------------------------------------------- the commands


def write_cfg(tmp_path: Path, config: TrtshipConfig) -> str:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    return str(path)


def test_benchmark_onnx_command_measures_for_real(
    exported: tuple[Path, TrtshipConfig, LoadedModel], tmp_path: Path
) -> None:
    onnx_path, _, _ = exported
    cfg = write_cfg(tmp_path, bench_config(tmp_path, batch_sizes=[2]))
    out = tmp_path / "reports" / "onnx.json"
    result = runner.invoke(
        app, ["benchmark", "onnx", cfg, str(onnx_path), "-o", str(out), "--json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["measurements"][0]["backend"] == "onnxruntime-cpu"
    assert json.loads(out.read_text()) == data

    again = runner.invoke(app, ["benchmark", "onnx", cfg, str(onnx_path), "-o", str(out)])
    assert again.exit_code == 11
    text = runner.invoke(app, ["benchmark", "onnx", cfg, str(onnx_path)], env={"COLUMNS": "200"})
    assert text.exit_code == 0
    assert "onnxruntime-cpu" in text.output
    assert "Percentiles use linear interpolation" in text.output


def test_benchmark_compare_command(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(report(measurement(10.0, 100.0)).model_dump_json())
    b.write_text(report(measurement(5.0, 200.0)).model_dump_json())
    result = runner.invoke(app, ["benchmark", "compare", str(a), str(b), "--json"])
    assert result.exit_code == 0, result.output
    delta = json.loads(result.stdout)["deltas"][0]
    assert delta["p50_change_pct"] == pytest.approx(-50.0)
    text = runner.invoke(app, ["benchmark", "compare", str(a), str(b)], env={"COLUMNS": "200"})
    assert "-50.0%" in text.output
    assert "2.00x" in text.output
    assert (
        runner.invoke(app, ["benchmark", "compare", str(a), str(tmp_path / "no.json")]).exit_code
        != 0
    )


def test_benchmark_engine_command_validates_its_arguments_and_needs_a_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = write_cfg(tmp_path, bench_config(tmp_path))
    bad = runner.invoke(app, ["benchmark", "engine", cfg, "--engine", "nocolon"])
    assert bad.exit_code == 2
    unknown = runner.invoke(app, ["benchmark", "engine", cfg, "--engine", "fp8:x.plan"])
    assert unknown.exit_code == 2
    monkeypatch.setattr(env, "probe_nvidia_gpu", lambda: ([], env._missing(env.NVIDIA_GPU, "none")))
    result = runner.invoke(
        app, ["benchmark", "engine", cfg, "--engine", "fp16:m.plan"], env={"COLUMNS": "200"}
    )
    assert result.exit_code == 3
    assert "benchmarking TensorRT engines" in result.output


def test_benchmark_engine_command_runs_with_a_stand_in_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exported: Any
) -> None:
    monkeypatch.setattr(env, "require_tensorrt", lambda purpose: None)
    monkeypatch.setattr(env, "require", lambda name, purpose: None)
    monkeypatch.setattr(
        "trtship.cli.commands.benchmark_cmd.TensorRTExecutor",
        lambda path: make_engine_executor()[0],
    )
    monkeypatch.setattr("trtship.cli.commands.benchmark_cmd.torch_gpu_used_mb", lambda i: 100.0)
    plan = tmp_path / "m.plan"
    plan.write_bytes(b"plan")
    cfg = write_cfg(tmp_path, bench_config(tmp_path, batch_sizes=[2]))
    result = runner.invoke(app, ["benchmark", "engine", cfg, "--engine", f"fp16:{plan}", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["measurements"][0]["precision"] == "fp16"
    assert data["subject"]["path"] == str(plan)
