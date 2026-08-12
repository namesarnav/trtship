"""Benchmarking a served model: the client-side measurement, the server-side statistics, and the
serving-overhead comparison.

The stub (tests/fakes/fake_triton.py) is not Triton and its statistics are synthetic constants, so
these tests prove the plumbing (which requests are counted, what is subtracted from what, how the
figures are reported), not the performance of any real server.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from tests.fakes.fake_triton import FakeTritonServer, StubModel, StubTensor
from tests.helpers import MLP_INPUT, MLP_PROFILE, build_config, export_to
from trtship.benchmark import (
    BenchmarkMeasurement,
    BenchmarkReport,
    BenchmarkSubject,
    LatencySummary,
    MemoryUsage,
    ServerSideTimes,
    benchmark_triton,
)
from trtship.benchmark.triton_target import server_side_delta
from trtship.cli.main import app
from trtship.config import Precision, TrtshipConfig
from trtship.errors import BenchmarkError, TritonError
from trtship.models import load_model
from trtship.onnx.runtime import OrtSession
from trtship.reporting import (
    compare_serving,
    render_benchmark_markdown,
    render_serving_markdown,
)
from trtship.triton import ModelStatistics, StageStatistic, TritonClient
from trtship.triton.client import Protocol
from trtship.utils import env
from trtship.utils.timeutil import utc_now

runner = CliRunner()
PROTOCOLS: list[Protocol] = ["http", "grpc"]
ITERS = 5
WARMUP = 2


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[FakeTritonServer, Path, TrtshipConfig]]:
    """(stub server serving the MLP, config file path, config) with a repository plan present."""
    config = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)
    onnx_path = tmp_path / "m.onnx"
    export_to(config, onnx_path)
    (tmp_path / "repo/m/1").mkdir(parents=True)
    (tmp_path / "repo/m/1/model.plan").write_bytes(b"placeholder plan")
    ort = OrtSession(onnx_path)
    with FakeTritonServer() as server:
        server.state.add(
            StubModel(
                name="m",
                inputs=[StubTensor("x", "FP32", [-1, 16])],
                outputs=[StubTensor(n, "FP32", [-1, 4]) for n in ort.output_names],
                function=ort.run,
            )
        )
        data = config.model_dump(mode="json")
        data["model"]["factory"] = "trtship_fixtures.models:tiny_mlp"
        data["triton"] = {
            "repository_dir": str(tmp_path / "repo"),
            "http_port": server.http_port,
            "grpc_port": server.grpc_port,
        }
        data["benchmark"] = {
            "warmup_iters": WARMUP,
            "iters": ITERS,
            "batch_sizes": [1, 4],
            "concurrency": [1],
        }
        data["artifacts"] = {"root": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")}
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(data))
        yield server, path, TrtshipConfig.model_validate(data)


def run_benchmark(
    server: FakeTritonServer, config: TrtshipConfig, protocol: Protocol
) -> BenchmarkReport:
    url = server.http_url if protocol == "http" else server.grpc_url
    return benchmark_triton(
        lambda: TritonClient(protocol, url, timeout_s=5.0),
        load_model(config.model),
        config,
        env.probe_all(),
        protocol=protocol,
        precision=Precision.FP16,
        endpoint=url,
    )


# --------------------------------------------------------------------------- measurement


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_a_served_model_is_measured_through_the_client(
    served: tuple[FakeTritonServer, Path, TrtshipConfig], protocol: Protocol
) -> None:
    server, _, config = served
    report = run_benchmark(server, config, protocol)

    assert report.subject.kind == "triton"
    assert len(report.measurements) == 2
    for m in report.measurements:
        assert (m.backend, m.precision) == (f"triton-{protocol}", "fp16")
        assert set(m.phases) == {"preprocess", "execute", "postprocess", "end_to_end"}
        assert m.phases["end_to_end"].count == ITERS
        assert m.memory.gpu_mb is None  # the GPU belongs to the server process
        assert any("whole request call" in n for n in m.notes)
    assert BenchmarkReport.model_validate_json(report.model_dump_json()) == report


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_server_side_means_come_from_the_servers_statistics_for_the_timed_section_only(
    served: tuple[FakeTritonServer, Path, TrtshipConfig], protocol: Protocol
) -> None:
    server, _, config = served
    report = run_benchmark(server, config, protocol)

    for m in report.measurements:
        side = m.server_side
        assert side is not None
        assert side.source == "Triton inference statistics"
        # The stub charges fixed per-request costs, so the per-request means are those constants
        # (100 / 50 / 400 / 30 microseconds) however many requests warmup and the cold call added.
        assert side.queue_ms == pytest.approx(0.1)
        assert side.compute_input_ms == pytest.approx(0.05)
        assert side.compute_infer_ms == pytest.approx(0.4)
        assert side.compute_output_ms == pytest.approx(0.03)
        assert side.total_ms == pytest.approx(0.58)
        assert side.requests == ITERS  # warmup and the first call are excluded
    # Every request (cold call + warmup + timed) reached the server for each batch size.
    assert server.state.models["m"].successes == 2 * (1 + WARMUP + ITERS)


def test_concurrent_workers_each_get_their_own_client(
    served: tuple[FakeTritonServer, Path, TrtshipConfig],
) -> None:
    server, _, config = served
    data = config.model_dump(mode="json")
    data["benchmark"]["concurrency"] = [3]
    data["benchmark"]["batch_sizes"] = [1]
    concurrent = TrtshipConfig.model_validate(data)
    report = run_benchmark(server, concurrent, "http")
    (m,) = report.measurements
    assert m.concurrency == 3
    assert m.phases["end_to_end"].count == 3 * ITERS
    assert m.server_side is not None
    assert m.server_side.requests == 3 * ITERS


def test_a_model_that_is_not_ready_is_refused_before_measuring(
    served: tuple[FakeTritonServer, Path, TrtshipConfig],
) -> None:
    server, _, config = served
    server.state.models["m"].ready = False
    with pytest.raises(BenchmarkError, match="not ready"):
        run_benchmark(server, config, "http")


def test_unavailable_statistics_leave_server_side_empty_and_say_why(
    served: tuple[FakeTritonServer, Path, TrtshipConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, _, config = served

    def refuse(self: TritonClient, model: str) -> ModelStatistics:
        raise TritonError("statistics are disabled on this server")

    monkeypatch.setattr(TritonClient, "inference_statistics", refuse)
    report = run_benchmark(server, config, "http")
    for m in report.measurements:
        assert m.server_side is None
        assert any("server-side timing unavailable" in n for n in m.notes)
        assert m.phases["end_to_end"].count == ITERS  # the client-side measurement is unaffected


# --------------------------------------------------------------------------- statistics delta


def stats(count: int, ns_each: tuple[int, int, int, int]) -> ModelStatistics:
    def stage(ns: int) -> StageStatistic:
        return StageStatistic(count=count, ns=ns * count)

    queue, comp_in, infer, comp_out = ns_each
    return ModelStatistics(
        success=stage(sum(ns_each)),
        queue=stage(queue),
        compute_input=stage(comp_in),
        compute_infer=stage(infer),
        compute_output=stage(comp_out),
        inference_count=count,
        execution_count=count,
    )


def test_the_delta_is_the_per_request_mean_over_the_interval() -> None:
    before = stats(10, (1_000_000, 0, 0, 0))
    after = stats(30, (2_000_000, 500_000, 3_000_000, 250_000))
    times, note = server_side_delta(before, after, requests_sent=20)
    # queue: (30*2ms - 10*1ms) / 20 = 2.5ms, compute_input: 30*0.5ms / 20 = 0.75ms ...
    assert times is not None
    assert note is None
    assert times.requests == 20
    assert times.queue_ms == pytest.approx(2.5)
    assert times.compute_input_ms == pytest.approx(0.75)
    assert times.compute_infer_ms == pytest.approx(4.5)
    assert times.compute_output_ms == pytest.approx(0.375)


def test_a_count_mismatch_is_reported_not_hidden() -> None:
    times, note = server_side_delta(stats(0, (1, 1, 1, 1)), stats(7, (1, 1, 1, 1)), 5)
    assert times is not None
    assert times.requests == 7
    assert note is not None
    assert "7" in note
    assert "5" in note


def test_no_counted_requests_yields_no_figures() -> None:
    times, note = server_side_delta(stats(4, (1, 1, 1, 1)), stats(4, (1, 1, 1, 1)), 5)
    assert times is None
    assert note is not None
    assert "no successful requests" in note


# --------------------------------------------------------------------------- overhead comparison


def summary(mean: float) -> LatencySummary:
    return LatencySummary(
        count=10,
        min_ms=mean * 0.9,
        mean_ms=mean,
        stdev_ms=mean * 0.05,
        p50_ms=mean,
        p90_ms=mean * 1.1,
        p95_ms=mean * 1.2,
        p99_ms=mean * 1.3,
        max_ms=mean * 1.5,
    )


def measurement(
    backend: str,
    mean: float,
    *,
    precision: str = "fp16",
    batch: int = 1,
    concurrency: int = 1,
    side: ServerSideTimes | None = None,
) -> BenchmarkMeasurement:
    phases = {p: summary(mean) for p in ("preprocess", "execute", "postprocess", "end_to_end")}
    return BenchmarkMeasurement(
        backend=backend,
        precision=precision,
        device="d",
        batch_size=batch,
        concurrency=concurrency,
        warmup_iters=1,
        iters=10,
        first_call_ms=mean,
        phases=phases,
        throughput_samples_per_s=batch / mean * 1000,
        requests_per_s=1000 / mean,
        duration_s=1.0,
        memory=MemoryUsage(),
        server_side=side,
    )


def report(
    kind: str, measurements: list[BenchmarkMeasurement], gpus: list[str] | None = None
) -> BenchmarkReport:
    return BenchmarkReport(
        generated_at=utc_now(),
        model_name="m",
        weights_sha256="w",
        subject=BenchmarkSubject(kind=kind, path=None, sha256=None),
        environment={"gpus": gpus or ["gpu-a"], "versions": {"tensorrt": "10"}},
        seed=0,
        measurements=measurements,
    )


SIDE = ServerSideTimes(
    source="Triton inference statistics",
    requests=10,
    queue_ms=0.5,
    compute_input_ms=0.25,
    compute_infer_ms=2.0,
    compute_output_ms=0.25,
)


def test_overhead_is_served_minus_direct_with_a_derived_client_and_network_share() -> None:
    direct = report("engine", [measurement("tensorrt", 3.0)])
    served_report = report("triton", [measurement("triton-http", 5.0, side=SIDE)])
    comparison = compare_serving([direct], [served_report])

    (row,) = comparison.rows
    assert row.overhead_ms == pytest.approx(2.0)
    assert row.overhead_pct == pytest.approx(200.0 / 3.0)
    assert row.server_total_ms == pytest.approx(3.0)
    assert row.client_and_network_ms == pytest.approx(2.0)  # 5.0 served - 3.0 server-side
    assert (row.queue_ms, row.compute_infer_ms) == (0.5, 2.0)
    assert comparison.warnings == []
    assert comparison.unmatched == []


def test_rows_without_a_counterpart_are_listed_with_the_reason() -> None:
    direct = report("engine", [measurement("tensorrt", 3.0), measurement("tensorrt", 6.0, batch=8)])
    served_report = report(
        "triton",
        [
            measurement("triton-http", 5.0, side=SIDE),
            measurement("triton-http", 5.0, concurrency=4, side=SIDE),
            measurement("triton-grpc", 5.0, precision="fp32", side=SIDE),
        ],
    )
    comparison = compare_serving([direct], [served_report])
    assert [r.backend for r in comparison.rows] == ["triton-http"]
    text = "\n".join(comparison.unmatched)
    assert "conc=4" in text
    assert "triton-grpc/fp32 batch=1" in text
    assert "tensorrt/fp16 batch=8 conc=1: no served measurement" in text


def test_missing_server_statistics_leave_the_breakdown_empty() -> None:
    direct = report("engine", [measurement("tensorrt", 3.0)])
    served_report = report("triton", [measurement("triton-http", 5.0)])
    (row,) = compare_serving([direct], [served_report]).rows
    assert row.overhead_ms == pytest.approx(2.0)
    assert row.queue_ms is None
    assert row.client_and_network_ms is None


def test_reports_from_different_machines_are_flagged() -> None:
    direct = report("engine", [measurement("tensorrt", 3.0)], gpus=["gpu-a"])
    served_report = report("triton", [measurement("triton-http", 5.0)], gpus=["gpu-b"])
    comparison = compare_serving([direct], [served_report])
    assert any("different GPUs" in w for w in comparison.warnings)


def test_nothing_to_pair_is_a_warning_not_an_empty_success() -> None:
    comparison = compare_serving(
        [report("engine", [measurement("tensorrt", 3.0)])],
        [report("onnx", [measurement("onnxruntime-cpu", 1.0)])],
    )
    assert comparison.rows == []
    assert any("could be paired" in w for w in comparison.warnings)


def test_markdown_renders_the_breakdown_and_labels_the_derived_column() -> None:
    direct = report("engine", [measurement("tensorrt", 3.0)])
    served_report = report("triton", [measurement("triton-http", 5.0, side=SIDE)])
    text = render_serving_markdown(compare_serving([direct], [served_report]))
    assert "Client + network (derived)" in text
    assert "| triton-http | fp16 | 1 | 3.000 | 5.000 | +2.000 |" in text
    benchmark_text = render_benchmark_markdown(served_report)
    assert "## Server-side means per request" in benchmark_text
    assert "| triton-http | 1 | 1 | 0.500 | 0.250 | 2.000 | 0.250 | 10 |" in benchmark_text


# --------------------------------------------------------------------------- CLI


def test_cli_benchmarks_both_protocols_into_one_report(
    served: tuple[FakeTritonServer, Path, TrtshipConfig], tmp_path: Path
) -> None:
    _, config_path, _ = served
    out = tmp_path / "served.json"
    result = runner.invoke(
        app,
        [
            "benchmark", "triton", str(config_path), "--protocol", "http", "--protocol", "grpc",
            "--precision", "fp16", "--output", str(out),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Server-side means per request" in result.output
    data = json.loads(out.read_text())
    assert data["subject"]["kind"] == "triton"
    assert data["subject"]["path"].endswith("model.plan")
    assert data["subject"]["precision"] == "fp16"
    assert {m["backend"] for m in data["measurements"]} == {"triton-http", "triton-grpc"}
    assert all(m["server_side"]["requests"] == ITERS for m in data["measurements"])


def test_cli_refuses_to_overwrite_and_rejects_unknown_protocols(
    served: tuple[FakeTritonServer, Path, TrtshipConfig], tmp_path: Path
) -> None:
    _, config_path, _ = served
    existing = tmp_path / "exists.json"
    existing.write_text("{}")
    clash = runner.invoke(app, ["benchmark", "triton", str(config_path), "-o", str(existing)])
    assert clash.exit_code != 0
    assert existing.read_text() == "{}"
    bad = runner.invoke(app, ["benchmark", "triton", str(config_path), "--protocol", "smtp"])
    assert bad.exit_code != 0
    assert "http or grpc" in bad.output


def test_cli_reports_a_server_that_is_not_running(
    served: tuple[FakeTritonServer, Path, TrtshipConfig], tmp_path: Path
) -> None:
    server, config_path, _ = served
    server.state.ready = False
    result = runner.invoke(app, ["benchmark", "triton", str(config_path)])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_cli_overhead_pairs_saved_reports(tmp_path: Path) -> None:
    direct = report("engine", [measurement("tensorrt", 3.0)])
    served_report = report("triton", [measurement("triton-http", 5.0, side=SIDE)])
    a, b = tmp_path / "direct.json", tmp_path / "served.json"
    a.write_text(direct.model_dump_json())
    b.write_text(served_report.model_dump_json())

    human = runner.invoke(app, ["benchmark", "overhead", str(a), str(b)])
    assert human.exit_code == 0, human.output
    assert "Serving overhead" in human.output
    assert "derived" in human.output

    machine = runner.invoke(app, ["benchmark", "overhead", str(a), str(b), "--json"])
    payload: dict[str, Any] = json.loads(machine.output)
    assert payload["rows"][0]["overhead_ms"] == pytest.approx(2.0)
