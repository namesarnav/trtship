"""`trtship benchmark ...`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from trtship.benchmark import (
    BenchmarkReport,
    benchmark_engines,
    benchmark_onnx,
    benchmark_triton,
    torch_gpu_used_mb,
)
from trtship.cli.guard import handle_errors
from trtship.cli.render import console, emit_json
from trtship.config import Precision, load_config
from trtship.errors import ArtifactError, ConfigError, TritonError
from trtship.models import load_model
from trtship.reporting import (
    compare_reports,
    compare_serving,
    load_benchmark_reports,
    render_benchmark,
    render_comparison,
    render_serving,
)
from trtship.tensorrt import TensorRTExecutor
from trtship.triton import TritonClient, wait_until_ready
from trtship.triton.client import Protocol
from trtship.utils import env
from trtship.utils.fs import atomic_write_json

benchmark_app = typer.Typer(
    no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False
)


def _finish(
    report_json: dict[str, object], as_json: bool, output: Path | None, force: bool
) -> None:
    if output is not None:
        if output.exists() and not force:
            raise ArtifactError(
                f"refusing to overwrite existing file: {output}",
                hint="Pass --force to overwrite it.",
            )
        atomic_write_json(output, report_json)
    if as_json:
        emit_json(report_json)


@benchmark_app.command("onnx")
@handle_errors
def benchmark_onnx_command(
    config_path: Annotated[Path, typer.Argument(metavar="CONFIG", help="trtship YAML config.")],
    onnx_path: Annotated[Path, typer.Argument(metavar="MODEL.onnx", help="ONNX model.")],
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="Override a value: section.key=value.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Also write the JSON report here.")
    ] = None,
    force: Annotated[bool, typer.Option(help="Overwrite --output if it exists.")] = False,
) -> None:
    """Benchmark an ONNX model on ONNX Runtime's CPU provider (real measurements, CPU-labelled)."""
    config = load_config(config_path, overrides=overrides or [])
    if output is not None and output.exists() and not force:
        raise ArtifactError(
            f"refusing to overwrite existing file: {output}", hint="Pass --force to overwrite it."
        )
    model = load_model(config.model)
    report = benchmark_onnx(onnx_path, model, config, env.probe_all())
    _finish(report.model_dump(mode="json"), json_output, output, force)
    if not json_output:
        render_benchmark(report, console)


def _parse_engine(spec: str) -> tuple[Precision, Path]:
    precision_text, separator, path_text = spec.partition(":")
    if not separator or not path_text or precision_text not in {p.value for p in Precision}:
        raise ConfigError(
            f"invalid --engine {spec!r}", hint="Use PRECISION:PATH, e.g. fp16:build/m.fp16.plan"
        )
    return Precision(precision_text), Path(path_text)


@benchmark_app.command("engine")
@handle_errors
def benchmark_engine_command(
    config_path: Annotated[Path, typer.Argument(metavar="CONFIG", help="trtship YAML config.")],
    engines: Annotated[
        list[str],
        typer.Option("--engine", help="PRECISION:PATH of a TensorRT plan (repeatable)."),
    ],
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="Override a value: section.key=value.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Also write the JSON report here.")
    ] = None,
    force: Annotated[bool, typer.Option(help="Overwrite --output if it exists.")] = False,
) -> None:
    """Benchmark TensorRT engines directly (needs a GPU, TensorRT, and a CUDA build of PyTorch)."""
    config = load_config(config_path, overrides=overrides or [])
    parsed = [_parse_engine(spec) for spec in engines]
    if output is not None and output.exists() and not force:
        raise ArtifactError(
            f"refusing to overwrite existing file: {output}", hint="Pass --force to overwrite it."
        )
    env.require_tensorrt(purpose="benchmarking TensorRT engines")
    env.require(env.TORCH_CUDA, purpose="benchmarking TensorRT engines")
    model = load_model(config.model)
    report = benchmark_engines(
        parsed,
        model,
        config,
        env.probe_all(),
        executor_factory=TensorRTExecutor,
        gpu_used_mb=lambda: torch_gpu_used_mb(config.tensorrt.device_index),
    )
    _finish(report.model_dump(mode="json"), json_output, output, force)
    if not json_output:
        render_benchmark(report, console)


@benchmark_app.command("compare")
@handle_errors
def benchmark_compare_command(
    run_a: Annotated[Path, typer.Argument(metavar="RUN_A", help="Run directory or report JSON.")],
    run_b: Annotated[Path, typer.Argument(metavar="RUN_B", help="Run directory or report JSON.")],
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the comparison as JSON.")
    ] = False,
) -> None:
    """Compare the benchmark results of two runs (measurements matched by backend/shape)."""
    comparison = compare_reports(load_benchmark_reports(run_a), load_benchmark_reports(run_b))
    if json_output:
        emit_json(comparison.model_dump(mode="json"))
    else:
        render_comparison(comparison, console, run_a.name, run_b.name)


def _protocols(chosen: list[str]) -> list[Protocol]:
    protocols: list[Protocol] = []
    for name in chosen or ["http"]:
        if name not in ("http", "grpc"):
            raise ConfigError(f"--protocol must be http or grpc, not {name!r}")
        protocol: Protocol = "http" if name == "http" else "grpc"
        if protocol not in protocols:
            protocols.append(protocol)
    return protocols


@benchmark_app.command("triton")
@handle_errors
def benchmark_triton_command(
    config_path: Annotated[Path, typer.Argument(metavar="CONFIG", help="trtship YAML config.")],
    protocols: Annotated[
        list[str] | None,
        typer.Option("--protocol", help="http or grpc (repeatable). Default: http."),
    ] = None,
    precision: Annotated[
        Precision | None,
        typer.Option("--precision", help="Precision of the served engine, recorded in the report."),
    ] = None,
    repository: Annotated[
        Path | None,
        typer.Option("--repository", "-r", help="Repository being served. Default: triton.*."),
    ] = None,
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="Override a value: section.key=value.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Also write the JSON report here.")
    ] = None,
    force: Annotated[bool, typer.Option(help="Overwrite --output if it exists.")] = False,
) -> None:
    """Benchmark the model served by a running Triton server (`trtship serve` first).

    Measures through the real client API and reads the server's own queue and compute statistics
    around each timed section. The GPU and tool versions recorded describe this machine, which may
    not be the server's host.
    """
    chosen = _protocols(protocols or [])
    config = load_config(config_path, overrides=overrides or [])
    if output is not None and output.exists() and not force:
        raise ArtifactError(
            f"refusing to overwrite existing file: {output}", hint="Pass --force to overwrite it."
        )
    env.require(env.TRITON_CLIENT, purpose="benchmarking a Triton deployment")
    name = config.model.name
    root = repository or config.triton.repository_dir
    plan = root / name / str(config.triton.model_version) / "model.plan"
    served_precision = precision or config.triton.precision
    model = load_model(config.model)
    environment = env.probe_all()

    reports: list[BenchmarkReport] = []
    for protocol in chosen:

        def open_client(protocol: Protocol = protocol) -> TritonClient:
            return TritonClient.from_settings(config.triton, protocol)

        probe = open_client()
        try:
            wait_until_ready(probe, name, timeout_s=10.0)
        except TritonError:
            probe.close()
            raise
        probe.close()
        reports.append(
            benchmark_triton(
                open_client,
                model,
                config,
                environment,
                protocol=protocol,
                precision=served_precision,
                endpoint=probe.url,
                plan=plan if plan.is_file() else None,
            )
        )
    report = reports[0].model_copy(
        update={
            "measurements": [m for r in reports for m in r.measurements],
            "skipped": [s for r in reports for s in r.skipped],
        }
    )
    _finish(report.model_dump(mode="json"), json_output, output, force)
    if not json_output:
        render_benchmark(report, console)


@benchmark_app.command("overhead")
@handle_errors
def benchmark_overhead_command(
    direct: Annotated[
        Path, typer.Argument(metavar="DIRECT", help="Run directory or report JSON (tensorrt).")
    ],
    served: Annotated[
        Path, typer.Argument(metavar="SERVED", help="Run directory or report JSON (triton).")
    ],
    json_output: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
) -> None:
    """Show what serving through Triton costs over running the same engine directly."""
    comparison = compare_serving(load_benchmark_reports(direct), load_benchmark_reports(served))
    if json_output:
        emit_json(comparison.model_dump(mode="json"))
    else:
        render_serving(comparison, console)
