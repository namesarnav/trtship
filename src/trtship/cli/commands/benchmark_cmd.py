"""`trtship benchmark ...`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from trtship.benchmark import benchmark_engines, benchmark_onnx, torch_gpu_used_mb
from trtship.cli.guard import handle_errors
from trtship.cli.render import console, emit_json
from trtship.config import Precision, load_config
from trtship.errors import ArtifactError, ConfigError
from trtship.models import load_model
from trtship.reporting import (
    compare_reports,
    load_benchmark_reports,
    render_benchmark,
    render_comparison,
)
from trtship.tensorrt import TensorRTExecutor
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
