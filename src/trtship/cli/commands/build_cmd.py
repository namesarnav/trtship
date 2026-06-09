"""`trtship build`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from trtship.cli.render import console, emit_json
from trtship.config import Precision, load_config
from trtship.models import infer_signature, load_model
from trtship.tensorrt import build_engine
from trtship.utils import env
from trtship.utils.units import format_bytes


def build_command(
    config_path: Annotated[Path, typer.Argument(metavar="CONFIG", help="trtship YAML config.")],
    onnx_path: Annotated[Path, typer.Argument(metavar="MODEL.onnx", help="ONNX model to build.")],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Directory for the .plan files.")
    ],
    precision: Annotated[
        list[Precision] | None,
        typer.Option("--precision", help="Build only this precision (repeatable)."),
    ] = None,
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="Override a value: section.key=value.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print results as JSON.")] = False,
) -> None:
    """Build TensorRT engines from an ONNX model. Needs an NVIDIA GPU and TensorRT."""
    config = load_config(config_path, overrides=overrides or [])
    env.require_tensorrt(purpose="building TensorRT engines")
    model = load_model(config.model)
    signature = infer_signature(model, config.model, config.tensorrt.profiles, seed=config.seed)
    results = []
    for chosen in precision or config.tensorrt.precisions:
        target = output_dir / f"{model.name}.{chosen.value}.plan"
        results.append(build_engine(onnx_path, signature, config, chosen, target))
    if json_output:
        emit_json([r.model_dump(mode="json") for r in results])
        return
    for result in results:
        console.print(
            f"[green]built[/] {result.path}  {format_bytes(result.size_bytes)}  "
            f"{result.precision.value}  TensorRT {result.tensorrt_version}  {result.build_time_s}s"
        )
        for warning in result.warnings:
            console.print(f"  [yellow]warning:[/] {warning}", highlight=False)
