"""`trtship export`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from trtship.cli.render import console, emit_json
from trtship.config import load_config
from trtship.export import export_onnx
from trtship.models import infer_signature, load_model
from trtship.utils.units import format_bytes


def export_command(
    config_path: Annotated[Path, typer.Argument(metavar="CONFIG", help="trtship YAML config.")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to write the ONNX model (must not exist).")
    ],
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="Override a value: section.key=value.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
) -> None:
    """Export the model to ONNX and verify the graph against the model's signature."""
    config = load_config(config_path, overrides=overrides or [])
    model = load_model(config.model)
    signature = infer_signature(model, config.model, config.tensorrt.profiles, seed=config.seed)
    result = export_onnx(model, signature, config, output)
    if json_output:
        emit_json(result.model_dump(mode="json"))
        return
    meta = result.metadata
    console.print(f"[green]exported[/] {result.path}")
    console.print(f"  size      {format_bytes(result.size_bytes)}")
    console.print(f"  sha256    {result.sha256}")
    console.print(f"  opset     {meta.opset} ({meta.exporter} exporter, IR {meta.ir_version})")
    console.print(f"  inputs    {', '.join(meta.input_names)}")
    console.print(f"  outputs   {', '.join(meta.output_names)}")
    if meta.dynamic_axes:
        console.print(f"  dynamic   {meta.dynamic_axes}")
    for warning in meta.warnings:
        console.print(f"[yellow]warning:[/] {warning}", highlight=False, markup=False)
