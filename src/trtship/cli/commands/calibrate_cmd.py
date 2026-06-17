"""`trtship calibrate`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from trtship.calibration import calibrate
from trtship.cli.render import console, emit_json
from trtship.config import load_config
from trtship.models import infer_signature, load_model
from trtship.utils import env


def calibrate_command(
    config_path: Annotated[Path, typer.Argument(metavar="CONFIG", help="trtship YAML config.")],
    onnx_path: Annotated[Path, typer.Argument(metavar="MODEL.onnx", help="ONNX model.")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Calibration cache directory to create.")
    ],
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="Override a value: section.key=value.")
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the metadata as JSON.")
    ] = False,
) -> None:
    """Run INT8 calibration and write a calibration cache directory (needs a GPU and TensorRT)."""
    config = load_config(config_path, overrides=overrides or [])
    env.require_tensorrt(purpose="INT8 calibration")
    env.require(env.TORCH_CUDA, purpose="INT8 calibration")
    model = load_model(config.model)
    signature = infer_signature(model, config.model, config.tensorrt.profiles, seed=config.seed)
    metadata = calibrate(
        onnx_path, signature, config, output, model_weights_sha256=model.weights_sha256
    )
    if json_output:
        emit_json(metadata.model_dump(mode="json"))
        return
    console.print(f"[green]calibrated[/] {output}")
    console.print(
        f"  data     {metadata.dataset.name} ({metadata.dataset.items} items, "
        f"{metadata.sample_count} used in {metadata.num_batches} batches of {metadata.batch_size})"
    )
    console.print(f"  method   {metadata.method}, TensorRT {metadata.tensorrt_version}")
    console.print(f"  cache    sha256 {metadata.cache_sha256}")
    if not metadata.representative:
        console.print(
            "[yellow]warning:[/] synthetic data is not representative; INT8 accuracy is unreliable"
        )
