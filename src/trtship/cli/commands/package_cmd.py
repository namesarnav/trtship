"""`trtship package`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from trtship.cli.render import console, emit_json
from trtship.config import load_config
from trtship.errors import TritonError
from trtship.triton import build_repository, load_engine_info
from trtship.utils.units import format_bytes


def package_command(
    config_path: Annotated[Path, typer.Argument(metavar="CONFIG", help="trtship YAML config.")],
    engine: Annotated[Path, typer.Argument(metavar="MODEL.plan", help="TensorRT engine to serve.")],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="New repository directory. Default: triton.repository_dir."
        ),
    ] = None,
    engine_info: Annotated[
        Path | None,
        typer.Option(
            "--engine-info",
            help="JSON with the engine's tensors (default: MODEL.plan.json, written by `build`).",
        ),
    ] = None,
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="Override a value: section.key=value.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print results as JSON.")] = False,
) -> None:
    """Create a Triton model repository from a built engine. Needs no GPU."""
    config = load_config(config_path, overrides=overrides or [])
    info_path = engine_info or engine.with_name(engine.name + ".json")
    if not info_path.is_file():
        raise TritonError(
            f"no engine metadata at {info_path}",
            hint="Pass --engine-info, or build the engine with `trtship build` (it writes "
            "MODEL.plan.json next to the plan).",
        )
    info = load_engine_info(info_path)
    destination = output or config.triton.repository_dir
    result = build_repository(destination, config.model.name, engine, info, config.triton)
    if json_output:
        emit_json(result.model_dump(mode="json"))
        return
    console.print(f"[green]created[/] {result.repository}", highlight=False)
    console.print(
        f"model {result.model_name}  version {result.version}  max_batch_size "
        f"{result.max_batch_size}  plan {format_bytes(engine.stat().st_size)}",
        highlight=False,
    )
    console.print(f"sha256 {result.repository_sha256}")
