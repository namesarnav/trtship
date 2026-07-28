"""`trtship serve`, `trtship stop`, `trtship status`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from trtship.cli.render import console, emit_json
from trtship.config import load_config
from trtship.triton.server import ServerStatus, TritonServer
from trtship.utils import env

_Config = Annotated[Path, typer.Argument(metavar="CONFIG", help="trtship YAML config.")]
_Overrides = Annotated[
    list[str] | None, typer.Option("--set", help="Override a value: section.key=value.")
]
_Json = Annotated[bool, typer.Option("--json", help="Print results as JSON.")]


def render_status(status: ServerStatus) -> None:
    color = {"running": "green", "absent": "dim"}.get(status.state, "yellow")
    console.print(f"container {status.container}: [{color}]{status.state}[/]", highlight=False)
    if status.image:
        console.print(f"image     {status.image}", highlight=False)
    if status.ready is not None:
        console.print(f"ready     {'yes' if status.ready else 'no'}")
    for model in status.models or []:
        console.print(f"model     {model.name}: {'ready' if model.ready else 'not ready'}")
    if status.http_url:
        console.print(f"http      {status.http_url}", highlight=False)
        console.print(f"grpc      {status.grpc_endpoint}", highlight=False)
        console.print(f"metrics   {status.metrics_url}", highlight=False)
    for note in status.notes:
        console.print(f"[yellow]note:[/] {note}", highlight=False)


def serve_command(
    config_path: _Config,
    repository: Annotated[
        Path | None,
        typer.Option(
            "--repository", "-r", help="Model repository. Default: triton.repository_dir."
        ),
    ] = None,
    overrides: _Overrides = None,
    json_output: _Json = False,
) -> None:
    """Start Triton in Docker on a model repository and wait until it is ready."""
    config = load_config(config_path, overrides=overrides or [])
    env.require(env.DOCKER, purpose="serving with Triton")
    env.require(env.DOCKER_NVIDIA_RUNTIME, purpose="serving TensorRT engines with Triton")
    server = TritonServer(config.triton)
    status = server.serve(repository or config.triton.repository_dir)
    if json_output:
        emit_json(status.model_dump(mode="json"))
    else:
        render_status(status)


def stop_command(
    config_path: _Config, overrides: _Overrides = None, json_output: _Json = False
) -> None:
    """Stop and remove the Triton container."""
    config = load_config(config_path, overrides=overrides or [])
    env.require(env.DOCKER, purpose="stopping the Triton container")
    stopped = TritonServer(config.triton).stop()
    if json_output:
        emit_json({"container": config.triton.container_name, "stopped": stopped})
    elif stopped:
        console.print(f"[green]stopped[/] {config.triton.container_name}", highlight=False)
    else:
        console.print(f"container {config.triton.container_name} is not running", highlight=False)


def status_command(
    config_path: _Config, overrides: _Overrides = None, json_output: _Json = False
) -> None:
    """Show the Triton container's state and whether the server and its models are ready."""
    config = load_config(config_path, overrides=overrides or [])
    env.require(env.DOCKER, purpose="inspecting the Triton container")
    status = TritonServer(config.triton).status()
    if json_output:
        emit_json(status.model_dump(mode="json"))
    else:
        render_status(status)
