"""`trtship validate ...`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from trtship.cli.guard import handle_errors
from trtship.cli.render import console, emit_json
from trtship.config import load_config
from trtship.errors import ArtifactError
from trtship.models import infer_signature, load_model
from trtship.onnx import validate_onnx
from trtship.reporting.validation_report import render_onnx_validation
from trtship.utils.fs import atomic_write_json

validate_app = typer.Typer(
    no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False
)


@validate_app.command("onnx")
@handle_errors
def validate_onnx_command(
    config_path: Annotated[Path, typer.Argument(metavar="CONFIG", help="trtship YAML config.")],
    onnx_path: Annotated[Path, typer.Argument(metavar="MODEL.onnx", help="ONNX model to check.")],
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="Override a value: section.key=value.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Also write the JSON report here.")
    ] = None,
    force: Annotated[bool, typer.Option(help="Overwrite --output if it exists.")] = False,
) -> None:
    """Check an ONNX model and compare it with the PyTorch model. Exits 6 on a failed check."""
    config = load_config(config_path, overrides=overrides or [])
    if output is not None and output.exists() and not force:
        raise ArtifactError(
            f"refusing to overwrite existing file: {output}", hint="Pass --force to overwrite it."
        )
    model = load_model(config.model)
    signature = infer_signature(model, config.model, config.tensorrt.profiles, seed=config.seed)
    report = validate_onnx(onnx_path, model, signature, config)
    payload = report.model_dump(mode="json")
    if output is not None:
        atomic_write_json(output, payload)
    if json_output:
        emit_json(payload)
    else:
        render_onnx_validation(report, console)
        if output is not None:
            console.print(f"\nwrote {output}")
    report.raise_for_failure()
