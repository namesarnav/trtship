"""trtship command-line entry point."""

from __future__ import annotations

from typing import Annotated

import typer

from trtship.cli.commands import config_cmd, doctor, inspect_cmd, version
from trtship.cli.guard import handle_errors
from trtship.logging import configure_logging

app = typer.Typer(
    name="trtship",
    help="Turn a PyTorch model into a benchmarked, calibrated, Triton-servable TensorRT engine.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback()
def _root(
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            envvar="TRTSHIP_LOG_LEVEL",
            help="DEBUG, INFO, WARNING or ERROR.",
            case_sensitive=False,
        ),
    ] = "WARNING",
    json_logs: Annotated[
        bool, typer.Option("--json-logs", help="Emit logs as JSON lines on stderr.")
    ] = False,
) -> None:
    try:
        configure_logging(level=log_level, json_logs=json_logs)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--log-level") from exc


app.command("version", help="Show the trtship version.")(handle_errors(version.version_command))
app.command("doctor", help="Detect the Python, CUDA, TensorRT, ONNX and Docker environment.")(
    handle_errors(doctor.doctor_command)
)
app.command("inspect", help="Report a model's architecture, parameters, memory and I/O.")(
    handle_errors(inspect_cmd.inspect_command)
)
app.add_typer(config_cmd.config_app, name="config", help="Inspect and validate configuration.")


def main() -> None:
    app()
