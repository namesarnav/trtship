"""trtship command-line entry point."""

from __future__ import annotations

from typing import Annotated

import typer

from trtship.cli.commands import (
    build_cmd,
    config_cmd,
    doctor,
    export_cmd,
    init_cmd,
    inspect_cmd,
    optimize_cmd,
    report_cmd,
    run_cmd,
    validate_cmd,
    version,
)
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
app.command("export", help="Export the model to ONNX and verify the graph.")(
    handle_errors(export_cmd.export_command)
)
app.command("init", help="Write a starter configuration.")(handle_errors(init_cmd.init_command))
app.command("run", help="Run the pipeline (inspect, export, validate, optimize, ...).")(
    handle_errors(run_cmd.run_command)
)
app.command("report", help="Summarize a run.")(handle_errors(report_cmd.report_command))
app.command("build", help="Build TensorRT engines from an ONNX model (needs a GPU).")(
    handle_errors(build_cmd.build_command)
)
app.command("optimize", help="Write an optimized copy of an ONNX model.")(
    handle_errors(optimize_cmd.optimize_command)
)
app.add_typer(config_cmd.config_app, name="config", help="Inspect and validate configuration.")
app.add_typer(validate_cmd.validate_app, name="validate", help="Validate exported artifacts.")


def main() -> None:
    app()
