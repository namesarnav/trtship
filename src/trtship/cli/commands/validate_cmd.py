"""`trtship validate ...`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from trtship.cli.guard import handle_errors
from trtship.cli.render import console, emit_json
from trtship.config import Precision, load_config
from trtship.errors import ArtifactError, TritonError
from trtship.models import infer_signature, load_model
from trtship.onnx import validate_onnx
from trtship.reporting import render_engine_validation
from trtship.reporting.validation_report import render_onnx_validation
from trtship.tensorrt import TensorRTExecutor
from trtship.triton import TritonClient, TritonExecutor, wait_until_ready
from trtship.triton.client import Protocol
from trtship.utils import env
from trtship.utils.fs import atomic_write_json
from trtship.validation.engine import EngineUnderTest, validate_engines

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


@validate_app.command("engine")
@handle_errors
def validate_engine_command(
    config_path: Annotated[Path, typer.Argument(metavar="CONFIG", help="trtship YAML config.")],
    engine_path: Annotated[Path, typer.Argument(metavar="ENGINE.plan", help="TensorRT plan.")],
    onnx_path: Annotated[
        Path, typer.Option("--onnx", help="The ONNX model the engine was built from.")
    ],
    precision: Annotated[
        Precision, typer.Option("--precision", help="Precision the engine was built at.")
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
    """Compare a TensorRT engine with PyTorch and ONNX Runtime. Exits 6 on failure. Needs a GPU."""
    config = load_config(config_path, overrides=overrides or [])
    if output is not None and output.exists() and not force:
        raise ArtifactError(
            f"refusing to overwrite existing file: {output}", hint="Pass --force to overwrite it."
        )
    env.require_tensorrt(purpose="validating a TensorRT engine")
    env.require(env.TORCH_CUDA, purpose="validating a TensorRT engine")
    model = load_model(config.model)
    signature = infer_signature(model, config.model, config.tensorrt.profiles, seed=config.seed)
    report = validate_engines(
        [EngineUnderTest(precision=precision, path=str(engine_path))],
        onnx_path,
        model,
        signature,
        config,
        TensorRTExecutor,
    )
    payload = report.model_dump(mode="json")
    if output is not None:
        atomic_write_json(output, payload)
    if json_output:
        emit_json(payload)
    else:
        render_engine_validation(report, console)
    report.raise_for_failure()


@validate_app.command("triton")
@handle_errors
def validate_triton_command(
    config_path: Annotated[Path, typer.Argument(metavar="CONFIG", help="trtship YAML config.")],
    onnx_path: Annotated[
        Path, typer.Option("--onnx", help="The ONNX model the served engine was built from.")
    ],
    precision: Annotated[
        Precision, typer.Option("--precision", help="Precision of the served engine.")
    ],
    protocol: Annotated[
        str, typer.Option("--protocol", help="Client protocol: http or grpc.")
    ] = "http",
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
    """Send the validation inputs through a running Triton server and compare with PyTorch.

    Uses the same shape points and per-precision tolerances as `validate engine`. Exits 6 on
    failure. The server must already be running (`trtship serve`).
    """
    if protocol not in ("http", "grpc"):
        raise ArtifactError(f"--protocol must be http or grpc, not {protocol!r}")
    chosen: Protocol = "http" if protocol == "http" else "grpc"
    config = load_config(config_path, overrides=overrides or [])
    if output is not None and output.exists() and not force:
        raise ArtifactError(
            f"refusing to overwrite existing file: {output}", hint="Pass --force to overwrite it."
        )
    env.require(env.TRITON_CLIENT, purpose="validating a Triton deployment")
    name = config.model.name
    root = repository or config.triton.repository_dir
    plan = root / name / str(config.triton.model_version) / "model.plan"
    if not plan.is_file():
        raise TritonError(
            f"the served plan was not found at {plan}",
            hint="Pass --repository with the model repository that is being served.",
        )
    model = load_model(config.model)
    signature = infer_signature(model, config.model, config.tensorrt.profiles, seed=config.seed)

    def open_executor(_: Path) -> TritonExecutor:
        client = TritonClient.from_settings(config.triton, chosen)
        try:
            wait_until_ready(client, name, timeout_s=10.0)
        except BaseException:
            client.close()
            raise
        return TritonExecutor(client, name)

    report = validate_engines(
        [EngineUnderTest(precision=precision, path=str(plan))],
        onnx_path,
        model,
        signature,
        config,
        open_executor,
        backend="triton-http" if chosen == "http" else "triton-grpc",
    )
    payload = report.model_dump(mode="json")
    if output is not None:
        atomic_write_json(output, payload)
    if json_output:
        emit_json(payload)
    else:
        render_engine_validation(report, console)
    report.raise_for_failure()
