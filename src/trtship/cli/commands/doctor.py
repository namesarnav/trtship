"""`trtship doctor`: report what this machine can and cannot run."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from trtship.cli.render import console, emit_json
from trtship.errors import ConfigError, EnvironmentUnavailableError
from trtship.utils import env
from trtship.utils.env import Capability, CapabilityStatus, EnvironmentReport

_STYLE = {
    CapabilityStatus.OK: "[green]ok[/]",
    CapabilityStatus.MISSING: "[yellow]missing[/]",
    CapabilityStatus.ERROR: "[bold red]error[/]",
}

_KNOWN = (
    env.PYTHON,
    env.TORCH,
    env.TORCH_CUDA,
    env.NVIDIA_GPU,
    env.CUDA_TOOLKIT,
    env.TENSORRT,
    env.ONNX,
    env.ONNXRUNTIME,
    env.TRITON_CLIENT,
    env.DOCKER,
    env.DOCKER_NVIDIA_RUNTIME,
    env.TRITON_SERVER,
)

# What each part of the pipeline needs. "Ready" means every listed capability is ok.
_READINESS: dict[str, tuple[str, ...]] = {
    "ONNX export and validation (CPU)": (env.PYTHON, env.TORCH, env.ONNX, env.ONNXRUNTIME),
    "TensorRT build, calibration, benchmark": (env.NVIDIA_GPU, env.TENSORRT),
    "Triton serving via Docker": (env.DOCKER, env.DOCKER_NVIDIA_RUNTIME),
    "Triton client": (env.TRITON_CLIENT,),
}


def readiness(report: EnvironmentReport) -> dict[str, bool]:
    return {
        area: all(report.capability(name).ok for name in needs)
        for area, needs in _READINESS.items()
    }


def _render(report: EnvironmentReport) -> None:
    table = Table(title="trtship doctor", title_justify="left", show_lines=False)
    for column in ("Component", "Status", "Version", "Detail"):
        table.add_column(column, overflow="fold")
    for cap in report.capabilities:
        table.add_row(cap.name, _STYLE[cap.status], cap.version or "-", escape(cap.detail or ""))
    console.print(table)
    if report.gpus:
        gpus = Table(title="GPUs", title_justify="left")
        for column in ("Index", "Name", "Compute capability", "Memory (MiB)", "Driver"):
            gpus.add_column(column)
        for gpu in report.gpus:
            gpus.add_row(
                str(gpu.index),
                gpu.name,
                gpu.compute_capability or "-",
                str(gpu.memory_mb or "-"),
                gpu.driver_version or "-",
            )
        console.print(gpus)
    console.print("[bold]Pipeline readiness[/]")
    for area, ready in readiness(report).items():
        console.print(f"  {'[green]ready  [/]' if ready else '[yellow]blocked[/]'}  {area}")


def _failures(report: EnvironmentReport, required: list[str]) -> list[Capability]:
    names = [*env.CORE_CAPABILITIES, *required]
    seen: set[str] = set()
    failed = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        cap = report.capability(name)
        if not cap.ok:
            failed.append(cap)
    return failed


def doctor_command(
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    require: Annotated[
        list[str] | None,
        typer.Option(
            "--require",
            help=f"Also fail (exit 3) unless this capability is ok. One of: {', '.join(_KNOWN)}.",
        ),
    ] = None,
) -> None:
    required = require or []
    unknown = [name for name in required if name not in _KNOWN]
    if unknown:
        raise ConfigError(
            f"unknown capability: {', '.join(unknown)}",
            hint=f"Known capabilities: {', '.join(_KNOWN)}",
        )
    report = env.probe_all()
    if json_output:
        emit_json({**report.model_dump(mode="json"), "readiness": readiness(report)})
    else:
        _render(report)
    failed = _failures(report, required)
    if failed:
        summary = "; ".join(f"{cap.name}: {cap.status.value} ({cap.detail})" for cap in failed)
        raise EnvironmentUnavailableError(
            f"required capabilities are not available: {summary}",
            details={"failed": [cap.name for cap in failed]},
        )
