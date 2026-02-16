"""Environment and capability detection.

Probing never raises: a missing or broken component is reported as data. Code that needs a
capability calls a ``require_*`` function, which raises :class:`EnvironmentUnavailableError`
with an actionable hint. GPU work never falls back to CPU silently.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import platform
import shutil
import sys
import warnings
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from trtship import __version__
from trtship.errors import CommandError, EnvironmentUnavailableError
from trtship.utils.git import GitInfo, git_info
from trtship.utils.subprocess import CommandResult, run_command
from trtship.utils.timeutil import utc_now

SCHEMA_VERSION: Final = 1
MIN_PYTHON: Final = (3, 11)

# Capability names (stable identifiers used by `doctor --require`).
PYTHON: Final = "python"
TORCH: Final = "torch"
TORCH_CUDA: Final = "torch_cuda"
NVIDIA_GPU: Final = "nvidia_gpu"
CUDA_TOOLKIT: Final = "cuda_toolkit"
TENSORRT: Final = "tensorrt"
ONNX: Final = "onnx"
ONNXRUNTIME: Final = "onnxruntime"
TRITON_CLIENT: Final = "tritonclient"
DOCKER: Final = "docker"
DOCKER_NVIDIA_RUNTIME: Final = "docker_nvidia_runtime"
TRITON_SERVER: Final = "tritonserver"

CORE_CAPABILITIES: Final = (PYTHON, TORCH, ONNX, ONNXRUNTIME)


class CapabilityStatus(StrEnum):
    OK = "ok"
    MISSING = "missing"
    ERROR = "error"  # installed or present, but not usable


class Capability(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    status: CapabilityStatus
    version: str | None = None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is CapabilityStatus.OK


class GpuDevice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int
    name: str
    compute_capability: str | None = None
    memory_mb: int | None = None
    driver_version: str | None = None


class EnvironmentReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = SCHEMA_VERSION
    captured_at: datetime
    trtship_version: str
    python_version: str
    platform: str
    capabilities: list[Capability]
    gpus: list[GpuDevice]
    git: GitInfo | None = None

    def capability(self, name: str) -> Capability:
        for cap in self.capabilities:
            if cap.name == name:
                return cap
        raise KeyError(name)

    def versions(self) -> dict[str, str | None]:
        """Version strings for the components the reproducibility record cares about."""
        return {cap.name: cap.version for cap in self.capabilities if cap.version}


# --------------------------------------------------------------------------- helpers


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _ok(name: str, version: str | None = None, detail: str | None = None) -> Capability:
    return Capability(name=name, status=CapabilityStatus.OK, version=version, detail=detail)


def _missing(name: str, detail: str) -> Capability:
    return Capability(name=name, status=CapabilityStatus.MISSING, detail=detail)


def _error(name: str, detail: str, version: str | None = None) -> Capability:
    return Capability(name=name, status=CapabilityStatus.ERROR, version=version, detail=detail)


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


# --------------------------------------------------------------------------- probes


def probe_python() -> Capability:
    version = platform.python_version()
    if sys.version_info < MIN_PYTHON:
        needed = ".".join(map(str, MIN_PYTHON))
        return _error(PYTHON, f"Python >= {needed} required", version)
    return _ok(PYTHON, version, sys.executable)


def probe_torch() -> Capability:
    version = _distribution_version("torch")
    if version is None:
        return _missing(TORCH, "torch is not installed")
    return _ok(TORCH, version)


def probe_torch_cuda() -> Capability:
    """Whether the installed torch build can actually use a CUDA device."""
    if _distribution_version("torch") is None:
        return _missing(TORCH_CUDA, "torch is not installed")
    try:
        torch = importlib.import_module("torch")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            available = bool(torch.cuda.is_available())
        cuda_build = getattr(torch.version, "cuda", None)
    except Exception as exc:
        return _error(TORCH_CUDA, f"could not query torch CUDA state: {exc}")
    if available:
        count = int(torch.cuda.device_count())
        return _ok(TORCH_CUDA, cuda_build, f"{count} CUDA device(s) visible to torch")
    if cuda_build is None:
        return _missing(TORCH_CUDA, "torch was built without CUDA (CPU wheel)")
    reason = _first_line(str(caught[0].message)) if caught else "torch.cuda.is_available() is False"
    return _error(TORCH_CUDA, reason, cuda_build)


_SMI_QUERY_FULL = "index,name,compute_cap,memory.total,driver_version"
_SMI_QUERY_LEGACY = "index,name,memory.total,driver_version"


def parse_nvidia_smi_csv(text: str, *, has_compute_cap: bool = True) -> list[GpuDevice]:
    """Parse ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits`` output."""
    devices: list[GpuDevice] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        expected = 5 if has_compute_cap else 4
        if len(fields) != expected:
            raise ValueError(f"unexpected nvidia-smi row ({len(fields)} fields): {line!r}")
        if has_compute_cap:
            index, name, capability, memory, driver = fields
        else:
            (index, name, memory, driver), capability = fields, ""
        devices.append(
            GpuDevice(
                index=int(index),
                name=name,
                compute_capability=capability or None,
                memory_mb=int(float(memory)) if memory.replace(".", "", 1).isdigit() else None,
                driver_version=driver or None,
            )
        )
    return devices


def _query_nvidia_smi() -> tuple[list[GpuDevice], Capability]:
    try:
        result = _smi(_SMI_QUERY_FULL)
        has_cap = True
        if not result.ok and "valid field" in result.output.lower():
            result, has_cap = _smi(_SMI_QUERY_LEGACY), False
    except CommandError as exc:
        return [], _missing(NVIDIA_GPU, exc.message)
    if not result.ok:
        return [], _error(NVIDIA_GPU, _first_line(result.output) or "nvidia-smi failed")
    try:
        devices = parse_nvidia_smi_csv(result.stdout, has_compute_cap=has_cap)
    except ValueError as exc:
        return [], _error(NVIDIA_GPU, str(exc))
    if not devices:
        return [], _missing(NVIDIA_GPU, "nvidia-smi reported no devices")
    driver = devices[0].driver_version
    summary = ", ".join(
        f"{d.name} (sm_{(d.compute_capability or '?').replace('.', '')})" for d in devices
    )
    return devices, _ok(NVIDIA_GPU, driver, summary)


def _smi(query: str) -> CommandResult:
    return run_command(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], timeout=15
    )


def probe_nvidia_gpu() -> tuple[list[GpuDevice], Capability]:
    return _query_nvidia_smi()


def probe_cuda_toolkit() -> Capability:
    try:
        result = run_command(["nvcc", "--version"], timeout=15)
    except CommandError:
        return _missing(CUDA_TOOLKIT, "nvcc not found (the CUDA toolkit is optional at runtime)")
    if not result.ok:
        return _error(CUDA_TOOLKIT, _first_line(result.output))
    for line in result.stdout.splitlines():
        if "release" in line:
            release = line.split("release", 1)[1].split(",")[0].strip()
            return _ok(CUDA_TOOLKIT, release)
    return _ok(CUDA_TOOLKIT, None, _first_line(result.stdout))


def probe_tensorrt() -> Capability:
    version = _distribution_version("tensorrt", "tensorrt-cu12", "tensorrt-cu13", "tensorrt_lean")
    try:
        module = importlib.import_module("tensorrt")
    except ImportError as exc:
        if version is None:
            return _missing(TENSORRT, "TensorRT Python bindings are not installed")
        return _error(TENSORRT, f"installed ({version}) but failed to import: {exc}", version)
    except Exception as exc:
        return _error(TENSORRT, f"import failed: {exc}", version)
    return _ok(TENSORRT, str(getattr(module, "__version__", version or "unknown")))


def probe_onnx() -> Capability:
    version = _distribution_version("onnx")
    return _ok(ONNX, version) if version else _missing(ONNX, "onnx is not installed")


def probe_onnxruntime() -> Capability:
    version = _distribution_version("onnxruntime", "onnxruntime-gpu")
    if version is None:
        return _missing(ONNXRUNTIME, "onnxruntime is not installed")
    try:
        ort = importlib.import_module("onnxruntime")
        providers = ", ".join(ort.get_available_providers())
    except Exception as exc:
        return _error(ONNXRUNTIME, f"installed but failed to import: {exc}", version)
    return _ok(ONNXRUNTIME, version, f"providers: {providers}")


def probe_tritonclient() -> Capability:
    version = _distribution_version("tritonclient")
    if version is None:
        return _missing(
            TRITON_CLIENT, "tritonclient is not installed (pip install 'trtship[triton]')"
        )
    missing = [
        extra
        for extra, module in (("http", "geventhttpclient"), ("grpc", "grpc"))
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        return _error(
            TRITON_CLIENT,
            f"tritonclient installed without: {', '.join(missing)} (install tritonclient[all])",
            version,
        )
    return _ok(TRITON_CLIENT, version)


def probe_docker() -> tuple[Capability, Capability]:
    """Return (docker, docker_nvidia_runtime)."""
    no_runtime = _missing(DOCKER_NVIDIA_RUNTIME, "docker is not usable")
    if shutil.which("docker") is None:
        return _missing(DOCKER, "docker CLI not found"), no_runtime
    try:
        info = run_command(["docker", "info", "--format", "{{json .Runtimes}}"], timeout=20)
    except CommandError as exc:
        return _error(DOCKER, exc.message), no_runtime
    if not info.ok:
        return _error(DOCKER, _first_line(info.output) or "docker daemon unreachable"), no_runtime
    try:
        version = run_command(["docker", "version", "--format", "{{.Server.Version}}"], timeout=20)
    except CommandError:
        version = None
    docker = _ok(DOCKER, version.stdout.strip() if version and version.ok else None)
    if '"nvidia"' in info.stdout:
        return docker, _ok(DOCKER_NVIDIA_RUNTIME, None, "nvidia runtime registered with Docker")
    return docker, _missing(
        DOCKER_NVIDIA_RUNTIME,
        "the nvidia runtime is not registered; install and configure the NVIDIA Container Toolkit",
    )


def probe_tritonserver() -> Capability:
    binary = shutil.which("tritonserver")
    if binary is None:
        return _missing(
            TRITON_SERVER,
            "no local tritonserver binary (trtship serves Triton through its Docker image)",
        )
    return _ok(TRITON_SERVER, None, binary)


def probe_all(*, repo_dir: Path | None = None) -> EnvironmentReport:
    """Probe every capability and return a JSON-serializable snapshot."""
    gpus, gpu_capability = probe_nvidia_gpu()
    docker, docker_nvidia = probe_docker()
    capabilities = [
        probe_python(),
        probe_torch(),
        probe_torch_cuda(),
        gpu_capability,
        probe_cuda_toolkit(),
        probe_tensorrt(),
        probe_onnx(),
        probe_onnxruntime(),
        probe_tritonclient(),
        docker,
        docker_nvidia,
        probe_tritonserver(),
    ]
    return EnvironmentReport(
        captured_at=utc_now(),
        trtship_version=__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        capabilities=capabilities,
        gpus=gpus,
        git=git_info(repo_dir or Path.cwd()),
    )


# --------------------------------------------------------------------------- requirements

_HINTS: Final[dict[str, str]] = {
    NVIDIA_GPU: "Run `nvidia-smi` to check the driver; a driver/library mismatch is fixed by "
    "rebooting or reloading the nvidia kernel modules.",
    TORCH_CUDA: "Install a CUDA build of PyTorch and verify `nvidia-smi` works.",
    TENSORRT: "Install TensorRT: `uv sync --extra trt` (requires an NVIDIA GPU environment).",
    DOCKER: "Install Docker and make sure the daemon is running and accessible.",
    DOCKER_NVIDIA_RUNTIME: "Install the NVIDIA Container Toolkit and run "
    "`sudo nvidia-ctk runtime configure --runtime=docker`, then restart Docker.",
    TRITON_CLIENT: "Install the Triton client: `uv sync --extra triton`.",
}


def require(name: str, *, purpose: str) -> Capability:
    """Assert that capability ``name`` is usable, or raise with an actionable error."""
    probes: dict[str, Callable[[], Capability]] = {
        NVIDIA_GPU: lambda: probe_nvidia_gpu()[1],
        TORCH_CUDA: probe_torch_cuda,
        TENSORRT: probe_tensorrt,
        TRITON_CLIENT: probe_tritonclient,
        DOCKER: lambda: probe_docker()[0],
        DOCKER_NVIDIA_RUNTIME: lambda: probe_docker()[1],
    }
    if name not in probes:
        raise KeyError(f"no requirement probe registered for {name!r}")
    capability = probes[name]()
    if not capability.ok:
        raise EnvironmentUnavailableError(
            f"{purpose} requires {name}, which is {capability.status.value}: {capability.detail}",
            hint=_HINTS.get(name),
            details={"capability": name, "status": capability.status.value},
        )
    return capability


def require_gpu(*, purpose: str) -> Capability:
    """The NVIDIA driver stack must expose a device; there is no CPU fallback.

    Stages that also move tensors with torch should additionally ``require(TORCH_CUDA, ...)``.
    """
    return require(NVIDIA_GPU, purpose=purpose)


def require_tensorrt(*, purpose: str) -> Capability:
    require_gpu(purpose=purpose)
    return require(TENSORRT, purpose=purpose)
