"""Running Triton Inference Server in a Docker container.

Every Docker call is an argv list (never a shell string), and every failure carries the container's
last log lines, because that is where Triton says why a model did not load.

The Docker runner and the HTTP transport are injected so the orchestration (argument construction,
readiness waiting, failure cleanup) is tested without Docker. What has *not* been exercised is a
real Triton container: that needs a GPU, the NVIDIA container runtime, and the server image.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from trtship.config import TritonConfig
from trtship.errors import ConfigError, TritonError
from trtship.logging import get_logger
from trtship.utils.subprocess import CommandResult, run_command

log = get_logger(__name__)

CONTAINER_REPOSITORY = "/models"
_LOG_TAIL_LINES = 60
_POLL_INTERVAL_S = 1.0

Runner = Callable[..., CommandResult]


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes


HttpTransport = Callable[[str, str, float], HttpResponse]
"""(method, url, timeout) -> response. Raises OSError when the server cannot be reached."""


def urllib_transport(method: str, url: str, timeout: float) -> HttpResponse:
    if not url.startswith("http://"):
        raise ValueError(f"only plain http URLs are supported: {url}")
    request = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None)  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return HttpResponse(response.status, response.read())
    except urllib.error.HTTPError as exc:
        return HttpResponse(exc.code, exc.read())
    except urllib.error.URLError as exc:
        raise OSError(str(exc.reason)) from exc


class ModelState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    ready: bool


class ServerStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    container: str
    state: str  # Docker's container state, or "absent"
    image: str | None = None
    ready: bool | None = None  # None when the server could not be asked
    models: list[ModelState] | None = None
    http_url: str | None = None
    grpc_endpoint: str | None = None
    metrics_url: str | None = None
    notes: list[str] = Field(default_factory=list)


def repository_models(repository: Path) -> list[str]:
    """Names of the models in a Triton repository (directories holding a ``config.pbtxt``)."""
    if not repository.is_dir():
        raise TritonError(
            f"model repository not found: {repository}",
            hint="Create one with `trtship package`, or pass --repository.",
        )
    models = sorted(p.parent.name for p in repository.glob("*/config.pbtxt"))
    if not models:
        raise TritonError(
            f"no models in repository {repository} (expected <model>/config.pbtxt)",
            hint="Create one with `trtship package`.",
        )
    return models


def probe_host(settings: TritonConfig) -> str:
    """The address to reach the published ports on from this machine."""
    if settings.bind_address in {"0.0.0.0", "::"}:  # noqa: S104
        return "127.0.0.1"
    return f"[{settings.bind_address}]" if ":" in settings.bind_address else settings.bind_address


def http_url(settings: TritonConfig) -> str:
    return f"http://{probe_host(settings)}:{settings.http_port}"


def build_run_argv(settings: TritonConfig, repository: Path) -> list[str]:
    """The ``docker run`` command that starts Triton on ``repository`` (read-only)."""
    if settings.image is None:
        raise ConfigError(
            "triton.image is not set",
            hint="Set triton.image to the Triton release that ships the TensorRT version that "
            "built your engine, e.g. nvcr.io/nvidia/tritonserver:<xx.yy>-py3.",
        )
    bind = f"[{settings.bind_address}]" if ":" in settings.bind_address else settings.bind_address
    ports = [
        f"{bind}:{settings.http_port}:8000",
        f"{bind}:{settings.grpc_port}:8001",
        f"{bind}:{settings.metrics_port}:8002",
    ]
    argv = ["docker", "run", "--detach", "--name", settings.container_name, "--gpus", "all"]
    for port in ports:
        argv += ["--publish", port]
    argv += [
        "--volume",
        f"{repository.resolve()}:{CONTAINER_REPOSITORY}:ro",
        settings.image,
        "tritonserver",
        f"--model-repository={CONTAINER_REPOSITORY}",
        "--strict-model-config=true",
        "--model-control-mode=none",
    ]
    return argv


class TritonServer:
    """Start, stop, and inspect one Triton container."""

    def __init__(
        self,
        settings: TritonConfig,
        *,
        runner: Runner = run_command,
        transport: HttpTransport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self._run = runner
        self._http = transport
        self._sleep = sleep
        self._clock = clock

    # ------------------------------------------------------------------ docker helpers

    def _docker(self, argv: Sequence[str], *, timeout: float = 60.0) -> CommandResult:
        return self._run(list(argv), timeout=timeout)

    def container_state(self) -> str:
        """Docker's state of the container (``running``, ``exited``, ...), or ``absent``."""
        name = self.settings.container_name
        result = self._docker(["docker", "inspect", "--format", "{{.State.Status}}", name])
        if not result.ok:
            if "no such" in result.output.lower():
                return "absent"
            raise TritonError(
                f"docker inspect failed for {name}: {result.output or 'no output'}",
                hint="Check that the Docker daemon is running and accessible.",
            )
        return result.stdout.strip() or "absent"

    def logs_tail(self) -> str:
        result = self._docker(
            ["docker", "logs", "--tail", str(_LOG_TAIL_LINES), self.settings.container_name]
        )
        return result.output

    def _remove(self) -> None:
        self._docker(["docker", "rm", "--force", self.settings.container_name])

    # ------------------------------------------------------------------ serve

    def serve(self, repository: Path) -> ServerStatus:
        """Start the container and return once the server and every model report ready.

        On any failure the container is removed after its logs are captured in the error, so a
        failed ``serve`` never leaves a half-started server holding the ports.
        """
        models = repository_models(repository)
        argv = build_run_argv(self.settings, repository)
        state = self.container_state()
        if state != "absent":
            raise TritonError(
                f"container {self.settings.container_name!r} already exists ({state})",
                hint="Run `trtship stop` first, or set triton.container_name.",
            )
        log.info("starting Triton: %s", " ".join(argv))
        started = self._docker(argv, timeout=300.0)  # a first run may need to pull the image
        if not started.ok:
            raise TritonError(
                f"docker could not start the container: {started.output or 'no output'}",
                hint="Check the image name and access, the port mappings, and that the NVIDIA "
                "container runtime is configured (`trtship doctor`).",
            )
        try:
            self._wait_until_ready(models)
        except BaseException:
            self._remove()
            raise
        return self.status()

    def _wait_until_ready(self, models: list[str]) -> None:
        deadline = self._clock() + self.settings.startup_timeout_s
        base = http_url(self.settings)
        while True:
            state = self.container_state()
            if state != "running":
                raise TritonError(
                    f"the Triton container stopped during startup (state: {state})",
                    hint=self._logs_hint(),
                )
            if self._ready(base, models):
                return
            if self._clock() >= deadline:
                raise TritonError(
                    f"Triton did not become ready within {self.settings.startup_timeout_s:g}s",
                    hint=self._logs_hint(),
                )
            self._sleep(_POLL_INTERVAL_S)

    def _ready(self, base: str, models: list[str]) -> bool:
        try:
            if self._http("GET", f"{base}/v2/health/ready", 5.0).status != 200:
                return False
            return all(
                self._http("GET", f"{base}/v2/models/{m}/ready", 5.0).status == 200 for m in models
            )
        except OSError:
            return False  # not listening yet

    def _logs_hint(self) -> str:
        return "Last container output:\n" + (self.logs_tail() or "(no output)")

    # ------------------------------------------------------------------ stop

    def stop(self) -> bool:
        """Stop and remove the container. Returns False if there was nothing to stop."""
        if self.container_state() == "absent":
            return False
        name = self.settings.container_name
        stopped = self._docker(["docker", "stop", "--time", "30", name], timeout=90.0)
        if not stopped.ok and "no such" not in stopped.output.lower():
            raise TritonError(
                f"docker could not stop {name}: {stopped.output or 'no output'}",
                hint="Inspect it with `docker ps -a`.",
            )
        removed = self._docker(["docker", "rm", "--force", name])
        if not removed.ok and "no such" not in removed.output.lower():
            raise TritonError(f"docker could not remove {name}: {removed.output or 'no output'}")
        return True

    # ------------------------------------------------------------------ status

    def status(self) -> ServerStatus:
        settings = self.settings
        state = self.container_state()
        if state == "absent":
            return ServerStatus(container=settings.container_name, state="absent")
        image = self._docker(
            ["docker", "inspect", "--format", "{{.Config.Image}}", settings.container_name]
        ).stdout.strip()
        base = http_url(settings)
        notes: list[str] = []
        ready: bool | None = None
        models: list[ModelState] | None = None
        if state == "running":
            try:
                ready = self._http("GET", f"{base}/v2/health/ready", 5.0).status == 200
                models = self._models(base, notes)
            except OSError as exc:
                notes.append(f"the server did not answer on {base}: {exc}")
        return ServerStatus(
            container=settings.container_name,
            state=state,
            image=image or None,
            ready=ready,
            models=models,
            http_url=base,
            grpc_endpoint=f"{probe_host(settings)}:{settings.grpc_port}",
            metrics_url=f"http://{probe_host(settings)}:{settings.metrics_port}/metrics",
            notes=notes,
        )

    def _models(self, base: str, notes: list[str]) -> list[ModelState] | None:
        response = self._http("POST", f"{base}/v2/repository/index", 5.0)
        if response.status != 200:
            notes.append(f"model index unavailable (HTTP {response.status})")
            return None
        try:
            index: list[dict[str, Any]] = json.loads(response.body)
            return [ModelState(name=e["name"], ready=e.get("state") == "READY") for e in index]
        except (ValueError, KeyError, TypeError) as exc:
            notes.append(f"could not read the model index: {exc}")
            return None
