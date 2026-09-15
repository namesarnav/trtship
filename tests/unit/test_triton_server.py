"""Triton server lifecycle, with Docker and HTTP replaced by scripted stand-ins.

These prove trtship's orchestration: the docker commands it builds, how it waits for readiness, and
that failures leave nothing behind. They do not prove a real Triton container starts.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from trtship.config import TritonConfig
from trtship.errors import ConfigError, TritonError
from trtship.triton.server import (
    HttpResponse,
    TritonServer,
    build_run_argv,
    repository_models,
    urllib_transport,
)
from trtship.utils.subprocess import CommandResult

IMAGE = "nvcr.io/nvidia/tritonserver:24.08-py3"


def settings(**overrides: Any) -> TritonConfig:
    return TritonConfig.model_validate({"image": IMAGE, **overrides})


def result(stdout: str = "", returncode: int = 0, stderr: str = "") -> CommandResult:
    return CommandResult(("docker",), returncode, stdout, stderr)


class FakeDocker:
    """Answers docker commands from a mutable container state and records every call."""

    def __init__(self, state: str = "absent") -> None:
        self.state = state
        self.calls: list[list[str]] = []
        self.run_result = result("container-id\n")
        self.logs = "I0919 model m failed to load"
        self.states_after_run: list[str] = []  # popped one per inspect, then sticks

    def __call__(self, argv: Sequence[str], *, timeout: float = 30.0) -> CommandResult:
        argv = list(argv)
        self.calls.append(argv)
        verb = argv[1]
        if verb == "inspect" and "{{.State.Status}}" in argv:
            if self.state == "absent":
                return result(stderr="Error: No such object: trtship-triton", returncode=1)
            current = self.state
            if self.states_after_run:
                current = self.state = self.states_after_run.pop(0)
            return result(current + "\n")
        if verb == "inspect":
            return result(IMAGE + "\n")
        if verb == "run":
            if self.run_result.ok:
                self.state = "running"
            return self.run_result
        if verb == "logs":
            return result(self.logs)
        if verb in {"stop", "rm"}:
            if verb == "rm":
                self.state = "absent"
            return result("trtship-triton\n")
        raise AssertionError(f"unexpected docker command {argv}")

    def verbs(self) -> list[str]:
        return [c[1] for c in self.calls]


class FakeHttp:
    def __init__(self, routes: Mapping[str, HttpResponse | Exception]) -> None:
        self.routes = routes
        self.requests: list[tuple[str, str]] = []

    def __call__(self, method: str, url: str, timeout: float) -> HttpResponse:
        self.requests.append((method, url))
        response = self.routes.get(url, HttpResponse(404, b""))
        if isinstance(response, Exception):
            raise response
        return response


def ok() -> HttpResponse:
    return HttpResponse(200, b"")


BASE = "http://127.0.0.1:8000"
READY = {f"{BASE}/v2/health/ready": ok(), f"{BASE}/v2/models/m/ready": ok()}


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    (tmp_path / "repo/m/1").mkdir(parents=True)
    (tmp_path / "repo/m/config.pbtxt").write_text('name: "m"\n')
    return tmp_path / "repo"


def server(
    docker: FakeDocker,
    http: FakeHttp,
    cfg: TritonConfig | None = None,
    clock: Callable[[], float] | None = None,
) -> tuple[TritonServer, list[float]]:
    sleeps: list[float] = []
    now = [0.0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    return (
        TritonServer(
            cfg or settings(),
            runner=docker,
            transport=http,
            sleep=sleep,
            clock=clock or (lambda: now[0]),
        ),
        sleeps,
    )


# ---------------------------------------------------------------- argv


def test_the_run_command_mounts_the_repository_read_only_and_binds_locally(
    repository: Path,
) -> None:
    argv = build_run_argv(settings(), repository)
    assert argv[:8] == [
        "docker",
        "run",
        "--detach",
        "--name",
        "trtship-triton",
        "--gpus",
        "all",
        "--publish",
    ]
    assert "127.0.0.1:8000:8000" in argv
    assert "127.0.0.1:8001:8001" in argv
    assert "127.0.0.1:8002:8002" in argv
    assert f"{repository.resolve()}:/models:ro" in argv
    image_at = argv.index(IMAGE)
    assert argv[image_at + 1 :] == [
        "tritonserver",
        "--model-repository=/models",
        "--strict-model-config=true",
        "--model-control-mode=none",
    ]


def test_ports_and_address_follow_the_settings(repository: Path) -> None:
    argv = build_run_argv(
        settings(http_port=9000, grpc_port=9001, metrics_port=9002, bind_address="0.0.0.0"),
        repository,
    )
    assert "0.0.0.0:9000:8000" in argv
    assert "0.0.0.0:9002:8002" in argv


def test_an_ipv6_bind_address_is_bracketed(repository: Path) -> None:
    argv = build_run_argv(settings(bind_address="::1"), repository)
    assert "[::1]:8000:8000" in argv


def test_the_image_has_no_default(repository: Path) -> None:
    with pytest.raises(ConfigError, match=r"triton\.image"):
        build_run_argv(TritonConfig(), repository)


def test_invalid_addresses_and_container_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="IPv4 or IPv6"):
        TritonConfig(bind_address="localhost")
    with pytest.raises(ValueError, match="container name"):
        TritonConfig(container_name="bad name; rm -rf /")


def test_repository_models_lists_models_and_rejects_empty_repositories(
    repository: Path, tmp_path: Path
) -> None:
    assert repository_models(repository) == ["m"]
    (tmp_path / "empty").mkdir()
    with pytest.raises(TritonError, match="no models"):
        repository_models(tmp_path / "empty")
    with pytest.raises(TritonError, match="not found"):
        repository_models(tmp_path / "absent")


# ---------------------------------------------------------------- serve


def test_serve_starts_the_container_and_reports_ready(repository: Path) -> None:
    docker = FakeDocker()
    http = FakeHttp(
        {
            **READY,
            f"{BASE}/v2/repository/index": HttpResponse(
                200, json.dumps([{"name": "m", "version": "1", "state": "READY"}]).encode()
            ),
        }
    )
    srv, _ = server(docker, http)
    status = srv.serve(repository)
    assert status.state == "running"
    assert status.ready is True
    assert [(m.name, m.ready) for m in status.models or []] == [("m", True)]
    assert status.grpc_endpoint == "127.0.0.1:8001"
    assert docker.verbs().count("run") == 1
    assert "rm" not in docker.verbs()  # a healthy server is left running


def test_serve_waits_until_the_model_is_ready(repository: Path) -> None:
    docker = FakeDocker()
    responses = {
        f"{BASE}/v2/health/ready": ok(),
        f"{BASE}/v2/models/m/ready": HttpResponse(400, b""),
    }
    http = FakeHttp(responses)
    srv, sleeps = server(docker, http)

    original = http.__call__

    def flip(method: str, url: str, timeout: float) -> HttpResponse:
        if len(sleeps) >= 3:
            responses[f"{BASE}/v2/models/m/ready"] = ok()
        return original(method, url, timeout)

    srv._http = flip
    srv.serve(repository)
    assert len(sleeps) == 3


def test_a_server_that_is_not_listening_yet_is_waited_for(repository: Path) -> None:
    docker = FakeDocker()
    http = FakeHttp({f"{BASE}/v2/health/ready": ConnectionRefusedError("refused")})
    srv, sleeps = server(docker, http, settings(startup_timeout_s=3))
    with pytest.raises(TritonError, match="did not become ready within 3s"):
        srv.serve(repository)
    assert sleeps == [1.0, 1.0, 1.0]


def test_a_container_that_exits_reports_its_logs_and_is_removed(repository: Path) -> None:
    docker = FakeDocker()
    docker.states_after_run = ["running", "exited"]
    srv, _ = server(docker, FakeHttp({f"{BASE}/v2/health/ready": HttpResponse(503, b"")}))
    with pytest.raises(TritonError, match="stopped during startup") as excinfo:
        srv.serve(repository)
    assert "model m failed to load" in (excinfo.value.hint or "")
    assert docker.verbs()[-1] == "rm"
    assert docker.state == "absent"


def test_a_timeout_removes_the_container(repository: Path) -> None:
    docker = FakeDocker()
    srv, _ = server(docker, FakeHttp({}), settings(startup_timeout_s=2))
    with pytest.raises(TritonError, match="did not become ready"):
        srv.serve(repository)
    assert docker.state == "absent"


def test_an_existing_container_is_not_replaced(repository: Path) -> None:
    docker = FakeDocker(state="running")
    srv, _ = server(docker, FakeHttp(READY))
    with pytest.raises(TritonError, match="already exists"):
        srv.serve(repository)
    assert "run" not in docker.verbs()
    assert "rm" not in docker.verbs()  # the user's container is untouched


def test_a_failed_docker_run_is_reported_without_cleanup_of_others(repository: Path) -> None:
    docker = FakeDocker()
    docker.run_result = result(
        stderr="docker: Error response: port is already allocated", returncode=125
    )
    srv, _ = server(docker, FakeHttp({}))
    with pytest.raises(TritonError, match="port is already allocated"):
        srv.serve(repository)


def test_serve_rejects_a_missing_repository_before_touching_docker(tmp_path: Path) -> None:
    docker = FakeDocker()
    srv, _ = server(docker, FakeHttp({}))
    with pytest.raises(TritonError, match="not found"):
        srv.serve(tmp_path / "nope")
    assert docker.calls == []


# ---------------------------------------------------------------- stop and status


def test_stop_stops_and_removes(repository: Path) -> None:
    docker = FakeDocker(state="running")
    srv, _ = server(docker, FakeHttp({}))
    assert srv.stop() is True
    assert docker.verbs() == ["inspect", "stop", "rm"]


def test_stop_with_no_container_is_a_no_op() -> None:
    docker = FakeDocker()
    srv, _ = server(docker, FakeHttp({}))
    assert srv.stop() is False
    assert docker.verbs() == ["inspect"]


def test_status_of_a_missing_container() -> None:
    srv, _ = server(FakeDocker(), FakeHttp({}))
    status = srv.status()
    assert (status.state, status.ready, status.models) == ("absent", None, None)


def test_status_of_a_stopped_container_does_not_query_the_server() -> None:
    http = FakeHttp({})
    srv, _ = server(FakeDocker(state="exited"), http)
    status = srv.status()
    assert status.state == "exited"
    assert status.ready is None
    assert http.requests == []


def test_status_reports_an_unreachable_server_as_unknown() -> None:
    http = FakeHttp({f"{BASE}/v2/health/ready": OSError("connection refused")})
    srv, _ = server(FakeDocker(state="running"), http)
    status = srv.status()
    assert status.ready is None
    assert "did not answer" in status.notes[0]


def test_status_notes_a_model_index_that_is_unavailable() -> None:
    http = FakeHttp(
        {f"{BASE}/v2/health/ready": ok(), f"{BASE}/v2/repository/index": HttpResponse(400, b"")}
    )
    srv, _ = server(FakeDocker(state="running"), http)
    status = srv.status()
    assert status.ready is True
    assert status.models is None
    assert "HTTP 400" in status.notes[0]


def test_status_lists_models_that_are_not_ready() -> None:
    index = json.dumps([{"name": "a", "state": "READY"}, {"name": "b", "state": "UNAVAILABLE"}])
    http = FakeHttp(
        {
            f"{BASE}/v2/health/ready": HttpResponse(503, b""),
            f"{BASE}/v2/repository/index": HttpResponse(200, index.encode()),
        }
    )
    srv, _ = server(FakeDocker(state="running"), http)
    status = srv.status()
    assert status.ready is False
    assert [(m.name, m.ready) for m in status.models or []] == [("a", True), ("b", False)]


# ---------------------------------------------------------------- HTTP transport


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        code = 200 if self.path == "/v2/health/ready" else 404
        self.send_response(code)
        self.end_headers()
        self.wfile.write(b"body")

    def do_POST(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"[]")

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def stub_server() -> Iterator[str]:
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_the_real_transport_returns_status_codes_over_http(stub_server: str) -> None:
    assert urllib_transport("GET", f"{stub_server}/v2/health/ready", 2.0).status == 200
    missing = urllib_transport("GET", f"{stub_server}/nope", 2.0)
    assert (missing.status, missing.body) == (404, b"body")
    assert urllib_transport("POST", f"{stub_server}/v2/repository/index", 2.0).body == b"[]"


def test_the_real_transport_raises_oserror_when_nothing_listens() -> None:
    with pytest.raises(OSError):  # noqa: PT011
        urllib_transport("GET", "http://127.0.0.1:1/v2/health/ready", 1.0)


def test_the_real_transport_only_speaks_plain_http() -> None:
    with pytest.raises(ValueError, match="plain http"):
        urllib_transport("GET", "file:///etc/passwd", 1.0)


@pytest.mark.parametrize(
    "image", ["--privileged", "-v", "nvcr.io/x:1 --privileged", "image;rm", "", "a b"]
)
def test_an_image_that_could_be_read_as_a_docker_option_is_rejected(image: str) -> None:
    with pytest.raises(ValueError, match="image reference"):
        TritonConfig(image=image)


@pytest.mark.parametrize(
    "image",
    [
        "nvcr.io/nvidia/tritonserver:24.12-py3",
        "localhost:5000/team/triton:dev",
        "triton@sha256:" + "a" * 64,
    ],
)
def test_ordinary_image_references_are_accepted(image: str) -> None:
    assert TritonConfig(image=image).image == image


def test_a_repository_path_that_would_alter_the_mount_spec_is_refused(tmp_path: Path) -> None:
    hostile = tmp_path / "repo:rw"
    hostile.mkdir()
    with pytest.raises(ConfigError, match="contains ':'"):
        build_run_argv(settings(), hostile)
