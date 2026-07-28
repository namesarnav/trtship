"""`trtship serve|stop|status`, with the Triton server object replaced by a recording fake."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from trtship.cli.main import app
from trtship.config import TritonConfig
from trtship.errors import EnvironmentUnavailableError, TritonError
from trtship.triton.server import ModelState, ServerStatus
from trtship.utils import env

runner = CliRunner()


class RecordingServer:
    instances: list[RecordingServer] = []  # noqa: RUF012

    def __init__(self, settings: TritonConfig) -> None:
        self.settings = settings
        self.served: Path | None = None
        RecordingServer.instances.append(self)

    def serve(self, repository: Path) -> ServerStatus:
        self.served = repository
        return self.status()

    def stop(self) -> bool:
        return self.settings.container_name == "running-one"

    def status(self) -> ServerStatus:
        return ServerStatus(
            container=self.settings.container_name,
            state="running",
            image=self.settings.image,
            ready=True,
            models=[ModelState(name="m", ready=True)],
            http_url="http://127.0.0.1:8000",
            grpc_endpoint="127.0.0.1:8001",
            metrics_url="http://127.0.0.1:8002/metrics",
        )


@pytest.fixture(autouse=True)
def fake_server(monkeypatch: pytest.MonkeyPatch) -> None:
    RecordingServer.instances = []
    monkeypatch.setattr("trtship.cli.commands.server_cmd.TritonServer", RecordingServer)
    monkeypatch.setattr(env, "require", lambda name, purpose: None)


def config_file(tmp_path: Path, **triton: Any) -> Path:
    data = {
        "model": {
            "name": "m",
            "kind": "module",
            "factory": "trtship_fixtures.models:tiny_mlp",
            "inputs": [{"name": "x", "shape": ["batch", 16]}],
        },
        "triton": {"repository_dir": str(tmp_path / "repo"), "image": "img:1", **triton},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_serve_uses_the_configured_repository_by_default(tmp_path: Path) -> None:
    result = runner.invoke(app, ["serve", str(config_file(tmp_path))])
    assert result.exit_code == 0, result.output
    assert RecordingServer.instances[0].served == tmp_path / "repo"
    assert "model     m: ready" in result.output
    assert "127.0.0.1:8001" in result.output


def test_serve_accepts_a_repository_and_prints_json(tmp_path: Path) -> None:
    other = tmp_path / "other"
    result = runner.invoke(
        app, ["serve", str(config_file(tmp_path)), "--repository", str(other), "--json"]
    )
    assert result.exit_code == 0, result.output
    assert RecordingServer.instances[0].served == other
    payload = json.loads(result.output)
    assert payload["ready"] is True
    assert payload["models"] == [{"name": "m", "ready": True}]


def test_stop_reports_whether_anything_was_running(tmp_path: Path) -> None:
    nothing = runner.invoke(app, ["stop", str(config_file(tmp_path))])
    assert nothing.exit_code == 0
    assert "is not running" in nothing.output
    stopped = runner.invoke(
        app, ["stop", str(config_file(tmp_path, container_name="running-one")), "--json"]
    )
    assert json.loads(stopped.output) == {"container": "running-one", "stopped": True}


def test_status_prints_the_state(tmp_path: Path) -> None:
    result = runner.invoke(app, ["status", str(config_file(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "container trtship-triton: running" in result.output
    assert "image     img:1" in result.output


def test_a_missing_docker_stops_the_command_with_the_environment_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(name: str, purpose: str) -> None:
        raise EnvironmentUnavailableError(f"{purpose} requires {name}, which is missing")

    monkeypatch.setattr(env, "require", refuse)
    result = runner.invoke(app, ["serve", str(config_file(tmp_path))])
    assert result.exit_code == EnvironmentUnavailableError.exit_code
    assert not RecordingServer.instances


def test_server_failures_keep_the_triton_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(self: RecordingServer, repository: Path) -> ServerStatus:
        raise TritonError("model m failed to load", hint="Last container output:\nboom")

    monkeypatch.setattr(RecordingServer, "serve", fail)
    result = runner.invoke(app, ["serve", str(config_file(tmp_path))])
    assert result.exit_code == TritonError.exit_code
    assert "model m failed to load" in result.output
    assert "boom" in result.output
