"""The Docker files are consistent with each other and with what `trtship serve` runs.

Nothing here builds an image or needs Docker; it checks the files as text and as YAML.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from trtship.config import TritonConfig
from trtship.triton.server import CONTAINER_REPOSITORY, build_run_argv

DOCKER = Path(__file__).resolve().parents[2] / "docker"
DOCKERFILE = (DOCKER / "Dockerfile").read_text("utf-8")


def load(name: str) -> dict[str, Any]:
    data = yaml.safe_load((DOCKER / name).read_text("utf-8"))
    assert isinstance(data, dict)
    return data


def test_every_compose_build_target_exists_in_the_dockerfile() -> None:
    stages = set(re.findall(r"^FROM .* AS (\w+)$", DOCKERFILE, re.MULTILINE))
    assert stages == {"dev", "runtime"}
    for service in load("compose.yml")["services"].values():
        assert service["build"]["target"] in stages
        assert (DOCKER / service["build"]["dockerfile"].removeprefix("docker/")).is_file()


def test_the_triton_compose_file_starts_triton_the_way_trtship_serve_does() -> None:
    repository = Path("/repo")
    argv = build_run_argv(TritonConfig(image="img:1", bind_address="127.0.0.1"), repository)
    served_command = argv[argv.index("tritonserver") :]

    service = load("compose.triton.yml")["services"]["triton"]
    assert service["command"] == served_command
    assert service["volumes"][0].endswith(f":{CONTAINER_REPOSITORY}:ro")
    assert argv[argv.index("--volume") + 1].endswith(f":{CONTAINER_REPOSITORY}:ro")
    # Published ports default to loopback and to the same container ports.
    assert all(p.startswith("${BIND_ADDRESS:-127.0.0.1}:") for p in service["ports"])
    assert [p.rsplit(":", 1)[1] for p in service["ports"]] == ["8000", "8001", "8002"]


def test_the_triton_image_has_no_default() -> None:
    image = load("compose.triton.yml")["services"]["triton"]["image"]
    assert image.startswith("${TRITON_IMAGE:?")


def test_gpu_services_reserve_nvidia_devices() -> None:
    compose = load("compose.yml")["services"]["trtship"]
    triton = load("compose.triton.yml")["services"]["triton"]
    for service in (compose, triton):
        (device,) = service["deploy"]["resources"]["reservations"]["devices"]
        assert device["driver"] == "nvidia"
        assert "gpu" in device["capabilities"]


def test_the_runtime_image_does_not_run_as_root() -> None:
    runtime = DOCKERFILE.split("AS runtime")[1]
    assert re.search(r"^USER (?!root)\S+", runtime, re.MULTILINE)


def test_the_build_context_excludes_weights_engines_and_the_brief() -> None:
    ignored = (DOCKER.parent / ".dockerignore").read_text("utf-8").split()
    assert {"*.plan", "*.onnx", "*.pt", "*.pth", "runs", "project.md", ".git"} <= set(ignored)
