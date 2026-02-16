from __future__ import annotations

import importlib
import importlib.util
import subprocess
import types
import warnings
from pathlib import Path

import pytest

from trtship.errors import CommandError, EnvironmentUnavailableError
from trtship.utils import env
from trtship.utils.git import git_info
from trtship.utils.subprocess import CommandResult


def _result(stdout: str = "", stderr: str = "", code: int = 0) -> CommandResult:
    return CommandResult(("cmd",), code, stdout, stderr)


GOOD_SMI = "0, NVIDIA A10G, 8.6, 23028, 550.54.15\n1, NVIDIA A10G, 8.6, 23028, 550.54.15\n"


def test_parse_nvidia_smi_csv_full() -> None:
    devices = env.parse_nvidia_smi_csv(GOOD_SMI)
    assert [d.index for d in devices] == [0, 1]
    assert devices[0].name == "NVIDIA A10G"
    assert devices[0].compute_capability == "8.6"
    assert devices[0].memory_mb == 23028
    assert devices[0].driver_version == "550.54.15"


def test_parse_nvidia_smi_csv_legacy_has_no_compute_capability() -> None:
    devices = env.parse_nvidia_smi_csv("0, Tesla K80, 11441, 470.1\n", has_compute_cap=False)
    assert devices[0].compute_capability is None
    assert devices[0].memory_mb == 11441


def test_parse_nvidia_smi_csv_rejects_malformed_rows() -> None:
    with pytest.raises(ValueError, match="unexpected nvidia-smi row"):
        env.parse_nvidia_smi_csv("0, only-two-fields\n")


def test_gpu_probe_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "run_command", lambda *a, **k: _result(GOOD_SMI))
    devices, cap = env.probe_nvidia_gpu()
    assert cap.ok
    assert cap.version == "550.54.15"
    assert "sm_86" in (cap.detail or "")
    assert len(devices) == 2


def test_gpu_probe_reports_driver_mismatch_as_error_not_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Failed to initialize NVML: Driver/library version mismatch\nNVML library version: 580.1"
    monkeypatch.setattr(env, "run_command", lambda *a, **k: _result(stdout=text, code=18))
    devices, cap = env.probe_nvidia_gpu()
    assert devices == []
    assert cap.status is env.CapabilityStatus.ERROR
    assert cap.detail == "Failed to initialize NVML: Driver/library version mismatch"


def test_gpu_probe_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*a: object, **k: object) -> CommandResult:
        raise CommandError("executable not found: nvidia-smi")

    monkeypatch.setattr(env, "run_command", missing)
    _, cap = env.probe_nvidia_gpu()
    assert cap.status is env.CapabilityStatus.MISSING


def test_gpu_probe_falls_back_when_compute_cap_field_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake(argv: list[str], **_: object) -> CommandResult:
        calls.append(argv[1])
        if "compute_cap" in argv[1]:
            return _result(stdout='Field "compute_cap" is not a valid field to query.', code=2)
        return _result("0, Old GPU, 4096, 470.1\n")

    monkeypatch.setattr(env, "run_command", fake)
    devices, cap = env.probe_nvidia_gpu()
    assert cap.ok
    assert devices[0].compute_capability is None
    assert len(calls) == 2


def test_docker_probe_detects_nvidia_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trtship.utils.env.shutil.which", lambda name: "/usr/bin/docker")

    def fake(argv: list[str], **_: object) -> CommandResult:
        if argv[1] == "info":
            return _result(
                '{"io.containerd.runc.v2":{},"nvidia":{"path":"nvidia-container-runtime"}}'
            )
        return _result("27.0.1")

    monkeypatch.setattr(env, "run_command", fake)
    docker, nvidia = env.probe_docker()
    assert docker.ok
    assert docker.version == "27.0.1"
    assert nvidia.ok


def test_docker_probe_without_nvidia_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trtship.utils.env.shutil.which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(env, "run_command", lambda *a, **k: _result('{"runc":{}}'))
    docker, nvidia = env.probe_docker()
    assert docker.ok
    assert nvidia.status is env.CapabilityStatus.MISSING
    assert "NVIDIA Container Toolkit" in (nvidia.detail or "")


def test_docker_probe_daemon_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trtship.utils.env.shutil.which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        env,
        "run_command",
        lambda *a, **k: _result(stderr="Cannot connect to the Docker daemon", code=1),
    )
    docker, nvidia = env.probe_docker()
    assert docker.status is env.CapabilityStatus.ERROR
    assert not nvidia.ok


def test_docker_probe_missing_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trtship.utils.env.shutil.which", lambda name: None)
    docker, _ = env.probe_docker()
    assert docker.status is env.CapabilityStatus.MISSING


def test_require_gpu_raises_actionable_error_and_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        env, "run_command", lambda *a, **k: _result("Failed to initialize NVML: x", code=9)
    )
    with pytest.raises(EnvironmentUnavailableError) as info:
        env.require_gpu(purpose="engine build")
    assert "engine build" in info.value.message
    assert info.value.hint is not None
    assert info.value.details["capability"] == "nvidia_gpu"
    assert info.value.exit_code == 3


def test_require_tensorrt_checks_gpu_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "run_command", lambda *a, **k: _result("no devices", code=9))
    with pytest.raises(EnvironmentUnavailableError) as info:
        env.require_tensorrt(purpose="build")
    assert info.value.details["capability"] == "nvidia_gpu"


def test_require_unknown_capability_is_a_programming_error() -> None:
    with pytest.raises(KeyError):
        env.require("no_such_capability", purpose="x")


def test_python_probe_ok() -> None:
    cap = env.probe_python()
    assert cap.ok
    assert cap.version is not None


def test_probe_all_reports_every_named_capability_and_serializes() -> None:
    report = env.probe_all()
    names = {c.name for c in report.capabilities}
    assert {
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
    } <= names
    assert report.capability(env.PYTHON).ok
    # JSON round trip
    restored = env.EnvironmentReport.model_validate_json(report.model_dump_json())
    assert restored == report
    assert env.PYTHON in report.versions()


# --------------------------------------------------------------------------- module probes


def _fake_import(monkeypatch: pytest.MonkeyPatch, **modules: object) -> None:
    """Make ``importlib.import_module`` return/raise for selected names; delegate the rest."""
    real = importlib.import_module

    def fake(name: str, package: str | None = None) -> object:
        if name in modules:
            value = modules[name]
            if isinstance(value, BaseException):
                raise value
            return value
        return real(name, package)

    monkeypatch.setattr("trtship.utils.env.importlib.import_module", fake)


def _dist_versions(monkeypatch: pytest.MonkeyPatch, versions: dict[str, str]) -> None:
    def lookup(*names: str) -> str | None:
        return next((versions[n] for n in names if n in versions), None)

    monkeypatch.setattr(env, "_distribution_version", lookup)


def test_tensorrt_probe_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _dist_versions(monkeypatch, {"tensorrt": "10.3.0"})
    _fake_import(monkeypatch, tensorrt=types.SimpleNamespace(__version__="10.3.0.26"))
    cap = env.probe_tensorrt()
    assert cap.ok
    assert cap.version == "10.3.0.26"  # the runtime's own version wins over package metadata


def test_tensorrt_probe_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _dist_versions(monkeypatch, {})
    _fake_import(monkeypatch, tensorrt=ImportError("No module named tensorrt"))
    assert env.probe_tensorrt().status is env.CapabilityStatus.MISSING


def test_tensorrt_probe_installed_but_unimportable_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dist_versions(monkeypatch, {"tensorrt": "10.3.0"})
    _fake_import(monkeypatch, tensorrt=ImportError("libnvinfer.so.10: cannot open shared object"))
    cap = env.probe_tensorrt()
    assert cap.status is env.CapabilityStatus.ERROR
    assert "libnvinfer" in (cap.detail or "")
    assert cap.version == "10.3.0"


def test_tensorrt_probe_survives_a_crashing_native_import(monkeypatch: pytest.MonkeyPatch) -> None:
    _dist_versions(monkeypatch, {"tensorrt": "10.3.0"})
    _fake_import(monkeypatch, tensorrt=OSError("driver too old"))
    assert env.probe_tensorrt().status is env.CapabilityStatus.ERROR


def test_torch_cuda_probe_available(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2),
        version=types.SimpleNamespace(cuda="12.4"),
    )
    _dist_versions(monkeypatch, {"torch": "2.4.0"})
    _fake_import(monkeypatch, torch=fake_torch)
    cap = env.probe_torch_cuda()
    assert cap.ok
    assert cap.version == "12.4"
    assert "2 CUDA device" in (cap.detail or "")


def test_torch_cuda_probe_cpu_build(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        version=types.SimpleNamespace(cuda=None),
    )
    _dist_versions(monkeypatch, {"torch": "2.4.0+cpu"})
    _fake_import(monkeypatch, torch=fake_torch)
    cap = env.probe_torch_cuda()
    assert cap.status is env.CapabilityStatus.MISSING
    assert "CPU wheel" in (cap.detail or "")


def test_torch_cuda_probe_cuda_build_without_usable_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def is_available() -> bool:
        warnings.warn("CUDA initialization: driver too old\nsecond line", stacklevel=1)
        return False

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=is_available),
        version=types.SimpleNamespace(cuda="12.4"),
    )
    _dist_versions(monkeypatch, {"torch": "2.4.0"})
    _fake_import(monkeypatch, torch=fake_torch)
    cap = env.probe_torch_cuda()
    assert cap.status is env.CapabilityStatus.ERROR
    assert cap.detail == "CUDA initialization: driver too old"


def test_torch_probes_when_torch_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    _dist_versions(monkeypatch, {})
    assert env.probe_torch().status is env.CapabilityStatus.MISSING
    assert env.probe_torch_cuda().status is env.CapabilityStatus.MISSING


def test_tritonclient_probe_states(monkeypatch: pytest.MonkeyPatch) -> None:
    _dist_versions(monkeypatch, {})
    assert env.probe_tritonclient().status is env.CapabilityStatus.MISSING

    _dist_versions(monkeypatch, {"tritonclient": "2.50.0"})
    monkeypatch.setattr("trtship.utils.env.importlib.util.find_spec", lambda name: None)
    partial = env.probe_tritonclient()
    assert partial.status is env.CapabilityStatus.ERROR
    assert "http, grpc" in (partial.detail or "")

    monkeypatch.setattr("trtship.utils.env.importlib.util.find_spec", lambda name: object())
    assert env.probe_tritonclient().ok


def test_cuda_toolkit_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    nvcc = (
        "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 12.4, V12.4.131\n"
    )
    monkeypatch.setattr(env, "run_command", lambda *a, **k: _result(nvcc))
    cap = env.probe_cuda_toolkit()
    assert cap.ok
    assert cap.version == "12.4"

    def missing(*a: object, **k: object) -> CommandResult:
        raise CommandError("executable not found: nvcc")

    monkeypatch.setattr(env, "run_command", missing)
    assert env.probe_cuda_toolkit().status is env.CapabilityStatus.MISSING


def test_tritonserver_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trtship.utils.env.shutil.which", lambda name: None)
    assert env.probe_tritonserver().status is env.CapabilityStatus.MISSING
    monkeypatch.setattr(
        "trtship.utils.env.shutil.which", lambda name: "/opt/tritonserver/bin/tritonserver"
    )
    assert env.probe_tritonserver().ok


def test_python_probe_rejects_old_interpreters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trtship.utils.env.sys.version_info", (3, 9, 0, "final", 0))
    cap = env.probe_python()
    assert cap.status is env.CapabilityStatus.ERROR
    assert "3.11" in (cap.detail or "")


# --------------------------------------------------------------------------- git


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_git_info_reports_commit_branch_and_dirty(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "a.txt").write_text("a")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    info = git_info(tmp_path)
    assert info is not None
    assert len(info.commit) == 40
    assert info.branch == "main"
    assert info.dirty is False
    (tmp_path / "a.txt").write_text("changed")
    dirty = git_info(tmp_path)
    assert dirty is not None
    assert dirty.dirty is True


def test_git_info_outside_a_repository_is_none(tmp_path: Path) -> None:
    assert git_info(tmp_path) is None
