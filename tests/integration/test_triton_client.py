"""The Triton client wrapper against a KServe v2 stub, over real HTTP and gRPC sockets.

The stub (tests/fakes/fake_triton.py) is not Triton; these tests prove that trtship drives the real
``tritonclient`` library correctly and translates its results and failures.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from tests.fakes.fake_triton import Arrays, FakeTritonServer, StubModel, StubTensor
from trtship.config import TritonConfig
from trtship.errors import EnvironmentUnavailableError, TritonError
from trtship.triton import TritonClient, TritonExecutor, wait_until_ready
from trtship.triton.client import Protocol
from trtship.utils import env

PROTOCOLS: list[Protocol] = ["http", "grpc"]


def double_and_count(inputs: Arrays) -> Arrays:
    x = inputs["x"]
    return {"y": (x * 2).astype(np.float32), "n": np.array([x.shape[0]], dtype=np.int64)}


@pytest.fixture
def stub() -> Iterator[FakeTritonServer]:
    with FakeTritonServer() as server:
        server.state.add(
            StubModel(
                name="m",
                inputs=[StubTensor("x", "FP32", [-1, 3])],
                outputs=[StubTensor("y", "FP32", [-1, 3]), StubTensor("n", "INT64", [1])],
                function=double_and_count,
            )
        )
        yield server


def client(stub: FakeTritonServer, protocol: Protocol, timeout_s: float = 5.0) -> TritonClient:
    url = stub.http_url if protocol == "http" else stub.grpc_url
    return TritonClient(protocol, url, timeout_s=timeout_s)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_health_and_readiness(stub: FakeTritonServer, protocol: Protocol) -> None:
    with client(stub, protocol) as c:
        assert c.is_live() is True
        assert c.is_ready() is True
        assert c.model_ready("m") is True
        assert c.model_ready("absent") is False
        stub.state.models["m"].ready = False
        assert c.model_ready("m") is False
        stub.state.ready = False
        assert c.is_ready() is False


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_metadata_is_the_same_over_both_protocols(
    stub: FakeTritonServer, protocol: Protocol
) -> None:
    with client(stub, protocol) as c:
        server = c.server_metadata()
        assert (server.name, server.version) == ("stub-triton", "0.0")
        assert "binary_tensor_data" in server.extensions
        model = c.model_metadata("m")
    assert model.name == "m"
    assert model.platform == "tensorrt_plan"
    assert model.versions == ["1"]
    assert [(t.name, t.datatype, t.shape) for t in model.inputs] == [("x", "FP32", [-1, 3])]
    assert [(t.name, t.datatype, t.shape) for t in model.outputs] == [
        ("y", "FP32", [-1, 3]),
        ("n", "INT64", [1]),
    ]


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_inference_returns_every_output_as_numpy(
    stub: FakeTritonServer, protocol: Protocol
) -> None:
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    with client(stub, protocol) as c:
        result = c.infer("m", {"x": x})
    np.testing.assert_array_equal(result["y"], x * 2)
    assert result["y"].dtype == np.float32
    assert result["n"].tolist() == [2]
    assert result["n"].dtype == np.int64
    assert ("http" if protocol == "http" else "grpc", "m") in stub.state.infer_calls


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_inference_can_request_a_subset_of_outputs(
    stub: FakeTritonServer, protocol: Protocol
) -> None:
    with client(stub, protocol) as c:
        result = c.infer("m", {"x": np.ones((1, 3), dtype=np.float32)}, outputs=["n"])
    assert list(result) == ["n"]


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_non_contiguous_and_float16_inputs_round_trip(
    stub: FakeTritonServer, protocol: Protocol
) -> None:
    stub.state.add(
        StubModel(
            name="half",
            inputs=[StubTensor("x", "FP16", [-1, 3])],
            outputs=[StubTensor("y", "FP16", [-1, 3])],
            function=lambda i: {"y": i["x"] + np.float16(1)},
        )
    )
    x = np.arange(12, dtype=np.float16).reshape(3, 4)[:, :3]  # a non-contiguous view
    assert not x.flags["C_CONTIGUOUS"]
    with client(stub, protocol) as c:
        result = c.infer("half", {"x": x})
    np.testing.assert_array_equal(result["y"], x + np.float16(1))
    assert result["y"].dtype == np.float16


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_unknown_models_and_failed_inferences_are_triton_errors(
    stub: FakeTritonServer, protocol: Protocol
) -> None:
    def boom(_: Arrays) -> Arrays:
        raise RuntimeError("bad shape")

    stub.state.add(
        StubModel(
            "bad", [StubTensor("x", "FP32", [-1, 3])], [StubTensor("y", "FP32", [-1, 3])], boom
        )
    )
    with client(stub, protocol) as c:
        with pytest.raises(TritonError, match="unknown model"):
            c.model_metadata("absent")
        with pytest.raises(TritonError, match="bad shape") as excinfo:
            c.infer("bad", {"x": np.ones((1, 3), dtype=np.float32)})
    assert excinfo.value.details["protocol"] == protocol


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_an_unreachable_server_is_a_triton_error_with_a_hint(protocol: Protocol) -> None:
    with (
        TritonClient(protocol, "127.0.0.1:1", timeout_s=2.0) as c,
        pytest.raises(TritonError, match="liveness check failed") as excinfo,
    ):
        c.is_live()
    assert "trtship status" in (excinfo.value.hint or "")


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_the_executor_adapts_a_served_model_to_validation(
    stub: FakeTritonServer, protocol: Protocol
) -> None:
    executor = TritonExecutor(client(stub, protocol), "m")
    result = executor.run({"x": np.ones((4, 3), dtype=np.float32)})
    executor.close()
    assert result["y"].shape == (4, 3)


def test_from_settings_uses_the_matching_port(stub: FakeTritonServer) -> None:
    settings = TritonConfig(http_port=stub.http_port, grpc_port=stub.grpc_port)
    for protocol in PROTOCOLS:
        with TritonClient.from_settings(settings, protocol) as c:
            assert c.is_live()
            assert c.url.endswith(str(stub.http_port if protocol == "http" else stub.grpc_port))


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_wait_until_ready_waits_for_the_model(stub: FakeTritonServer, protocol: Protocol) -> None:
    stub.state.models["m"].ready = False
    sleeps: list[float] = []
    ticks = iter(range(100))

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 2:
            stub.state.models["m"].ready = True

    with client(stub, protocol) as c:
        wait_until_ready(c, "m", 30.0, sleep=sleep, clock=lambda: float(next(ticks)))
    assert len(sleeps) == 2


def test_wait_until_ready_times_out_naming_the_model(stub: FakeTritonServer) -> None:
    stub.state.models["m"].ready = False
    ticks = iter(range(0, 100, 10))
    with client(stub, "http") as c, pytest.raises(TritonError, match="'m' was not ready"):
        wait_until_ready(c, "m", 5.0, sleep=lambda _: None, clock=lambda: float(next(ticks)))


def test_wait_until_ready_tolerates_a_server_that_is_not_up_yet() -> None:
    ticks = iter(range(0, 100, 10))
    with (
        TritonClient("http", "127.0.0.1:1", timeout_s=1.0) as c,
        pytest.raises(TritonError, match="not ready"),
    ):
        wait_until_ready(c, "m", 5.0, sleep=lambda _: None, clock=lambda: float(next(ticks)))


def test_a_missing_tritonclient_is_an_environment_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str, purpose: str) -> Any:
        raise EnvironmentUnavailableError(f"{purpose} requires {name}, which is missing")

    monkeypatch.setattr(env, "require", missing)
    with pytest.raises(EnvironmentUnavailableError):
        TritonClient("http", "127.0.0.1:8000")
