"""HTTP and gRPC clients for a running Triton server, over the official ``tritonclient``.

``tritonclient`` is optional (``pip install 'trtship[triton]'``) and imported only when a client is
created. Both protocols expose the same operations and return plain Python/numpy values, and every
failure surfaces as :class:`~trtship.errors.TritonError` with the server's message.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from trtship.config import TritonConfig
from trtship.errors import TritonError
from trtship.triton.server import probe_host
from trtship.utils import env

Protocol = Literal["http", "grpc"]


class TensorMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    datatype: str
    shape: list[int]  # -1 marks a dynamic dimension; with batching the first axis is the batch


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    versions: list[str] = Field(default_factory=list)
    platform: str = ""
    inputs: list[TensorMetadata]
    outputs: list[TensorMetadata]


class ServerMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str
    extensions: list[str] = Field(default_factory=list)


def _load_module(protocol: Protocol) -> ModuleType:
    env.require(env.TRITON_CLIENT, purpose="talking to a Triton server")
    if protocol == "http":
        import tritonclient.http as module  # noqa: PLC0415
    else:
        import tritonclient.grpc as module  # noqa: PLC0415
    return module  # type: ignore[no-any-return]


def _tensors(raw: list[dict[str, Any]] | None) -> list[TensorMetadata]:
    # gRPC's JSON form encodes 64-bit integers (shapes) as strings.
    return [
        TensorMetadata(
            name=t["name"], datatype=t["datatype"], shape=[int(d) for d in t.get("shape", [])]
        )
        for t in raw or []
    ]


class TritonClient:
    """One connection to a Triton server over ``protocol``."""

    def __init__(
        self,
        protocol: Protocol,
        url: str,
        *,
        timeout_s: float = 30.0,
        module: ModuleType | None = None,
    ) -> None:
        self.protocol: Protocol = protocol
        self.url = url
        self.timeout_s = timeout_s
        self._module = module or _load_module(protocol)
        self._outputs: dict[str, list[str]] = {}
        if protocol == "http":
            self._client = self._module.InferenceServerClient(
                url, connection_timeout=timeout_s, network_timeout=timeout_s
            )
        else:
            self._client = self._module.InferenceServerClient(url)

    @classmethod
    def from_settings(
        cls, settings: TritonConfig, protocol: Protocol = "http", *, timeout_s: float = 30.0
    ) -> TritonClient:
        port = settings.http_port if protocol == "http" else settings.grpc_port
        return cls(protocol, f"{probe_host(settings)}:{port}", timeout_s=timeout_s)

    def __enter__(self) -> TritonClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------ plumbing

    def _call(self, what: str, call: Callable[[], Any]) -> Any:
        from tritonclient.utils import InferenceServerException  # noqa: PLC0415

        try:
            return call()
        except (InferenceServerException, OSError) as exc:
            raise TritonError(
                f"{what} failed over {self.protocol} at {self.url}: {exc}",
                hint="Check that the server is running (`trtship status`) and the port matches.",
                details={"protocol": self.protocol, "url": self.url},
            ) from exc

    def _timeout(self) -> dict[str, float]:
        return {"client_timeout": self.timeout_s} if self.protocol == "grpc" else {}

    # ------------------------------------------------------------------ health and metadata

    def is_live(self) -> bool:
        return bool(
            self._call("liveness check", lambda: self._client.is_server_live(**self._timeout()))
        )

    def is_ready(self) -> bool:
        return bool(
            self._call("readiness check", lambda: self._client.is_server_ready(**self._timeout()))
        )

    def model_ready(self, model: str) -> bool:
        return bool(
            self._call(
                f"readiness check of model {model!r}",
                lambda: self._client.is_model_ready(model, **self._timeout()),
            )
        )

    def server_metadata(self) -> ServerMetadata:
        raw = self._call("server metadata request", lambda: self._metadata(None))
        return ServerMetadata(
            name=raw["name"], version=raw["version"], extensions=list(raw.get("extensions", []))
        )

    def model_metadata(self, model: str) -> ModelMetadata:
        raw = self._call(f"metadata request for model {model!r}", lambda: self._metadata(model))
        return ModelMetadata(
            name=raw["name"],
            versions=[str(v) for v in raw.get("versions", [])],
            platform=raw.get("platform", ""),
            inputs=_tensors(raw.get("inputs")),
            outputs=_tensors(raw.get("outputs")),
        )

    def _metadata(self, model: str | None) -> dict[str, Any]:
        client = self._client
        if self.protocol == "grpc":
            kwargs: dict[str, Any] = {"as_json": True, **self._timeout()}
            raw = (
                client.get_server_metadata(**kwargs)
                if model is None
                else client.get_model_metadata(model, **kwargs)
            )
        else:
            raw = (
                client.get_server_metadata() if model is None else client.get_model_metadata(model)
            )
        return dict(raw)

    # ------------------------------------------------------------------ inference

    def _output_names(self, model: str) -> list[str]:
        if model not in self._outputs:
            self._outputs[model] = [t.name for t in self.model_metadata(model).outputs]
        return self._outputs[model]

    def _requested_output(self, name: str) -> Any:
        # Binary tensor data avoids JSON round trips of large float arrays over HTTP.
        if self.protocol == "http":
            return self._module.InferRequestedOutput(name, binary_data=True)
        return self._module.InferRequestedOutput(name)

    def infer(
        self,
        model: str,
        inputs: Mapping[str, npt.NDArray[Any]],
        outputs: list[str] | None = None,
    ) -> dict[str, npt.NDArray[Any]]:
        """Run one request and return the requested outputs (default: every output)."""
        from tritonclient.utils import np_to_triton_dtype  # noqa: PLC0415

        module = self._module
        request_inputs = []
        for name, array in inputs.items():
            contiguous = np.ascontiguousarray(array)
            tensor = module.InferInput(
                name, list(contiguous.shape), np_to_triton_dtype(contiguous.dtype)
            )
            tensor.set_data_from_numpy(contiguous)
            request_inputs.append(tensor)
        names = outputs or self._output_names(model)
        requested = [self._requested_output(n) for n in names]
        result = self._call(
            f"inference on model {model!r}",
            lambda: self._client.infer(model, request_inputs, outputs=requested, **self._timeout()),
        )
        arrays = {name: result.as_numpy(name) for name in names}
        missing = [n for n, a in arrays.items() if a is None]
        if missing:
            raise TritonError(f"the response for model {model!r} has no output(s): {missing}")
        return arrays


class TritonExecutor:
    """Adapts a served model to the executor protocol of numerical validation."""

    def __init__(self, client: TritonClient, model: str) -> None:
        self.client = client
        self.model = model

    def run(self, inputs: Mapping[str, npt.NDArray[Any]]) -> dict[str, npt.NDArray[Any]]:
        return self.client.infer(self.model, inputs)

    def close(self) -> None:
        self.client.close()


def wait_until_ready(
    client: TritonClient,
    model: str,
    timeout_s: float,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Block until the server and ``model`` report ready, or raise after ``timeout_s``."""
    deadline = clock() + timeout_s
    while True:
        try:
            if client.is_ready() and client.model_ready(model):
                return
        except TritonError:
            pass  # not accepting connections yet
        if clock() >= deadline:
            raise TritonError(
                f"model {model!r} was not ready over {client.protocol} within {timeout_s:g}s",
                hint="Check `trtship status` and the container logs.",
            )
        sleep(1.0)
