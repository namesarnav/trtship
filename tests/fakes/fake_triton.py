"""A stand-in for a Triton server that speaks the KServe v2 protocol over HTTP and gRPC.

It exists so trtship's *client* code and the real ``tritonclient`` library are exercised over real
sockets without a GPU. It is not Triton: it serves plain Python functions, so it proves nothing
about TensorRT models or Triton's scheduler.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from concurrent import futures
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import grpc
import numpy as np
import numpy.typing as npt
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.utils import np_to_triton_dtype, triton_to_np_dtype

Arrays = dict[str, npt.NDArray[Any]]


@dataclass
class StubTensor:
    name: str
    datatype: str
    shape: list[int]


@dataclass
class StubModel:
    name: str
    inputs: list[StubTensor]
    outputs: list[StubTensor]
    function: Callable[[Arrays], Arrays]
    ready: bool = True
    platform: str = "tensorrt_plan"


@dataclass
class StubState:
    models: dict[str, StubModel] = field(default_factory=dict)
    live: bool = True
    ready: bool = True
    infer_calls: list[tuple[str, str]] = field(default_factory=list)  # (protocol, model)

    def add(self, model: StubModel) -> None:
        self.models[model.name] = model


def _model_metadata(model: StubModel) -> dict[str, Any]:
    return {
        "name": model.name,
        "versions": ["1"],
        "platform": model.platform,
        "inputs": [
            {"name": t.name, "datatype": t.datatype, "shape": t.shape} for t in model.inputs
        ],
        "outputs": [
            {"name": t.name, "datatype": t.datatype, "shape": t.shape} for t in model.outputs
        ],
    }


# ---------------------------------------------------------------- HTTP


def _http_handler(state: StubState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:
            pass

        def _send(
            self, status: int, body: bytes = b"", headers: Mapping[str, str] | None = None
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            self._send(status, json.dumps(payload).encode(), {"Content-Type": "application/json"})

        def do_GET(self) -> None:
            path = self.path
            if path == "/v2/health/live":
                self._send(200 if state.live else 400)
            elif path == "/v2/health/ready":
                self._send(200 if state.ready else 400)
            elif path == "/v2":
                self._json(
                    200,
                    {"name": "stub-triton", "version": "0.0", "extensions": ["binary_tensor_data"]},
                )
            elif path.startswith("/v2/models/"):
                parts = path.split("/")
                model = state.models.get(parts[3])
                if model is None:
                    self._json(
                        400, {"error": f"Request for unknown model: '{parts[3]}' is not found"}
                    )
                elif len(parts) > 4 and parts[4] == "ready":
                    self._send(200 if model.ready else 400)
                else:
                    self._json(200, _model_metadata(model))
            else:
                self._send(404)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            if self.path == "/v2/repository/index":
                self._json(
                    200,
                    [
                        {
                            "name": m.name,
                            "version": "1",
                            "state": "READY" if m.ready else "UNAVAILABLE",
                        }
                        for m in state.models.values()
                    ],
                )
                return
            parts = self.path.split("/")
            if len(parts) == 5 and parts[4] == "infer":
                self._infer(parts[3], body)
            else:
                self._send(404)

        def _infer(self, name: str, body: bytes) -> None:
            model = state.models.get(name)
            if model is None:
                self._json(400, {"error": f"Request for unknown model: '{name}' is not found"})
                return
            header_length = int(self.headers.get("Inference-Header-Content-Length", len(body)))
            header = json.loads(body[:header_length])
            binary = body[header_length:]
            inputs: Arrays = {}
            offset = 0
            for tensor in header["inputs"]:
                dtype = triton_to_np_dtype(tensor["datatype"])
                size = tensor.get("parameters", {}).get("binary_data_size")
                if size is not None:
                    raw = binary[offset : offset + size]
                    offset += size
                    array = np.frombuffer(raw, dtype=dtype).reshape(tensor["shape"])
                else:
                    array = np.array(tensor["data"], dtype=dtype).reshape(tensor["shape"])
                inputs[tensor["name"]] = array
            state.infer_calls.append(("http", name))
            try:
                outputs = model.function(inputs)
            except Exception as exc:
                self._json(400, {"error": f"inference failed: {exc}"})
                return
            wanted = {o["name"]: o for o in header.get("outputs", [])}
            descriptors: list[dict[str, Any]] = []
            chunks: list[bytes] = []
            for out_name, array in outputs.items():
                if wanted and out_name not in wanted:
                    continue
                descriptor: dict[str, Any] = {
                    "name": out_name,
                    "datatype": np_to_triton_dtype(array.dtype),
                    "shape": list(array.shape),
                }
                if wanted.get(out_name, {}).get("parameters", {}).get("binary_data"):
                    raw = np.ascontiguousarray(array).tobytes()
                    descriptor["parameters"] = {"binary_data_size": len(raw)}
                    chunks.append(raw)
                else:
                    descriptor["data"] = array.flatten().tolist()
                descriptors.append(descriptor)
            head = json.dumps(
                {"model_name": name, "model_version": "1", "outputs": descriptors}
            ).encode()
            self._send(
                200,
                head + b"".join(chunks),
                {
                    "Content-Type": "application/octet-stream",
                    "Inference-Header-Content-Length": str(len(head)),
                },
            )

    return Handler


# ---------------------------------------------------------------- gRPC


class _Servicer(service_pb2_grpc.GRPCInferenceServiceServicer):  # type: ignore[misc]
    def __init__(self, state: StubState) -> None:
        self.state = state

    def _model(self, name: str, context: grpc.ServicerContext) -> StubModel:
        model = self.state.models.get(name)
        if model is None:
            context.abort(
                grpc.StatusCode.NOT_FOUND, f"Request for unknown model: '{name}' is not found"
            )
        assert model is not None
        return model

    def ServerLive(self, request: Any, context: Any) -> Any:
        return service_pb2.ServerLiveResponse(live=self.state.live)

    def ServerReady(self, request: Any, context: Any) -> Any:
        return service_pb2.ServerReadyResponse(ready=self.state.ready)

    def ModelReady(self, request: Any, context: Any) -> Any:
        model = self.state.models.get(request.name)
        return service_pb2.ModelReadyResponse(ready=bool(model and model.ready))

    def ServerMetadata(self, request: Any, context: Any) -> Any:
        return service_pb2.ServerMetadataResponse(
            name="stub-triton", version="0.0", extensions=["binary_tensor_data"]
        )

    def ModelMetadata(self, request: Any, context: Any) -> Any:
        model = self._model(request.name, context)
        response = service_pb2.ModelMetadataResponse(
            name=model.name, versions=["1"], platform=model.platform
        )
        for tensor in model.inputs:
            response.inputs.add(name=tensor.name, datatype=tensor.datatype, shape=tensor.shape)
        for tensor in model.outputs:
            response.outputs.add(name=tensor.name, datatype=tensor.datatype, shape=tensor.shape)
        return response

    def ModelInfer(self, request: Any, context: Any) -> Any:
        model = self._model(request.model_name, context)
        inputs: Arrays = {}
        for tensor, raw in zip(request.inputs, request.raw_input_contents, strict=True):
            inputs[tensor.name] = np.frombuffer(
                raw, dtype=triton_to_np_dtype(tensor.datatype)
            ).reshape(list(tensor.shape))
        self.state.infer_calls.append(("grpc", model.name))
        try:
            outputs = model.function(inputs)
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, f"inference failed: {exc}")
        wanted = {o.name for o in request.outputs}
        response = service_pb2.ModelInferResponse(model_name=model.name, model_version="1")
        for name, array in outputs.items():
            if wanted and name not in wanted:
                continue
            response.outputs.add(
                name=name, datatype=np_to_triton_dtype(array.dtype), shape=list(array.shape)
            )
            response.raw_output_contents.append(np.ascontiguousarray(array).tobytes())
        return response


class FakeTritonServer:
    """Starts the HTTP and gRPC stubs on ephemeral local ports."""

    def __init__(self) -> None:
        self.state = StubState()
        self._http = ThreadingHTTPServer(("127.0.0.1", 0), _http_handler(self.state))
        self._grpc = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(
            _Servicer(self.state), self._grpc
        )
        self.grpc_port = self._grpc.add_insecure_port("127.0.0.1:0")
        self.http_port = self._http.server_address[1]
        self._thread = threading.Thread(
            target=lambda: self._http.serve_forever(poll_interval=0.02), daemon=True
        )

    def __enter__(self) -> FakeTritonServer:
        self._thread.start()
        self._grpc.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._http.shutdown()
        self._http.server_close()
        self._grpc.stop(grace=None).wait()

    @property
    def http_url(self) -> str:
        return f"127.0.0.1:{self.http_port}"

    @property
    def grpc_url(self) -> str:
        return f"127.0.0.1:{self.grpc_port}"
