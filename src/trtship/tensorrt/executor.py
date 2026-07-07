"""Running a serialized TensorRT engine.

Device memory goes through :class:`DeviceMemory`: torch CUDA tensors in production, a host stand-in
in tests, so the binding logic (input shapes, tensor addresses, output allocation, dtype handling)
is unit-tested without a GPU. Only optimization profile 0 is used.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from trtship.errors import EngineRuntimeError, EnvironmentUnavailableError
from trtship.logging import get_logger
from trtship.models.dtypes import to_numpy
from trtship.tensorrt.engine_info import EngineInfo, TensorBinding, describe_engine
from trtship.tensorrt.loader import load_tensorrt

log = get_logger(__name__)


class DeviceTimer(Protocol):
    """Times a stretch of device work. ``stop_ms`` waits for the work to finish."""

    def start(self) -> None: ...

    def stop_ms(self) -> float: ...


class DeviceMemory(Protocol):
    @property
    def stream(self) -> int:
        """The CUDA stream handle passed to ``execute_async_v3``."""

    def allocate(self, shape: tuple[int, ...], dtype: np.dtype[Any]) -> tuple[Any, int]:
        """A buffer of ``shape``/``dtype`` and its device pointer."""

    def upload(self, handle: Any, array: npt.NDArray[Any]) -> None: ...

    def download(self, handle: Any) -> npt.NDArray[Any]: ...

    def synchronize(self) -> None: ...

    def timer(self) -> DeviceTimer: ...


class TorchDeviceMemory:
    """Device buffers backed by torch CUDA tensors."""

    def __init__(self, device_index: int = 0) -> None:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            raise EnvironmentUnavailableError(
                "running a TensorRT engine needs a CUDA build of PyTorch to manage GPU buffers",
                hint="Install a CUDA build of PyTorch (a plain `uv sync` on a GPU machine).",
            )
        self._torch = torch
        self._device = torch.device("cuda", device_index)
        torch.cuda.set_device(self._device)

    @property
    def stream(self) -> int:
        return int(self._torch.cuda.current_stream().cuda_stream)

    def allocate(self, shape: tuple[int, ...], dtype: np.dtype[Any]) -> tuple[Any, int]:
        torch_dtype = self._torch.from_numpy(np.empty(0, dtype=dtype)).dtype
        tensor = self._torch.empty(shape, dtype=torch_dtype, device=self._device)
        return tensor, int(tensor.data_ptr())

    def upload(self, handle: Any, array: npt.NDArray[Any]) -> None:
        handle.copy_(self._torch.from_numpy(np.ascontiguousarray(array)))

    def download(self, handle: Any) -> npt.NDArray[Any]:
        return np.asarray(handle.cpu().numpy())

    def synchronize(self) -> None:
        self._torch.cuda.synchronize(self._device)

    def timer(self) -> DeviceTimer:
        return _CudaEventTimer(self._torch)


class _CudaEventTimer:
    """GPU-side timing with CUDA events, which excludes host launch overhead and queueing."""

    def __init__(self, torch: Any) -> None:
        self._start = torch.cuda.Event(enable_timing=True)
        self._stop = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self._start.record()

    def stop_ms(self) -> float:
        self._stop.record()
        self._stop.synchronize()
        return float(self._start.elapsed_time(self._stop))


class BoundExecution:
    """Inputs uploaded and outputs allocated for one shape; execute repeatedly without copies."""

    def __init__(
        self,
        context: Any,
        memory: DeviceMemory,
        outputs: dict[str, tuple[Any, TensorBinding]],
        name: str,
        inputs: list[tuple[str, Any, np.dtype[Any]]],
    ) -> None:
        self._context = context
        self._memory = memory
        self._outputs = outputs
        self._name = name
        self._inputs = inputs  # (name, buffer, dtype): keeps the buffers alive; used by refresh()

    def refresh(self, inputs: Mapping[str, npt.NDArray[Any]]) -> None:
        """Copy new input data (same shapes) into the bound buffers."""
        for name, handle, dtype in self._inputs:
            self._memory.upload(handle, np.asarray(inputs[name]).astype(dtype, copy=False))

    @property
    def memory(self) -> DeviceMemory:
        return self._memory

    def execute(self) -> None:
        """Enqueue one inference on the stream (does not wait)."""
        if not self._context.execute_async_v3(self._memory.stream):
            raise EngineRuntimeError(f"TensorRT failed to execute engine {self._name}")

    def synchronize(self) -> None:
        self._memory.synchronize()

    def outputs(self) -> dict[str, npt.NDArray[Any]]:
        """Copy the current outputs to the host (synchronizes first)."""
        self._memory.synchronize()
        return {name: self._memory.download(handle) for name, (handle, _) in self._outputs.items()}


class TensorRTExecutor:
    def __init__(
        self,
        plan: bytes | Path,
        *,
        trt: Any | None = None,
        memory: DeviceMemory | None = None,
        name: str | None = None,
    ) -> None:
        self._trt = trt if trt is not None else load_tensorrt(purpose="running a TensorRT engine")
        self._memory = memory if memory is not None else TorchDeviceMemory()
        self.name = name or (plan.name if isinstance(plan, Path) else "engine")
        blob = plan.read_bytes() if isinstance(plan, Path) else plan
        runtime = self._trt.Runtime(_QuietLogger(self._trt))
        self._engine = runtime.deserialize_cuda_engine(blob)
        if self._engine is None:
            raise EngineRuntimeError(
                f"cannot load TensorRT engine {self.name}",
                hint="A plan loads only with the TensorRT version and GPU architecture that built "
                "it; rebuild it here.",
            )
        self.info: EngineInfo = describe_engine(self._trt, self._engine)
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise EngineRuntimeError(f"cannot create an execution context for {self.name}")

    def bind(self, inputs: Mapping[str, npt.NDArray[Any]]) -> BoundExecution:
        """Set shapes and addresses for ``inputs`` and upload them; the result can be executed
        repeatedly without further copies and owns its buffers."""
        for binding in self.info.inputs:
            if binding.name not in inputs:
                raise EngineRuntimeError(
                    f"missing input {binding.name!r} for engine {self.name}",
                    details={"expected": [b.name for b in self.info.inputs]},
                )
        input_handles: list[tuple[str, Any, np.dtype[Any]]] = []
        for binding in self.info.inputs:
            array = np.asarray(inputs[binding.name]).astype(to_numpy(binding.dtype), copy=False)
            self._set_input(binding, array)
            handle, pointer = self._memory.allocate(tuple(array.shape), array.dtype)
            self._memory.upload(handle, array)
            self._context.set_tensor_address(binding.name, pointer)
            input_handles.append((binding.name, handle, array.dtype))

        outputs: dict[str, tuple[Any, TensorBinding]] = {}
        for binding in self.info.outputs:
            shape = tuple(int(d) for d in self._context.get_tensor_shape(binding.name))
            if any(d < 0 for d in shape):
                raise EngineRuntimeError(
                    f"output {binding.name!r} of {self.name} has a data-dependent shape "
                    f"{list(shape)}",
                    hint="Outputs whose size is only known after execution are not supported.",
                )
            handle, pointer = self._memory.allocate(shape, to_numpy(binding.dtype))
            self._context.set_tensor_address(binding.name, pointer)
            outputs[binding.name] = (handle, binding)
        return BoundExecution(self._context, self._memory, outputs, self.name, input_handles)

    def run(self, inputs: Mapping[str, npt.NDArray[Any]]) -> dict[str, npt.NDArray[Any]]:
        """One inference: bind, execute, and return the outputs as numpy arrays."""
        bound = self.bind(inputs)
        bound.execute()
        return bound.outputs()

    def _set_input(self, binding: TensorBinding, array: npt.NDArray[Any]) -> None:
        if binding.is_dynamic:
            if not self._context.set_input_shape(binding.name, tuple(array.shape)):
                ranges = [r.model_dump() for r in binding.profiles]
                raise EngineRuntimeError(
                    f"input {binding.name!r} of shape {list(array.shape)} is outside the engine's "
                    "optimization profile",
                    hint="Rebuild with a profile that covers this shape (tensorrt.profiles).",
                    details={"profile_ranges": ranges},
                )
        elif list(array.shape) != binding.shape:
            raise EngineRuntimeError(
                f"input {binding.name!r} has shape {list(array.shape)} but the engine is fixed to "
                f"{binding.shape}"
            )

    def close(self) -> None:
        self._context = None
        self._engine = None

    def __enter__(self) -> TensorRTExecutor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _QuietLogger:
    """Placeholder replaced at runtime by a real ``ILogger`` subclass (see ``__new__``)."""

    def __new__(cls, trt: Any) -> Any:
        class Logger(trt.ILogger):  # type: ignore[misc]
            def __init__(self) -> None:
                trt.ILogger.__init__(self)

            def log(self, severity: Any, msg: str) -> None:
                if int(severity) <= int(trt.ILogger.Severity.WARNING):
                    log.warning("tensorrt: %s", msg)

        return Logger()
