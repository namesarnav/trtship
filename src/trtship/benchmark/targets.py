"""Benchmark backends: ONNX Runtime (CPU) and TensorRT (direct)."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from trtship.benchmark.runner import PhaseTimes
from trtship.benchmark.schema import MemoryUsage
from trtship.config import TrtshipConfig
from trtship.errors import BenchmarkError, ConfigError
from trtship.models import make_inputs, resolve_symbol_sizes
from trtship.onnx.runtime import OrtSession
from trtship.tensorrt.executor import BoundExecution, TensorRTExecutor


def cpu_rss_mb() -> float | None:
    """Peak resident set size of this process in MiB; ``None`` where the OS cannot report it."""
    try:
        import resource  # noqa: PLC0415 - not available on Windows
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


def torch_gpu_used_mb(index: int = 0) -> float | None:
    """Device memory in use (all processes) on GPU ``index`` in MiB; ``None`` without CUDA."""
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info(index)
    return float(total - free) / (1024 * 1024)


def batch_inputs(config: TrtshipConfig, batch_size: int, seed: int) -> dict[str, npt.NDArray[Any]]:
    """Deterministic inputs at ``batch_size``: profile ``opt`` shapes with the batch axis replaced.

    The batch size must lie in the first profile's range for the leading axis, so an engine is only
    ever benchmarked at shapes it was built for.
    """
    specs = config.model.inputs
    lead = specs[0].shape[0]
    sizes = resolve_symbol_sizes(config.model, config.tensorrt.profiles, "opt")
    if isinstance(lead, str):
        if config.tensorrt.profiles:
            profile = config.tensorrt.profiles[0]
            given = profile.inputs.get(specs[0].name)
            if given is not None and not given.min[0] <= batch_size <= given.max[0]:
                raise ConfigError(
                    f"benchmark batch size {batch_size} is outside the profile range "
                    f"[{given.min[0]}, {given.max[0]}] for input {specs[0].name!r}",
                    hint="Choose benchmark.batch_sizes within tensorrt.profiles.",
                )
        sizes[lead] = batch_size
    elif lead != batch_size:
        raise ConfigError(
            f"input {specs[0].name!r} has a fixed batch axis of {lead}; cannot benchmark batch "
            f"size {batch_size}"
        )
    tensors = make_inputs(specs, sizes, seed=seed, device="cpu")
    return {name: tensor.numpy() for name, tensor in tensors.items()}


class OrtTarget:
    """ONNX Runtime on the CPU with its default graph optimizations."""

    backend = "onnxruntime-cpu"
    device = "cpu"
    precision: str | None = None
    supports_concurrency = True

    def __init__(self, onnx_path: Path, config: TrtshipConfig, seed: int) -> None:
        self._session = OrtSession(onnx_path, optimize=True)
        self._config = config
        self._seed = seed
        self._inputs: dict[str, npt.NDArray[Any]] = {}
        self.providers = self._session.providers

    def prepare(self, batch_size: int) -> None:
        self._inputs = batch_inputs(self._config, batch_size, self._seed)

    def make_worker(self) -> Callable[[], PhaseTimes]:
        session, source = self._session, self._inputs

        def once() -> PhaseTimes:
            t0 = time.perf_counter()
            feed = {name: np.ascontiguousarray(array) for name, array in source.items()}
            t1 = time.perf_counter()
            outputs = session.run(feed)
            t2 = time.perf_counter()
            _ = [np.asarray(value) for value in outputs.values()]
            t3 = time.perf_counter()
            return PhaseTimes((t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000)

        return once

    def memory(self) -> MemoryUsage:
        return MemoryUsage(gpu_mb=None, cpu_rss_mb=cpu_rss_mb())

    def notes(self) -> list[str]:
        return [f"providers: {', '.join(self.providers)}; graph optimizations enabled"]

    def close(self) -> None:
        self._inputs = {}


class TensorRTTarget:
    """A TensorRT engine executed directly: no serving layer, one request at a time.

    preprocess = host->device copy of the inputs, execute = GPU time measured with CUDA events,
    postprocess = device->host copy of the outputs.
    """

    backend = "tensorrt"
    supports_concurrency = False

    def __init__(
        self,
        executor: TensorRTExecutor,
        config: TrtshipConfig,
        seed: int,
        *,
        precision: str,
        device: str,
        gpu_used_mb: Callable[[], float | None] | None = None,
        gpu_baseline_mb: float | None = None,
    ) -> None:
        self._executor = executor
        self._config = config
        self._seed = seed
        self.precision = precision
        self.device = device
        self._probe = gpu_used_mb
        self._baseline = gpu_baseline_mb
        self._inputs: dict[str, npt.NDArray[Any]] = {}
        self._bound: BoundExecution | None = None

    def prepare(self, batch_size: int) -> None:
        self._inputs = batch_inputs(self._config, batch_size, self._seed)
        self._bound = self._executor.bind(self._inputs)

    def make_worker(self) -> Callable[[], PhaseTimes]:
        bound = self._bound
        if bound is None:
            raise BenchmarkError("the target was not prepared")
        inputs = self._inputs

        def once() -> PhaseTimes:
            memory = bound.memory
            t0 = time.perf_counter()
            bound.refresh(inputs)
            memory.synchronize()
            t1 = time.perf_counter()
            timer = memory.timer()
            timer.start()
            bound.execute()
            execute_ms = timer.stop_ms()
            t2 = time.perf_counter()
            bound.outputs()
            t3 = time.perf_counter()
            return PhaseTimes((t1 - t0) * 1000, execute_ms, (t3 - t2) * 1000)

        return once

    def memory(self) -> MemoryUsage:
        used = self._probe() if self._probe else None
        delta = used - self._baseline if used is not None and self._baseline is not None else None
        return MemoryUsage(gpu_mb=delta, cpu_rss_mb=cpu_rss_mb())

    def notes(self) -> list[str]:
        return [
            "GPU memory is the growth in device memory used since before the engine was loaded"
            if self._baseline is not None
            else "GPU memory was not measured",
        ]

    def close(self) -> None:
        self._bound = None
        self._executor.close()


def describe_environment(versions: Mapping[str, str | None], gpus: list[str]) -> dict[str, Any]:
    return {"versions": dict(versions), "gpus": gpus}
