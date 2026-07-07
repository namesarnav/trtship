"""Engine executors for tests. None of these is TensorRT: they stand in for an engine so the
validation logic (tolerances, shape points, reporting) can be tested on any machine."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch

from tests.fakes.fake_tensorrt import (
    FakeBuilderConfig,
    FakeOptions,
    FakeProfile,
    HostMemory,
    make_fake_trt,
)
from trtship.models import LoadedModel
from trtship.tensorrt import TensorRTExecutor

Outputs = dict[str, np.ndarray]
Transform = Callable[[Outputs, Mapping[str, np.ndarray]], Outputs]


class TorchBackedExecutor:
    """Runs the PyTorch model, optionally corrupting outputs to simulate an inaccurate engine."""

    def __init__(self, model: LoadedModel, transform: Transform | None = None) -> None:
        self.model = model
        self.transform = transform
        self.closed = False
        self.calls = 0

    def run(self, inputs: Mapping[str, Any]) -> Outputs:
        self.calls += 1
        tensors = {name: torch.from_numpy(np.asarray(value)) for name, value in inputs.items()}
        outputs = {k: v.detach().numpy() for k, v in self.model.run(tensors).items()}
        return self.transform(outputs, inputs) if self.transform else outputs

    def close(self) -> None:
        self.closed = True


def noisy(scale: float, seed: int = 0) -> Transform:
    def transform(outputs: Outputs, inputs: Mapping[str, np.ndarray]) -> Outputs:
        rng = np.random.default_rng(seed)
        return {k: v + rng.normal(0, scale, v.shape).astype(v.dtype) for k, v in outputs.items()}

    return transform


def reversed_logits(outputs: Outputs, inputs: Mapping[str, np.ndarray]) -> Outputs:
    return {k: v[..., ::-1].copy() for k, v in outputs.items()}


def wrong_batch(outputs: Outputs, inputs: Mapping[str, np.ndarray]) -> Outputs:
    """Returns four rows whatever the batch: like a graph with the batch baked in."""
    return {k: np.resize(v, (4, *v.shape[1:])) for k, v in outputs.items()}


def double_first_four(inputs: dict[str, Any]) -> dict[str, Any]:
    return {"output": inputs["x"][:, :4] * 2}


def make_engine_executor(
    options: Any = None,
    profile: tuple[tuple[int, ...], ...] = ((1, 16), (4, 16), (8, 16)),
) -> tuple[Any, Any, Any]:
    """A real TensorRTExecutor over the fake TensorRT and host memory: (executor, memory, calls)."""
    memory = HostMemory()
    opts = options or FakeOptions()
    opts.memory = memory
    opts.compute = opts.compute or double_first_four
    trt, calls = make_fake_trt(opts)
    cfg = FakeBuilderConfig(calls, optimization_level=True)
    fake_profile = FakeProfile(valid=True)
    fake_profile.set_shape("x", *profile)
    cfg.add_optimization_profile(fake_profile)
    calls.configs.append(cfg)
    return TensorRTExecutor(b"plan", trt=trt, memory=memory, name="test.plan"), memory, calls
