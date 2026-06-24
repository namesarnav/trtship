"""TensorRTExecutor's binding logic against tests/fakes/fake_tensorrt.py (not real TensorRT)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.fakes.fake_tensorrt import (
    DataType,
    FakeBuilderConfig,
    FakeOptions,
    FakeProfile,
    HostMemory,
    TensorSpec,
    make_fake_trt,
)
from trtship.errors import EngineRuntimeError
from trtship.tensorrt import TensorRTExecutor


def double_first_four(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"output": inputs["x"][:, :4] * 2}


def executor(
    *,
    options: FakeOptions | None = None,
    profile: tuple[tuple[int, ...], ...] = ((1, 16), (4, 16), (8, 16)),
) -> tuple[TensorRTExecutor, HostMemory, Any]:
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


def test_run_returns_the_engine_outputs() -> None:
    engine, _, _ = executor()
    x = np.random.default_rng(0).standard_normal((3, 16)).astype(np.float32)
    out = engine.run({"x": x})
    assert set(out) == {"output"}
    np.testing.assert_allclose(out["output"], x[:, :4] * 2)
    assert engine.info.inputs[0].name == "x"


def test_inputs_are_cast_to_the_engine_dtype() -> None:
    engine, memory, _ = executor()
    out = engine.run({"x": np.ones((2, 16), dtype=np.float64)})
    assert out["output"].dtype == np.float32
    assert all(a.dtype == np.float32 for a in memory.by_ptr.values())


def test_int64_bindings_are_honored() -> None:
    options = FakeOptions(
        engine_inputs=[TensorSpec("x", (-1, 16), DataType.INT32)],
        engine_outputs=[TensorSpec("output", (-1, 4), DataType.INT32)],
        compute=lambda inputs: {"output": inputs["x"][:, :4] + 1},
    )
    engine, _, _ = executor(options=options)
    out = engine.run({"x": np.arange(32, dtype=np.int64).reshape(2, 16)})
    assert out["output"].dtype == np.int32
    assert out["output"][0].tolist() == [1, 2, 3, 4]


def test_shapes_outside_the_profile_are_reported_with_the_ranges() -> None:
    engine, _, _ = executor()
    with pytest.raises(
        EngineRuntimeError, match="outside the engine's optimization profile"
    ) as info:
        engine.run({"x": np.zeros((9, 16), dtype=np.float32)})  # max is 8
    assert info.value.details["profile_ranges"][0]["max"] == [8, 16]
    assert "tensorrt.profiles" in (info.value.hint or "")
    assert info.value.exit_code == 7


def test_static_engines_require_their_exact_shape() -> None:
    options = FakeOptions(
        engine_inputs=[TensorSpec("x", (3, 16))],
        engine_outputs=[TensorSpec("output", (3, 4))],
        compute=lambda inputs: {"output": inputs["x"][:, :4]},
    )
    engine, _, _ = executor(options=options)
    assert engine.run({"x": np.zeros((3, 16), np.float32)})["output"].shape == (3, 4)
    with pytest.raises(EngineRuntimeError, match="fixed to"):
        engine.run({"x": np.zeros((2, 16), np.float32)})


def test_missing_inputs_are_named() -> None:
    engine, _, _ = executor()
    with pytest.raises(EngineRuntimeError, match="missing input 'x'") as info:
        engine.run({})
    assert info.value.details["expected"] == ["x"]


def test_data_dependent_output_shapes_are_rejected() -> None:
    engine, _, _ = executor(options=FakeOptions(data_dependent_outputs=True))
    with pytest.raises(EngineRuntimeError, match="data-dependent shape"):
        engine.run({"x": np.zeros((2, 16), np.float32)})


def test_execution_failures_are_reported() -> None:
    engine, _, _ = executor(options=FakeOptions(execute_fails=True))
    with pytest.raises(EngineRuntimeError, match=r"failed to execute engine test\.plan"):
        engine.run({"x": np.zeros((2, 16), np.float32)})


def test_unloadable_plans_explain_the_version_constraint() -> None:
    with pytest.raises(EngineRuntimeError, match="cannot load TensorRT engine") as info:
        executor(options=FakeOptions(deserialize_fails=True))
    assert "same TensorRT version" in (info.value.hint or "") or "TensorRT version" in (
        info.value.hint or ""
    )


def test_context_creation_failures_are_reported() -> None:
    with pytest.raises(EngineRuntimeError, match="cannot create an execution context"):
        executor(options=FakeOptions(context_fails=True))


def test_a_bound_execution_can_be_repeated_without_re_uploading() -> None:
    engine, memory, _ = executor()
    x = np.ones((2, 16), dtype=np.float32)
    bound = engine.bind({"x": x})
    allocations = len(memory.by_ptr)
    for _ in range(3):
        bound.execute()
    bound.synchronize()
    assert len(memory.by_ptr) == allocations  # nothing was allocated or copied per execution
    np.testing.assert_allclose(bound.outputs()["output"], x[:, :4] * 2)
    context = next(iter(c for c in [engine._context]))
    assert context.executions == 3


def test_the_executor_is_a_context_manager() -> None:
    engine, _, _ = executor()
    with engine as active:
        assert active.run({"x": np.zeros((1, 16), np.float32)})["output"].shape == (1, 4)
    assert engine._context is None
