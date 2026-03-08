from __future__ import annotations

import pytest
from pydantic import ValidationError

from trtship.specs import DType, TensorSpec


def test_static_and_dynamic_axes() -> None:
    spec = TensorSpec(name="ids", dtype=DType.INT64, shape=["batch", "seq"])
    assert spec.is_dynamic
    assert spec.dynamic_axes == {0: "batch", 1: "seq"}
    assert spec.symbols == frozenset({"batch", "seq"})

    static = TensorSpec(name="x", shape=[1, 3, 224, 224])
    assert not static.is_dynamic
    assert static.dynamic_axes == {}


def test_concrete_shape_resolves_symbols_and_requires_all() -> None:
    spec = TensorSpec(name="x", shape=["batch", 16, "seq"])
    assert spec.concrete_shape({"batch": 4, "seq": 7}) == (4, 16, 7)
    with pytest.raises(KeyError, match="seq"):
        spec.concrete_shape({"batch": 4})


@pytest.mark.parametrize("shape", [[], [0], [-1, 2], [True, 2], ["1bad"], ["a-b"], [1.5]])
def test_bad_shapes_rejected(shape: list[object]) -> None:
    with pytest.raises(ValidationError):
        TensorSpec(name="x", shape=shape)


@pytest.mark.parametrize("name", ["", "1x", "a b", "a-b"])
def test_bad_names_rejected(name: str) -> None:
    with pytest.raises(ValidationError):
        TensorSpec(name=name, shape=[1])


def test_value_range_rules() -> None:
    assert TensorSpec(name="x", shape=[1], value_range=(0.0, 1.0)).value_range == (0.0, 1.0)
    assert TensorSpec(name="i", dtype=DType.INT64, shape=[1], value_range=(0, 30522)).value_range
    with pytest.raises(ValidationError, match="low < high"):
        TensorSpec(name="x", shape=[1], value_range=(1.0, 1.0))
    with pytest.raises(ValidationError, match="whole numbers"):
        TensorSpec(name="i", dtype=DType.INT32, shape=[1], value_range=(0.5, 10))
    with pytest.raises(ValidationError, match="bool"):
        TensorSpec(name="m", dtype=DType.BOOL, shape=[1], value_range=(0, 1))


def test_dtype_categories() -> None:
    assert DType.FLOAT16.is_floating
    assert DType.INT8.is_integer
    assert not DType.BOOL.is_integer
    assert not DType.BOOL.is_floating


def test_specs_are_immutable_and_reject_unknown_fields() -> None:
    spec = TensorSpec(name="x", shape=[1])
    with pytest.raises(ValidationError):
        spec.name = "y"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        TensorSpec(name="x", shape=[1], bogus=1)  # type: ignore[call-arg]
