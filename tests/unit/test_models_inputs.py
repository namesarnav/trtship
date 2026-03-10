from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from trtship.config import ModelConfig, OptimizationProfile
from trtship.errors import ConfigError, ModelError
from trtship.models import make_input, make_inputs, resolve_symbol_sizes
from trtship.specs import DType, TensorSpec


def spec(**kw: Any) -> TensorSpec:
    kw.setdefault("name", "x")
    kw.setdefault("shape", [4, 8])
    return TensorSpec(**kw)


def test_inputs_are_deterministic_per_seed_and_name() -> None:
    a = make_input(spec(), {}, seed=1)
    assert torch.equal(a, make_input(spec(), {}, seed=1))
    assert not torch.equal(a, make_input(spec(), {}, seed=2))
    assert not torch.equal(a, make_input(spec(name="y"), {}, seed=1))


def test_adding_an_input_does_not_change_existing_data() -> None:
    one = make_inputs([spec(name="a")], {}, seed=3)
    two = make_inputs([spec(name="b"), spec(name="a")], {}, seed=3)
    assert torch.equal(one["a"], two["a"])


def test_symbolic_dims_are_resolved() -> None:
    tensor = make_input(spec(shape=["batch", 8]), {"batch": 5}, seed=0)
    assert tuple(tensor.shape) == (5, 8)


def test_floats_default_to_standard_normal_and_respect_ranges() -> None:
    big = make_input(spec(shape=[20000]), {}, seed=0)
    assert abs(big.mean().item()) < 0.05
    assert abs(big.std().item() - 1.0) < 0.05
    ranged = make_input(spec(shape=[5000], value_range=(-2.0, 3.0)), {}, seed=0)
    assert ranged.min() >= -2.0
    assert ranged.max() < 3.0


def test_float16_dtype_is_applied() -> None:
    assert make_input(spec(dtype=DType.FLOAT16), {}, seed=0).dtype == torch.float16


@pytest.mark.parametrize(
    ("dtype", "torch_dtype"),
    [
        (DType.INT64, torch.int64),
        (DType.INT32, torch.int32),
        (DType.INT8, torch.int8),
        (DType.UINT8, torch.uint8),
    ],
)
def test_integer_dtypes_and_default_range(dtype: DType, torch_dtype: torch.dtype) -> None:
    tensor = make_input(spec(dtype=dtype, shape=[500]), {}, seed=0)
    assert tensor.dtype == torch_dtype
    assert set(tensor.tolist()) <= {0, 1}


def test_integer_value_range_is_half_open() -> None:
    tensor = make_input(spec(dtype=DType.INT64, shape=[4000], value_range=(5, 9)), {}, seed=0)
    assert tensor.min() >= 5
    assert tensor.max() <= 8
    assert set(tensor.tolist()) == {5, 6, 7, 8}


def test_integer_range_must_fit_the_dtype() -> None:
    with pytest.raises(ModelError, match="does not fit int8"):
        make_input(spec(dtype=DType.INT8, value_range=(0, 1000)), {}, seed=0)
    with pytest.raises(ModelError, match="does not fit uint8"):
        make_input(spec(dtype=DType.UINT8, value_range=(-1, 10)), {}, seed=0)


def test_bool_inputs() -> None:
    tensor = make_input(spec(dtype=DType.BOOL, shape=[1000]), {}, seed=0)
    assert tensor.dtype == torch.bool
    assert 0 < int(tensor.sum()) < 1000


def test_make_inputs_moves_to_device() -> None:
    out = make_inputs([spec()], {}, seed=0, device="cpu")
    assert out["x"].device.type == "cpu"


# --------------------------------------------------------------------------- symbol sizes


def model_cfg(inputs: list[dict[str, Any]]) -> ModelConfig:
    return ModelConfig.model_validate(
        {"name": "m", "kind": "module", "factory": "a.b:c", "inputs": inputs},
        context={"base_dir": Path.cwd()},
    )


def profile(**ranges: tuple[list[int], list[int], list[int]]) -> OptimizationProfile:
    return OptimizationProfile.model_validate(
        {"inputs": {k: {"min": v[0], "opt": v[1], "max": v[2]} for k, v in ranges.items()}}
    )


IDS = {"name": "ids", "dtype": "int64", "shape": ["batch", "seq"]}
MASK = {"name": "mask", "dtype": "int64", "shape": ["batch", "seq"]}


def test_sizes_come_from_the_selected_profile_shape() -> None:
    model = model_cfg([IDS, MASK])
    prof = profile(
        ids=([1, 8], [4, 32], [16, 128]),
        mask=([1, 8], [4, 32], [16, 128]),
    )
    assert resolve_symbol_sizes(model, [prof], "min") == {"batch": 1, "seq": 8}
    assert resolve_symbol_sizes(model, [prof], "opt") == {"batch": 4, "seq": 32}
    assert resolve_symbol_sizes(model, [prof], "max") == {"batch": 16, "seq": 128}


def test_conflicting_symbol_sizes_across_inputs_are_a_config_error() -> None:
    model = model_cfg([IDS, MASK])
    prof = profile(ids=([1, 8], [4, 32], [16, 128]), mask=([1, 8], [4, 64], [16, 128]))
    with pytest.raises(ConfigError, match="different opt sizes"):
        resolve_symbol_sizes(model, [prof], "opt")


def test_uncovered_symbols_use_fallback_then_distinct_probe_sizes() -> None:
    model = model_cfg([IDS])
    sizes = resolve_symbol_sizes(model, [], "opt")
    assert set(sizes) == {"batch", "seq"}
    assert sizes["batch"] != sizes["seq"]  # distinct, so output dims attribute unambiguously
    assert resolve_symbol_sizes(model, [], "opt", fallback={"batch": 9})["batch"] == 9


def test_partial_profile_fills_remaining_symbols() -> None:
    model = model_cfg([{"name": "x", "shape": ["batch", "width", 4]}])
    prof = profile(x=([1, 3, 4], [2, 3, 4], [8, 3, 4]))
    sizes = resolve_symbol_sizes(model, [prof], "opt")
    assert sizes["batch"] == 2
    assert sizes["width"] == 3  # width is pinned by the profile too (min=opt=max)


def test_static_model_has_no_symbols() -> None:
    assert resolve_symbol_sizes(model_cfg([{"name": "x", "shape": [2, 3]}]), [], "opt") == {}
