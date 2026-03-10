from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from trtship.config import ModelConfig, OptimizationProfile
from trtship.errors import ModelError
from trtship.models import ModelSignature, infer_signature, load_model
from trtship.specs import DType

F = "trtship_fixtures.models"
X = {"name": "x", "shape": ["batch", 16]}
TOKENS = [
    {"name": "input_ids", "dtype": "int64", "shape": ["batch", "seq"], "value_range": [0, 100]},
    {"name": "attention_mask", "dtype": "int64", "shape": ["batch", "seq"], "value_range": [0, 2]},
]


def build(factory: str, inputs: list[dict[str, Any]], **extra: Any) -> ModelConfig:
    return ModelConfig.model_validate(
        {"name": "m", "kind": "module", "factory": f"{F}:{factory}", "inputs": inputs, **extra},
        context={"base_dir": Path.cwd()},
    )


def profile(**ranges: tuple[list[int], list[int], list[int]]) -> list[OptimizationProfile]:
    return [
        OptimizationProfile.model_validate(
            {"inputs": {k: {"min": v[0], "opt": v[1], "max": v[2]} for k, v in ranges.items()}}
        )
    ]


def infer(config: ModelConfig, profiles: list[OptimizationProfile] | None = None) -> ModelSignature:
    return infer_signature(load_model(config), config, profiles or [])


def test_mlp_with_dynamic_batch_and_profile() -> None:
    sig = infer(build("tiny_mlp", [X]), profile(x=([1, 16], [4, 16], [8, 16])))
    assert [i.name for i in sig.inputs] == ["x"]
    (out,) = sig.outputs
    assert out.name == "output"
    assert out.dtype is DType.FLOAT32
    assert out.shape == ["batch", 4]
    assert sig.probe_sizes == [{"batch": 4}, {"batch": 1}]  # opt vs. min


def test_mlp_without_a_profile_uses_distinct_probe_sizes() -> None:
    sig = infer(build("tiny_mlp", [X]))
    assert sig.outputs[0].shape == ["batch", 4]
    a, b = sig.probe_sizes
    assert a["batch"] != b["batch"]


def test_static_model_runs_once() -> None:
    sig = infer(build("tiny_mlp", [{"name": "x", "shape": [3, 16]}]))
    assert sig.outputs[0].shape == [3, 4]
    assert len(sig.probe_sizes) == 1


def test_profile_pinning_every_symbol_runs_once() -> None:
    sig = infer(build("tiny_mlp", [X]), profile(x=([2, 16], [2, 16], [2, 16])))
    assert len(sig.probe_sizes) == 1
    assert sig.outputs[0].shape == [2, 4]  # cannot observe variation, so it is treated as static


def test_multi_input_multi_output_with_dynamic_sequence() -> None:
    sig = infer(
        build("token_classifier", TOKENS, output_names=["logits", "hidden"]),
        profile(
            input_ids=([1, 8], [4, 32], [16, 128]),
            attention_mask=([1, 8], [4, 32], [16, 128]),
        ),
    )
    by_name = {o.name: o for o in sig.outputs}
    assert list(by_name) == ["logits", "hidden"]
    assert by_name["logits"].shape == ["batch", 3]
    assert by_name["hidden"].shape == ["batch", "seq", 8]
    assert sig.inputs[0].value_range == (0, 100)


def test_dict_outputs_keep_their_keys() -> None:
    sig = infer(build("dict_output", [{"name": "x", "shape": ["batch", 6]}]))
    by_name = {o.name: o for o in sig.outputs}
    assert list(by_name) == ["doubled", "total"]  # None value dropped
    assert by_name["doubled"].shape == ["batch", 6]
    assert by_name["total"].shape == ["batch", 1]


def test_dimension_no_single_symbol_explains_gets_a_synthetic_symbol() -> None:
    sig = infer(build("flatten_batch_seq", [{"name": "x", "shape": ["batch", "seq", 4]}]))
    assert sig.outputs[0].shape == ["output_dim0", 4]


@pytest.mark.parametrize(
    ("factory", "fragment"),
    [
        ("scalar_output", "scalar"),
        ("float64_output", "unsupported tensor dtype"),
        ("non_tensor_output", "must all be tensors"),
    ],
)
def test_unsupported_outputs_are_rejected(factory: str, fragment: str) -> None:
    with pytest.raises(ModelError, match=fragment):
        infer(build(factory, [{"name": "x", "shape": ["batch", 4]}]))


def test_rank_changing_outputs_are_rejected() -> None:
    config = build("rank_shifting", [{"name": "x", "shape": ["batch", 4]}])
    with pytest.raises(ModelError, match="changes rank"):
        infer(config, profile(x=([1, 4], [4, 4], [8, 4])))


def test_forward_failures_explain_the_probe_sizes_used() -> None:
    config = build("exploding_forward", [{"name": "x", "shape": ["batch", 4]}])
    with pytest.raises(ModelError, match="shape mismatch in layer 3") as info:
        infer(config)
    assert "probe sizes used" in (info.value.hint or "")
    assert "batch" in info.value.details["symbol_sizes"]


def test_signature_is_json_serializable() -> None:
    sig = infer(build("tiny_mlp", [X]))
    again = ModelSignature.model_validate_json(sig.model_dump_json())
    assert again == sig
