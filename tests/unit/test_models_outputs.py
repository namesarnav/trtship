from __future__ import annotations

import pytest
import torch

from trtship.errors import ModelError
from trtship.models import flatten_outputs

T = torch.zeros(2, 3)


def test_single_tensor_default_name() -> None:
    assert list(flatten_outputs(T)) == ["output"]


def test_single_tensor_declared_name() -> None:
    assert list(flatten_outputs(T, ["logits"])) == ["logits"]


@pytest.mark.parametrize("container", [tuple, list])
def test_sequences_are_named_positionally(container: type) -> None:
    out = flatten_outputs(container([T, T + 1]))
    assert list(out) == ["output_0", "output_1"]
    assert list(flatten_outputs(container([T, T + 1]), ["a", "b"])) == ["a", "b"]


def test_declared_count_must_match() -> None:
    with pytest.raises(ModelError, match="2 names but the model returned 1"):
        flatten_outputs(T, ["a", "b"])
    with pytest.raises(ModelError, match="1 names but the model returned 2"):
        flatten_outputs((T, T), ["a"])


def test_dict_keys_are_used_and_none_is_dropped() -> None:
    out = flatten_outputs({"logits": T, "aux": None, "hidden": T + 1})
    assert list(out) == ["logits", "hidden"]


def test_dict_declared_names_select_and_order() -> None:
    out = flatten_outputs({"a": T, "b": T + 1}, ["b", "a"])
    assert list(out) == ["b", "a"]
    with pytest.raises(ModelError, match="do not match"):
        flatten_outputs({"a": T, "b": T}, ["a", "zzz"])
    with pytest.raises(ModelError, match="do not match"):
        flatten_outputs({"a": T, "b": T}, ["a"])


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        ("text", "unsupported model output type"),
        (None, "unsupported model output type"),
        ((T, 3), "must all be tensors"),
        ({"a": T, "b": [1]}, "must all be tensors"),
        ((), "no tensor outputs"),
        ({}, "no tensor outputs"),
    ],
)
def test_unsupported_outputs(raw: object, fragment: str) -> None:
    with pytest.raises(ModelError, match=fragment):
        flatten_outputs(raw)
