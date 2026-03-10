from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from trtship.config import ModelConfig
from trtship.errors import EnvironmentUnavailableError, ModelError
from trtship.models import LoadedModel, hash_state_dict, load_model, resolve_device
from trtship.models.loader import _clip
from trtship.utils import env

FIXTURES = "trtship_fixtures.models"


def cfg(**overrides: Any) -> ModelConfig:
    data: dict[str, Any] = {
        "name": "tiny",
        "kind": "module",
        "factory": f"{FIXTURES}:tiny_mlp",
        "inputs": [{"name": "x", "shape": ["batch", 16]}],
    }
    data.update(overrides)
    return ModelConfig.model_validate(data, context={"base_dir": Path.cwd()})


def x_input(batch: int = 3) -> dict[str, torch.Tensor]:
    return {"x": torch.randn(batch, 16)}


# --------------------------------------------------------------------------- module kind


def test_module_kind_loads_in_eval_mode_on_cpu() -> None:
    model = load_model(cfg())
    assert isinstance(model, LoadedModel)
    assert not model.module.training
    assert model.device == torch.device("cpu")
    assert model.kind.value == "module"
    assert model.torch_version == str(torch.__version__)
    assert len(model.weights_sha256) == 64
    assert model.run(x_input(5))["output"].shape == (5, 4)


def test_factory_kwargs_are_passed() -> None:
    model = load_model(cfg(factory_kwargs={"out_features": 7}))
    assert model.run(x_input())["output"].shape == (3, 7)


def test_weights_hash_is_deterministic_and_seed_sensitive() -> None:
    a = load_model(cfg(factory_kwargs={"seed": 1})).weights_sha256
    b = load_model(cfg(factory_kwargs={"seed": 1})).weights_sha256
    c = load_model(cfg(factory_kwargs={"seed": 2})).weights_sha256
    assert a == b
    assert a != c


@pytest.mark.parametrize(
    ("factory", "fragment"),
    [
        ("no_such_pkg.models:tiny_mlp", "cannot import factory module"),
        (f"{FIXTURES}:nope", "no attribute"),
        (f"{FIXTURES}:not_callable", "not callable"),
        (f"{FIXTURES}:returns_not_a_module", "expected torch.nn.Module"),
        (f"{FIXTURES}:raises_on_build", "ValueError: bad hyperparameters"),
    ],
)
def test_factory_failures_are_model_errors(factory: str, fragment: str) -> None:
    with pytest.raises(ModelError, match=fragment):
        load_model(cfg(factory=factory))


def test_wrong_factory_kwargs_are_reported() -> None:
    with pytest.raises(ModelError, match="TypeError"):
        load_model(cfg(factory_kwargs={"bogus": 1}))


# --------------------------------------------------------------------------- checkpoint kind


def _save(tmp_path: Path, obj: Any, name: str = "ckpt.pt") -> Path:
    path = tmp_path / name
    torch.save(obj, path)
    return path


def _ckpt_cfg(path: Path, **extra: Any) -> ModelConfig:
    return cfg(kind="checkpoint", path=str(path), **extra)


def test_checkpoint_loads_weights_and_matches_source_hash(tmp_path: Path) -> None:
    source = load_model(cfg(factory_kwargs={"seed": 5}))
    path = _save(tmp_path, source.module.state_dict())
    loaded = load_model(_ckpt_cfg(path, factory_kwargs={"seed": 99}))  # different init
    assert loaded.weights_sha256 == source.weights_sha256  # identity follows weights, not format
    x = x_input()
    assert torch.equal(loaded.run(x)["output"], source.run(x)["output"])
    assert loaded.source_path == path


@pytest.mark.parametrize("wrapper", ["state_dict", "model_state_dict", "model"])
def test_checkpoint_unwraps_common_containers(tmp_path: Path, wrapper: str) -> None:
    source = load_model(cfg())
    path = _save(tmp_path, {wrapper: source.module.state_dict(), "epoch": 3})
    assert load_model(_ckpt_cfg(path)).weights_sha256 == source.weights_sha256


def test_checkpoint_strips_dataparallel_prefix(tmp_path: Path) -> None:
    source = load_model(cfg())
    prefixed = OrderedDict((f"module.{k}", v) for k, v in source.module.state_dict().items())
    path = _save(tmp_path, prefixed)
    assert load_model(_ckpt_cfg(path)).weights_sha256 == source.weights_sha256


def test_checkpoint_shape_mismatch_is_a_model_error(tmp_path: Path) -> None:
    other = load_model(cfg(factory_kwargs={"out_features": 9}))
    path = _save(tmp_path, other.module.state_dict())
    with pytest.raises(ModelError, match="does not fit the model"):
        load_model(_ckpt_cfg(path))  # default out_features=4


def test_checkpoint_key_mismatch_lists_keys(tmp_path: Path) -> None:
    path = _save(tmp_path, {"totally.different": torch.zeros(1)})
    with pytest.raises(ModelError, match="keys do not match") as info:
        load_model(_ckpt_cfg(path))
    assert "totally.different" in info.value.details["unexpected_keys"]
    assert info.value.details["missing_keys"]


def test_long_key_lists_are_clipped() -> None:
    clipped = _clip([f"layer{i}.weight" for i in range(25)])
    assert len(clipped) == 11
    assert clipped[0] == "layer0.weight"
    assert clipped[-1] == "... and 15 more"
    assert _clip(["a", "b"]) == ["a", "b"]


def test_checkpoint_with_non_tensor_entries_is_rejected(tmp_path: Path) -> None:
    path = _save(tmp_path, {"lr": 0.1, "step": 4})
    with pytest.raises(ModelError, match="not a plain state dict"):
        load_model(_ckpt_cfg(path))


def test_checkpoint_that_is_not_a_mapping(tmp_path: Path) -> None:
    path = _save(tmp_path, torch.zeros(3))
    with pytest.raises(ModelError, match="expected a state dict"):
        load_model(_ckpt_cfg(path))


class _Pickled:
    """A non-tensor object: weights_only loading must refuse it."""

    def __init__(self) -> None:
        self.payload = [1, 2, 3]


def test_unsafe_pickles_are_refused_without_trust_source(tmp_path: Path) -> None:
    path = _save(tmp_path, {"obj": _Pickled()})
    with pytest.raises(ModelError, match="cannot load checkpoint") as info:
        load_model(_ckpt_cfg(path))
    assert "trust_source" in (info.value.hint or "")


def test_trust_source_allows_full_model_pickles(tmp_path: Path) -> None:
    source = load_model(cfg())
    path = _save(tmp_path, source.module)  # whole nn.Module, not a state dict
    with pytest.raises(ModelError):
        load_model(_ckpt_cfg(path))  # refused by default
    loaded = load_model(_ckpt_cfg(path, trust_source=True))
    assert loaded.weights_sha256 == source.weights_sha256


def test_missing_and_corrupt_checkpoint_files(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="not found"):
        load_model(_ckpt_cfg(tmp_path / "missing.pt"))
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(ModelError, match="cannot load checkpoint"):
        load_model(_ckpt_cfg(bad))


# --------------------------------------------------------------------------- torchscript kind


def _save_script(module: nn.Module, path: Path) -> None:
    scripted = torch.jit.script(module)
    scripted.save(str(path))  # type: ignore[no-untyped-call]


def _script_cfg(path: Path, **extra: Any) -> ModelConfig:
    return cfg(kind="torchscript", path=str(path), factory=None, **extra)


def test_torchscript_requires_trust_source(tmp_path: Path) -> None:
    path = tmp_path / "m.pt"
    _save_script(load_model(cfg()).module, path)
    with pytest.raises(ModelError, match=r"requires model\.trust_source") as info:
        load_model(_script_cfg(path))
    assert "executable content" in (info.value.hint or "")


def test_torchscript_loads_and_matches_eager(tmp_path: Path) -> None:
    eager = load_model(cfg())
    path = tmp_path / "m.pt"
    _save_script(eager.module, path)
    scripted = load_model(_script_cfg(path, trust_source=True))
    x = x_input()
    assert torch.allclose(scripted.run(x)["output"], eager.run(x)["output"])
    assert scripted.weights_sha256 == eager.weights_sha256


def test_corrupt_torchscript_is_a_model_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"garbage")
    with pytest.raises(ModelError, match="cannot load TorchScript"):
        load_model(_script_cfg(bad, trust_source=True))
    with pytest.raises(ModelError, match="not found"):
        load_model(_script_cfg(tmp_path / "none.pt", trust_source=True))


# --------------------------------------------------------------------------- device


def test_cuda_request_without_a_gpu_raises_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        env, "probe_nvidia_gpu", lambda: ([], env._missing(env.NVIDIA_GPU, "no driver"))
    )
    with pytest.raises(EnvironmentUnavailableError, match="cuda"):
        load_model(cfg(), device="cuda")
    with pytest.raises(EnvironmentUnavailableError):
        resolve_device("cuda:1")


def test_unsupported_device_type() -> None:
    with pytest.raises(ModelError, match="unsupported device"):
        resolve_device("meta")


# --------------------------------------------------------------------------- run()


def test_run_reports_missing_inputs_and_forward_failures() -> None:
    model = load_model(cfg())
    with pytest.raises(ModelError, match="missing model inputs"):
        model.run({})
    with pytest.raises(ModelError, match="forward pass failed") as info:
        model.run({"x": torch.randn(2, 5)})  # wrong feature size
    assert info.value.details["input_shapes"] == {"x": [2, 5]}
    assert "value_range" in (info.value.hint or "")


def test_run_uses_declared_output_names() -> None:
    model = load_model(
        cfg(
            factory=f"{FIXTURES}:token_classifier",
            inputs=[
                {"name": "input_ids", "dtype": "int64", "shape": ["batch", "seq"]},
                {"name": "attention_mask", "dtype": "int64", "shape": ["batch", "seq"]},
            ],
            output_names=["logits", "hidden"],
        )
    )
    out = model.run(
        {
            "input_ids": torch.randint(0, 100, (2, 5)),
            "attention_mask": torch.ones(2, 5, dtype=torch.int64),
        }
    )
    assert list(out) == ["logits", "hidden"]
    assert out["logits"].shape == (2, 3)
    assert out["hidden"].shape == (2, 5, 8)


# --------------------------------------------------------------------------- hashing


def test_hash_state_dict_properties() -> None:
    base = {"a": torch.arange(6, dtype=torch.float32).reshape(2, 3), "b": torch.ones(2)}
    same_reordered = {"b": torch.ones(2), "a": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    assert hash_state_dict(base) == hash_state_dict(same_reordered)

    changed = {**base, "b": torch.ones(2) * 2}
    assert hash_state_dict(base) != hash_state_dict(changed)
    renamed = {"a": base["a"], "c": base["b"]}
    assert hash_state_dict(base) != hash_state_dict(renamed)
    reshaped = {"a": base["a"].reshape(3, 2), "b": base["b"]}
    assert hash_state_dict(base) != hash_state_dict(reshaped)
    retyped = {"a": base["a"].to(torch.float16), "b": base["b"]}
    assert hash_state_dict(base) != hash_state_dict(retyped)


def test_hash_state_dict_handles_odd_tensors() -> None:
    odd = {
        "empty": torch.zeros(0, 3),
        "flag": torch.tensor([True, False]),
        "half": torch.ones(2, dtype=torch.float16),
        "bf16": torch.ones(2, dtype=torch.bfloat16),
        "strided": torch.arange(12.0).reshape(3, 4).t(),  # non-contiguous
    }
    assert len(hash_state_dict(odd)) == 64
    contiguous = {**odd, "strided": odd["strided"].contiguous()}
    assert hash_state_dict(odd) == hash_state_dict(contiguous)
