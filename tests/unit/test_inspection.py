from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from trtship.config import ModelConfig, ModelKind, OptimizationProfile
from trtship.models import LoadedModel, hash_state_dict, infer_signature, load_model
from trtship.models.inspection import ModelReport, inspect_model
from trtship.specs import TensorSpec

F = "trtship_fixtures.models"


def config_for(inputs: list[dict[str, Any]], factory: str = "tiny_mlp") -> ModelConfig:
    return ModelConfig.model_validate(
        {"name": "m", "kind": "module", "factory": f"{F}:{factory}", "inputs": inputs},
        context={"base_dir": Path.cwd()},
    )


def wrap(module: nn.Module, inputs: list[dict[str, Any]]) -> tuple[LoadedModel, ModelConfig]:
    """Build a LoadedModel around an arbitrary module (bypassing factory import)."""
    module.eval()
    config = config_for(inputs)
    model = LoadedModel(
        module=module,
        name="m",
        kind=ModelKind.MODULE,
        inputs=tuple(TensorSpec.model_validate(i) for i in inputs),
        declared_output_names=None,
        device=torch.device("cpu"),
        source_path=None,
        weights_sha256=hash_state_dict(module.state_dict()),
        torch_version=str(torch.__version__),
    )
    return model, config


X16 = [{"name": "x", "shape": ["batch", 16]}]
PROFILE = [
    OptimizationProfile.model_validate(
        {"inputs": {"x": {"min": [1, 16], "opt": [4, 16], "max": [8, 16]}}}
    )
]


def mlp_report(**kwargs: Any) -> ModelReport:
    model = load_model(config_for(X16))
    return inspect_model(model, config_for(X16), PROFILE, **kwargs)


def test_mlp_parameter_counts_are_exact() -> None:
    report = mlp_report()
    p = report.parameters
    assert p.total == 16 * 32 + 32 + 32 * 4 + 4 == 676
    assert p.trainable == 676
    assert p.frozen == 0
    assert p.tensors == 4
    assert p.by_dtype == {"torch.float32": 676}
    assert p.buffer_tensors == 0
    assert report.memory.parameter_bytes == 676 * 4
    assert report.memory.weights_bytes == 676 * 4
    assert report.module_count == 4  # Sequential + Linear + ReLU + Linear


def test_largest_tensors_are_sorted_and_limited() -> None:
    report = mlp_report(top=2)
    assert [t.name for t in report.parameters.largest] == ["0.weight", "2.weight"]
    first = report.parameters.largest[0]
    assert first.shape == [32, 16]
    assert first.numel == 512
    assert first.bytes == 2048
    assert first.requires_grad is True


def test_frozen_parameters_are_separated() -> None:
    model = load_model(config_for(X16))
    for param in model.module[0].parameters():  # type: ignore[index]
        param.requires_grad_(False)
    report = inspect_model(model, config_for(X16), PROFILE)
    assert report.parameters.trainable == 132
    assert report.parameters.frozen == 544
    first, _relu, last = report.architecture.children
    assert first.trainable_parameters == 0
    assert last.trainable_parameters == 132


def test_shared_parameters_are_counted_once() -> None:
    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 4)
            self.b.weight = self.a.weight

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.b(self.a(x))
            return out

    model, config = wrap(Tied(), [{"name": "x", "shape": ["batch", 4]}])
    report = inspect_model(model, config)
    assert report.parameters.total == 16 + 4 + 4  # one weight, two biases
    assert report.parameters.tensors == 3


def test_half_precision_halves_the_bytes() -> None:
    half_inputs = [{"name": "x", "dtype": "float16", "shape": ["batch", 16]}]
    config = config_for(half_inputs)
    model = load_model(config)
    model.module.half()
    report = inspect_model(model, config, PROFILE)
    assert report.parameters.by_dtype == {"torch.float16": 676}
    assert report.memory.parameter_bytes == 676 * 2


def test_buffers_are_reported_separately_from_parameters() -> None:
    module = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8))
    model, config = wrap(module, [{"name": "x", "shape": ["batch", 4]}])
    report = inspect_model(model, config)
    p, m = report.parameters, report.memory
    assert p.total == (4 * 8 + 8) + 16  # linear + bn weight/bias
    assert p.buffer_tensors == 3  # running_mean, running_var, num_batches_tracked
    assert p.buffer_elements == 8 + 8 + 1
    assert m.buffer_bytes == 8 * 4 + 8 * 4 + 8  # two float32 vectors + one int64
    assert m.weights_bytes == m.parameter_bytes + m.buffer_bytes


def test_tree_structure_and_depth_limit() -> None:
    module = nn.Sequential(nn.Sequential(nn.Linear(2, 2), nn.ReLU()), nn.Linear(2, 2))
    model, config = wrap(module, [{"name": "x", "shape": ["batch", 2]}])

    full = inspect_model(model, config, max_depth=5).architecture
    assert [c.name for c in full.children] == ["0", "1"]
    assert [c.name for c in full.children[0].children] == ["0.0", "0.1"]
    assert full.elided_modules == 0
    assert full.parameters == sum(c.parameters for c in full.children) == 12

    shallow = inspect_model(model, config, max_depth=1).architecture
    inner = shallow.children[0]
    assert inner.children == []
    assert inner.elided_modules == 2  # Linear + ReLU hidden
    assert inner.parameters == 6  # still counted

    root_only = inspect_model(model, config, max_depth=0).architecture
    assert root_only.children == []
    assert root_only.elided_modules == 4


def test_activation_bound_is_the_sum_of_leaf_outputs_at_the_opt_shape() -> None:
    report = mlp_report()
    m = report.memory
    batch = 4  # profile opt
    assert m.activation_probe_sizes == {"batch": batch}
    # Linear(16->32) + ReLU(32) + Linear(32->4) outputs, float32
    assert m.activation_bytes_upper_bound == (batch * 32 + batch * 32 + batch * 4) * 4
    assert m.notes == []


def test_activation_probe_without_a_profile_uses_reported_sizes() -> None:
    model = load_model(config_for(X16))
    report = inspect_model(model, config_for(X16))
    batch = report.memory.activation_probe_sizes["batch"]  # type: ignore[index]
    assert report.memory.activation_bytes_upper_bound == (batch * 68) * 4


def test_activation_bound_counts_tuple_outputs() -> None:
    class Leaf(nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x, x * 2

    model, config = wrap(Leaf(), [{"name": "x", "shape": [5, 3]}])
    report = inspect_model(model, config)
    assert report.memory.activation_bytes_upper_bound == 2 * 5 * 3 * 4


def test_scripted_modules_report_why_activations_are_unavailable(tmp_path: Path) -> None:
    scripted = torch.jit.script(load_model(config_for(X16)).module)
    model, config = wrap(scripted, X16)
    report = inspect_model(model, config)
    assert report.memory.activation_bytes_upper_bound is None
    assert report.memory.activation_probe_sizes is None
    assert "not supported on ScriptModules" in report.memory.notes[0]
    assert report.parameters.total == 676  # everything else still works


def test_activation_measurement_failure_does_not_fail_the_report() -> None:
    class FlakyOnThirdCall(nn.Module):
        calls = 0

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            FlakyOnThirdCall.calls += 1
            if FlakyOnThirdCall.calls == 3:  # signature inference makes calls 1-2
                raise RuntimeError("out of memory")
            out: torch.Tensor = self.linear(x)
            return out

    FlakyOnThirdCall.calls = 0
    model, config = wrap(FlakyOnThirdCall(), [{"name": "x", "shape": ["batch", 4]}])
    report = inspect_model(model, config)
    assert FlakyOnThirdCall.calls == 3
    assert report.memory.activation_bytes_upper_bound is None
    assert "could not be measured" in report.memory.notes[0]
    assert "out of memory" in report.memory.notes[0]
    assert report.parameters.total == 10


def test_a_precomputed_signature_avoids_extra_forward_passes() -> None:
    class Counting(nn.Module):
        calls = 0

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            Counting.calls += 1
            out: torch.Tensor = self.linear(x)
            return out

    Counting.calls = 0
    model, config = wrap(Counting(), [{"name": "x", "shape": ["batch", 4]}])
    signature = infer_signature(model, config)
    calls_for_signature = Counting.calls
    Counting.calls = 0
    inspect_model(model, config, signature=signature)
    assert calls_for_signature == 2
    assert Counting.calls == 1  # only the activation probe


def test_report_metadata_and_json_round_trip() -> None:
    model = load_model(config_for(X16))
    report = inspect_model(model, config_for(X16), PROFILE)
    assert report.schema_version == 1
    assert report.name == "m"
    assert report.kind == "module"
    assert report.weights_sha256 == model.weights_sha256
    assert report.torch_version == str(torch.__version__)
    assert report.device == "cpu"
    assert report.root_type == "Sequential"
    assert report.generated_at.tzinfo is not None
    assert [o.name for o in report.signature.outputs] == ["output"]
    assert report.signature.outputs[0].shape == ["batch", 4]
    assert ModelReport.model_validate_json(report.model_dump_json()) == report
