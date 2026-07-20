"""Generated config.pbtxt files, checked by parsing them with Triton's own protobuf schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2 as pb

from trtship.config import TritonConfig
from trtship.errors import ArtifactConflictError, ConfigError, TritonError
from trtship.specs import DType
from trtship.tensorrt import EngineInfo
from trtship.tensorrt.engine_info import ProfileRange, TensorBinding
from trtship.triton import build_repository, derive_max_batch_size, render_config_pbtxt


def binding(
    name: str,
    mode: str,
    shape: list[int],
    dtype: DType = DType.FLOAT32,
    profile: tuple[list[int], list[int], list[int]] | None = None,
) -> TensorBinding:
    profiles = [ProfileRange(min=profile[0], opt=profile[1], max=profile[2])] if profile else []
    return TensorBinding.model_validate(
        {
            "name": name,
            "mode": mode,
            "dtype": dtype,
            "shape": shape,
            "profiles": [p.model_dump() for p in profiles],
        }
    )


def dynamic_engine(max_batch: int = 8) -> EngineInfo:
    return EngineInfo(
        tensorrt_version="10.3.0",
        num_optimization_profiles=1,
        tensors=[
            binding("x", "input", [-1, 16], profile=([1, 16], [4, 16], [max_batch, 16])),
            binding("y", "output", [-1, 4]),
        ],
    )


def static_engine() -> EngineInfo:
    return EngineInfo(
        tensorrt_version="10.3.0",
        num_optimization_profiles=1,
        tensors=[binding("x", "input", [2, 16]), binding("y", "output", [2, 4])],
    )


def parse(text: str) -> Any:
    return text_format.Parse(text, pb.ModelConfig())


def test_a_dynamic_batch_engine_gets_batching_and_dims_without_the_batch_axis() -> None:
    config = parse(render_config_pbtxt("m", dynamic_engine(), TritonConfig()))
    assert config.name == "m"
    assert config.platform == "tensorrt_plan"
    assert config.default_model_filename == "model.plan"
    assert config.max_batch_size == 8
    assert [(i.name, i.data_type, list(i.dims)) for i in config.input] == [
        ("x", pb.TYPE_FP32, [16])
    ]
    assert [(o.name, list(o.dims)) for o in config.output] == [("y", [4])]
    group = config.instance_group[0]
    assert (group.count, group.kind, list(group.gpus)) == (1, pb.ModelInstanceGroup.KIND_GPU, [0])
    assert not config.HasField("dynamic_batching")


def test_a_static_engine_lists_every_axis_and_disables_batching() -> None:
    config = parse(render_config_pbtxt("m", static_engine(), TritonConfig()))
    assert config.max_batch_size == 0
    assert list(config.input[0].dims) == [2, 16]
    assert list(config.output[0].dims) == [2, 4]


def test_dynamic_batching_and_instances_follow_the_settings() -> None:
    settings = TritonConfig.model_validate(
        {
            "instance_count": 2,
            "instance_gpus": [0, 1],
            "dynamic_batching": {"preferred_batch_sizes": [4, 8], "max_queue_delay_us": 500},
        }
    )
    config = parse(render_config_pbtxt("m", dynamic_engine(), settings))
    assert list(config.dynamic_batching.preferred_batch_size) == [4, 8]
    assert config.dynamic_batching.max_queue_delay_microseconds == 500
    assert config.instance_group[0].count == 2
    assert list(config.instance_group[0].gpus) == [0, 1]


def test_dynamic_batching_without_preferred_sizes_is_valid() -> None:
    settings = TritonConfig.model_validate({"dynamic_batching": {}})
    config = parse(render_config_pbtxt("m", dynamic_engine(), settings))
    assert config.HasField("dynamic_batching")
    assert list(config.dynamic_batching.preferred_batch_size) == []


def test_every_dtype_maps_to_the_matching_triton_type() -> None:
    expected = {
        DType.FLOAT32: pb.TYPE_FP32,
        DType.FLOAT16: pb.TYPE_FP16,
        DType.INT64: pb.TYPE_INT64,
        DType.INT32: pb.TYPE_INT32,
        DType.INT8: pb.TYPE_INT8,
        DType.UINT8: pb.TYPE_UINT8,
        DType.BOOL: pb.TYPE_BOOL,
    }
    for dtype, triton_type in expected.items():
        info = EngineInfo(
            tensorrt_version="10.3.0",
            num_optimization_profiles=1,
            tensors=[
                binding("a", "input", [-1, 3], dtype, ([1, 3], [1, 3], [4, 3])),
                binding("b", "output", [-1, 3], dtype),
            ],
        )
        config = parse(render_config_pbtxt("m", info, TritonConfig()))
        assert config.input[0].data_type == triton_type
        assert config.output[0].data_type == triton_type


def test_multiple_inputs_and_outputs_keep_their_order_and_names() -> None:
    info = EngineInfo(
        tensorrt_version="10.3.0",
        num_optimization_profiles=1,
        tensors=[
            binding("input_ids", "input", [-1, -1], DType.INT64, ([1, 8], [4, 32], [8, 64])),
            binding("attention_mask", "input", [-1, -1], DType.INT64, ([1, 8], [4, 32], [16, 64])),
            binding("logits", "output", [-1, 2]),
            binding("hidden", "output", [-1, -1, 32]),
        ],
    )
    config = parse(render_config_pbtxt("bert", info, TritonConfig()))
    assert [i.name for i in config.input] == ["input_ids", "attention_mask"]
    assert [list(i.dims) for i in config.input] == [[-1], [-1]]
    assert [o.name for o in config.output] == ["logits", "hidden"]
    assert list(config.output[1].dims) == [-1, 32]
    assert config.max_batch_size == 8  # the smallest limit across inputs


def test_the_batch_limit_is_the_engines_unless_lowered() -> None:
    assert derive_max_batch_size(dynamic_engine(16), TritonConfig()) == 16
    assert derive_max_batch_size(dynamic_engine(16), TritonConfig(max_batch_size=4)) == 4
    assert derive_max_batch_size(dynamic_engine(16), TritonConfig(max_batch_size=0)) == 0
    assert derive_max_batch_size(static_engine(), TritonConfig()) == 0


def test_a_batch_limit_above_what_the_engine_supports_is_rejected() -> None:
    with pytest.raises(ConfigError, match="at most 8"):
        derive_max_batch_size(dynamic_engine(8), TritonConfig(max_batch_size=32))


def test_disabling_batching_on_a_dynamic_engine_keeps_the_full_shape() -> None:
    config = parse(render_config_pbtxt("m", dynamic_engine(), TritonConfig(max_batch_size=0)))
    assert config.max_batch_size == 0
    assert list(config.input[0].dims) == [-1, 16]


def test_dynamic_batching_needs_batching() -> None:
    settings = TritonConfig.model_validate({"dynamic_batching": {}})
    with pytest.raises(ConfigError, match="needs batching"):
        render_config_pbtxt("m", static_engine(), settings)


def test_preferred_batch_sizes_above_the_limit_are_rejected() -> None:
    settings = TritonConfig.model_validate({"dynamic_batching": {"preferred_batch_sizes": [4, 16]}})
    with pytest.raises(ConfigError, match=r"\[16\] exceed"):
        render_config_pbtxt("m", dynamic_engine(8), settings)


def test_batching_with_a_non_leading_dynamic_axis_is_rejected() -> None:
    info = EngineInfo(
        tensorrt_version="10.3.0",
        num_optimization_profiles=1,
        tensors=[
            binding("x", "input", [-1, 16], profile=([1, 16], [4, 16], [8, 16])),
            binding("y", "output", [4, 4]),  # the output is not batched
        ],
    )
    with pytest.raises(TritonError, match="dynamic leading"):
        render_config_pbtxt("m", info, TritonConfig())


def test_an_engine_without_outputs_is_rejected() -> None:
    info = EngineInfo(
        tensorrt_version="10.3.0",
        num_optimization_profiles=1,
        tensors=[binding("x", "input", [1, 2])],
    )
    with pytest.raises(TritonError, match="no inputs or no outputs"):
        render_config_pbtxt("m", info, TritonConfig())


# ---------------------------------------------------------------- repository


def plan_file(tmp_path: Path, content: bytes = b"plan-bytes") -> Path:
    plan = tmp_path / "m.fp16.plan"
    plan.write_bytes(content)
    return plan


def test_the_repository_has_triton_layout(tmp_path: Path) -> None:
    result = build_repository(
        tmp_path / "repo", "m", plan_file(tmp_path), dynamic_engine(), TritonConfig()
    )
    assert (tmp_path / "repo/m/1/model.plan").read_bytes() == b"plan-bytes"
    assert parse((tmp_path / "repo/m/config.pbtxt").read_text()).name == "m"
    assert result.max_batch_size == 8
    assert result.version == 1
    assert len(result.repository_sha256) == 64
    assert sorted(p.name for p in (tmp_path / "repo").iterdir()) == ["m"]  # no staging leftovers


def test_the_model_version_directory_follows_the_setting(tmp_path: Path) -> None:
    build_repository(
        tmp_path / "repo",
        "m",
        plan_file(tmp_path),
        dynamic_engine(),
        TritonConfig(model_version=3),
    )
    assert (tmp_path / "repo/m/3/model.plan").is_file()


def test_an_existing_repository_is_never_overwritten(tmp_path: Path) -> None:
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo/keep.txt").write_text("mine")
    with pytest.raises(ArtifactConflictError):
        build_repository(
            tmp_path / "repo", "m", plan_file(tmp_path), dynamic_engine(), TritonConfig()
        )
    assert (tmp_path / "repo/keep.txt").read_text() == "mine"


def test_a_failure_leaves_no_partial_repository(tmp_path: Path) -> None:
    settings = TritonConfig.model_validate({"dynamic_batching": {}})
    with pytest.raises(ConfigError):
        build_repository(tmp_path / "repo", "m", plan_file(tmp_path), static_engine(), settings)
    assert list(tmp_path.glob("repo*")) == []
    assert list(tmp_path.glob(".repo*")) == []


def test_a_missing_plan_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(TritonError, match="not found"):
        build_repository(
            tmp_path / "repo", "m", tmp_path / "absent.plan", dynamic_engine(), TritonConfig()
        )


def test_identical_inputs_give_the_same_repository_hash(tmp_path: Path) -> None:
    plan = plan_file(tmp_path)
    first = build_repository(tmp_path / "a", "m", plan, dynamic_engine(), TritonConfig())
    second = build_repository(tmp_path / "b", "m", plan, dynamic_engine(), TritonConfig())
    assert first.repository_sha256 == second.repository_sha256
