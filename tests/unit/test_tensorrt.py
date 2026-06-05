"""trtship's TensorRT translation logic, tested against tests/fakes/fake_tensorrt.py.

The fake records the calls trtship makes; these tests prove trtship asks for the right things and
reports failures well. They cannot prove real TensorRT accepts them: see tests/gpu.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnx
import pytest
from onnx import TensorProto, helper

from tests.fakes.fake_tensorrt import (
    DataType,
    FakeBuilderConfig,
    FakeCalls,
    FakeNetwork,
    FakeOnnxParser,
    FakeOptions,
    FakeProfile,
    ParserError,
    TensorSpec,
    make_fake_trt,
)
from tests.helpers import MLP_INPUT, MLP_PROFILE, build_config, export_to
from trtship.config import OptimizationProfile, Precision, TrtshipConfig
from trtship.errors import ConfigError, EngineBuildError, EnvironmentUnavailableError
from trtship.models import ModelSignature
from trtship.specs import TensorSpec as SpecModel
from trtship.tensorrt import (
    BuildResult,
    EngineInfo,
    TensorBinding,
    TrtVersion,
    build_engine,
    build_profile_shapes,
    check_supported,
    convert_dtype,
    describe_engine,
    load_tensorrt,
    parse_version,
    parser_errors,
    unsupported_operators,
)
from trtship.tensorrt.engine_info import ProfileRange
from trtship.utils import env

MIB = 1024 * 1024


# --------------------------------------------------------------------------- versions


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("10.3.0.26", TrtVersion(10, 3, 0)),
        ("8.6.1", TrtVersion(8, 6, 1)),
        ("9.2", TrtVersion(9, 2, 0)),
        (" 10.0.1+cu12 ", TrtVersion(10, 0, 1)),
    ],
)
def test_parse_version(text: str, expected: TrtVersion) -> None:
    assert parse_version(text) == expected


def test_parse_version_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="cannot parse"):
        parse_version("not-a-version")


def test_supported_versions() -> None:
    check_supported(TrtVersion(8, 6, 0))
    check_supported(TrtVersion(10, 3, 0))
    with pytest.raises(EnvironmentUnavailableError, match=r"8\.5\.0 is not supported"):
        check_supported(TrtVersion(8, 5, 0))
    assert TrtVersion(10, 0) > TrtVersion(8, 6, 9)
    assert str(TrtVersion(10, 3, 0)) == "10.3.0"


def test_load_tensorrt_refuses_to_proceed_without_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        env, "probe_nvidia_gpu", lambda: ([], env._missing(env.NVIDIA_GPU, "no driver"))
    )
    with pytest.raises(EnvironmentUnavailableError, match="building a TensorRT engine") as info:
        load_tensorrt(purpose="building a TensorRT engine")
    assert info.value.exit_code == 3


# --------------------------------------------------------------------------- profiles


def sig(*specs: dict[str, Any]) -> ModelSignature:
    return ModelSignature(
        inputs=[SpecModel.model_validate(s) for s in specs], outputs=[], probe_sizes=[{}]
    )


def profile(**ranges: tuple[list[int], list[int], list[int]]) -> OptimizationProfile:
    return OptimizationProfile.model_validate(
        {"inputs": {k: {"min": v[0], "opt": v[1], "max": v[2]} for k, v in ranges.items()}}
    )


def test_profiles_for_dynamic_inputs() -> None:
    signature = sig({"name": "x", "shape": ["batch", 16]})
    (shapes,) = build_profile_shapes(signature, [profile(x=([1, 16], [4, 16], [8, 16]))])
    assert shapes == {"x": ((1, 16), (4, 16), (8, 16))}


def test_static_inputs_get_their_fixed_shape_and_need_no_profile() -> None:
    signature = sig({"name": "x", "shape": [3, 16]})
    assert build_profile_shapes(signature, []) == []
    (shapes,) = build_profile_shapes(signature, [profile(other=([1], [1], [1]))])
    assert shapes["x"] == ((3, 16), (3, 16), (3, 16))


def test_mixed_static_and_dynamic_inputs() -> None:
    signature = sig({"name": "ids", "shape": ["batch", "seq"]}, {"name": "scale", "shape": [1]})
    (shapes,) = build_profile_shapes(signature, [profile(ids=([1, 8], [4, 32], [8, 64]))])
    assert shapes["scale"] == ((1,), (1,), (1,))


def test_dynamic_inputs_without_profiles_are_a_config_error() -> None:
    with pytest.raises(ConfigError, match=r"dynamic dimensions but tensorrt\.profiles is empty"):
        build_profile_shapes(sig({"name": "x", "shape": ["batch", 16]}), [])
    with pytest.raises(ConfigError, match="no entry for dynamic input 'x'"):
        build_profile_shapes(
            sig({"name": "x", "shape": ["batch", 16]}), [profile(y=([1], [1], [1]))]
        )


def test_multiple_profiles_are_kept_in_order() -> None:
    signature = sig({"name": "x", "shape": ["batch", 16]})
    profiles = [
        profile(x=([1, 16], [1, 16], [1, 16])),
        profile(x=([2, 16], [8, 16], [32, 16])),
    ]
    built = build_profile_shapes(signature, profiles)
    assert [p["x"][2] for p in built] == [(1, 16), (32, 16)]


# --------------------------------------------------------------------------- engine description


def deserialize(options: FakeOptions, *profile_shapes: dict[str, Any]) -> tuple[Any, Any]:
    """A fake engine as the builder would hand it back, with the given profiles set."""
    trt, calls = make_fake_trt(options)
    cfg = FakeBuilderConfig(calls, optimization_level=True)
    for shapes in profile_shapes:
        fake_profile = FakeProfile(valid=True)
        for name, (low, opt, high) in shapes.items():
            fake_profile.set_shape(name, low, opt, high)
        cfg.add_optimization_profile(fake_profile)
    calls.configs.append(cfg)
    engine = trt.Runtime(None).deserialize_cuda_engine(b"plan")
    return trt, engine


def test_describe_engine_with_the_tensor_api() -> None:
    trt, engine = deserialize(FakeOptions(), {"x": ((1, 16), (4, 16), (8, 16))})
    info = describe_engine(trt, engine)
    assert info.tensorrt_version == "10.3.0.26"
    assert info.num_optimization_profiles == 1
    (x,) = info.inputs
    (y,) = info.outputs
    assert (x.name, x.dtype.value, x.shape) == ("x", "float32", [-1, 16])
    assert x.profiles == [ProfileRange(min=[1, 16], opt=[4, 16], max=[8, 16])]
    assert x.is_dynamic
    assert (y.name, y.shape, y.profiles) == ("output", [-1, 4], [])
    assert info.max_batch_size() == 8


def test_describe_engine_with_the_bindings_api() -> None:
    trt, engine = deserialize(FakeOptions(tensor_api=False), {"x": ((1, 16), (2, 16), (16, 16))})
    info = describe_engine(trt, engine)
    assert [t.name for t in info.tensors] == ["x", "output"]
    assert info.inputs[0].profiles[0].max == [16, 16]
    assert info.max_batch_size() == 16


def test_static_engines_have_no_profile_ranges_and_no_batching() -> None:
    options = FakeOptions(
        engine_inputs=[TensorSpec("x", (3, 16))], engine_outputs=[TensorSpec("y", (3, 4))]
    )
    trt, engine = deserialize(options)
    info = describe_engine(trt, engine)
    assert info.inputs[0].profiles == []
    assert not info.inputs[0].is_dynamic
    assert info.max_batch_size() == 0


def test_max_batch_size_rules() -> None:
    def binding(name: str, shape: list[int], maxes: list[int]) -> TensorBinding:
        return TensorBinding(
            name=name,
            mode="input",
            dtype=convert_dtype(make_fake_trt()[0], DataType.FLOAT),
            shape=shape,
            profiles=[ProfileRange(min=[1] * len(shape), opt=maxes, max=maxes)],
        )

    two = EngineInfo(
        tensorrt_version="10.0",
        num_optimization_profiles=2,
        tensors=[binding("a", [-1, 8], [16, 8]), binding("b", [-1, 4], [8, 4])],
    )
    assert two.max_batch_size() == 8  # the smallest limit across inputs
    mixed = EngineInfo(
        tensorrt_version="10.0",
        num_optimization_profiles=1,
        tensors=[binding("a", [-1, 8], [16, 8]), binding("b", [4, 4], [4, 4])],
    )
    assert mixed.max_batch_size() == 0  # one input has a fixed leading axis
    assert (
        EngineInfo(tensorrt_version="1", num_optimization_profiles=1, tensors=[]).max_batch_size()
        == 0
    )


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        ("FLOAT", "float32"),
        ("HALF", "float16"),
        ("INT8", "int8"),
        ("INT32", "int32"),
        ("INT64", "int64"),
        ("BOOL", "bool"),
        ("UINT8", "uint8"),
    ],
)
def test_dtype_mapping(member: str, expected: str) -> None:
    trt, _ = make_fake_trt()
    assert convert_dtype(trt, getattr(DataType, member)).value == expected


def test_unsupported_engine_dtypes_are_reported() -> None:
    trt, _ = make_fake_trt()
    with pytest.raises(EngineBuildError, match="dtype trtship does not support"):
        convert_dtype(trt, DataType.BF16)


# --------------------------------------------------------------------------- parser diagnostics


def test_unsupported_operator_extraction() -> None:
    errors = [
        {"description": "In node 5 (parseGraph): No importer registered for op: FFT."},
        {"description": "getPluginCreator could not find plugin: Frobnicate version: 1"},
        {"description": "No importer registered for op: FFT. Attempting to import as plugin."},
        {"description": "Unsupported ONNX data type: DOUBLE"},
        {"description": "some other failure"},
    ]
    assert unsupported_operators(errors) == ["FFT", "Frobnicate"]
    assert unsupported_operators([]) == []


def test_parser_errors_are_collected() -> None:
    calls = FakeCalls(parse_errors=[ParserError("bad node", 3), ParserError("worse", 4)])
    parser = FakeOnnxParser(FakeNetwork(["x"]), None, calls)
    parser.parse_from_file("m.onnx")
    assert parser_errors(parser) == [
        {"index": 0, "code": "ErrorCode.UNSUPPORTED_NODE", "node": 3, "description": "bad node"},
        {"index": 1, "code": "ErrorCode.UNSUPPORTED_NODE", "node": 4, "description": "worse"},
    ]


# --------------------------------------------------------------------------- building


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, ModelSignature]:
    config = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)
    directory = tmp_path_factory.mktemp("trt")
    _, signature = export_to(config, directory / "m.onnx")
    return directory / "m.onnx", signature


def trt_config(**tensorrt: Any) -> TrtshipConfig:
    return build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE).model_copy(
        update={
            "tensorrt": build_config(
                "tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE
            ).tensorrt.model_copy(update=tensorrt)
        }
    )


def build(
    exported: tuple[Path, ModelSignature],
    tmp_path: Path,
    precision: Precision = Precision.FP32,
    *,
    options: FakeOptions | None = None,
    config: TrtshipConfig | None = None,
    calibrator: Any = None,
) -> tuple[BuildResult, FakeCalls]:
    onnx_path, signature = exported
    trt, calls = make_fake_trt(options)
    result = build_engine(
        onnx_path,
        signature,
        config or trt_config(),
        precision,
        tmp_path / f"m.{precision.value}.plan",
        calibrator=calibrator,
        trt=trt,
    )
    return result, calls


def test_fp32_build_translates_the_configuration(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    config = trt_config(workspace_mb=512, optimization_level=4)
    result, calls = build(exported, tmp_path, config=config)

    (cfg,) = calls.configs
    assert cfg.workspace_bytes == 512 * MIB
    assert cfg.flags == []
    assert cfg.builder_optimization_level == 4
    (fake_profile,) = cfg.profiles
    assert fake_profile.shapes == {"x": ((1, 16), (4, 16), (8, 16))}
    assert calls.parsed == [str(exported[0])]
    assert calls.network_flags == [0]  # TensorRT 10: networks are always explicit-batch

    assert result.precision is Precision.FP32
    assert result.builder_flags == []
    assert result.workspace_mb == 512
    assert result.optimization_level == 4
    assert result.profiles == [{"x": {"min": [1, 16], "opt": [4, 16], "max": [8, 16]}}]
    assert result.calibrator is None
    assert result.tensorrt_version == "10.3.0.26"
    assert result.engine.max_batch_size() == 8
    plan = Path(result.path)
    assert plan.read_bytes().startswith(b"FAKEPLAN")
    assert result.size_bytes == plan.stat().st_size
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.fp32.plan"]  # no temp leftovers


def test_old_tensorrt_gets_an_explicit_batch_network(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    _, calls = build(exported, tmp_path, options=FakeOptions(version="8.6.1"))
    assert calls.network_flags == [1 << 0]


def test_builders_without_an_optimization_level_setting(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    result, _ = build(exported, tmp_path, options=FakeOptions(has_optimization_level=False))
    assert result.optimization_level is None


def test_fp16_sets_the_flag_and_warns_on_gpus_without_fast_fp16(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    result, calls = build(exported, tmp_path, Precision.FP16)
    assert [f.name for f in calls.configs[0].flags] == ["FP16"]
    assert result.builder_flags == ["FP16"]
    assert result.warnings == []

    slow, _ = build(
        exported, tmp_path / "slow", Precision.FP16, options=FakeOptions(fast_fp16=False)
    )
    assert any("no fast FP16" in w for w in slow.warnings)


def test_int8_with_a_calibrator_enables_fp16_fallback_by_default(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    class Calibrator:
        pass

    calibrator = Calibrator()
    result, calls = build(exported, tmp_path, Precision.INT8, calibrator=calibrator)
    assert result.builder_flags == ["FP16", "INT8"]
    assert calls.configs[0].int8_calibrator is calibrator
    assert result.calibrator == "Calibrator"

    only_int8, _ = build(
        exported,
        tmp_path / "strict",
        Precision.INT8,
        calibrator=calibrator,
        config=trt_config(int8_fp16_fallback=False),
    )
    assert only_int8.builder_flags == ["INT8"]

    slow, _ = build(
        exported,
        tmp_path / "slow",
        Precision.INT8,
        calibrator=calibrator,
        options=FakeOptions(fast_int8=False),
    )
    assert any("no fast INT8" in w for w in slow.warnings)


def test_int8_without_calibration_is_refused_before_touching_tensorrt(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    with pytest.raises(EngineBuildError, match="INT8 needs a calibrator") as info:
        build(exported, tmp_path, Precision.INT8)
    assert "calibrate stage" in (info.value.hint or "")
    assert list(tmp_path.iterdir()) == []


def test_int8_is_allowed_without_a_calibrator_for_explicitly_quantized_models(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    qdq = helper.make_graph(
        [
            helper.make_node("QuantizeLinear", ["x", "s"], ["q"]),
            helper.make_node("DequantizeLinear", ["q", "s"], ["y"]),
        ],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 16])],
        [helper.make_tensor("s", TensorProto.FLOAT, [], [0.1])],
    )
    path = tmp_path / "qdq.onnx"
    onnx.save(helper.make_model(qdq, opset_imports=[helper.make_opsetid("", 17)]), str(path))
    trt, _ = make_fake_trt()
    result = build_engine(
        path, exported[1], trt_config(), Precision.INT8, tmp_path / "q.plan", trt=trt
    )
    assert result.calibrator is None
    assert "INT8" in result.builder_flags


def test_timing_cache_is_loaded_and_saved(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    cache = tmp_path / "cache" / "timing.bin"
    config = trt_config(timing_cache_path=cache)
    first, _ = build(exported, tmp_path / "a", config=config)
    assert first.timing_cache.loaded_bytes == 0
    assert first.timing_cache.saved_bytes > 0
    assert cache.read_bytes().endswith(b"|tactics")

    saved_by_first = cache.read_bytes()
    second, calls = build(exported, tmp_path / "b", config=config)
    assert second.timing_cache.loaded_bytes == first.timing_cache.saved_bytes
    assert calls.configs[0].timing_cache is not None
    assert calls.configs[0].timing_cache.initial == saved_by_first  # seeded from the saved cache
    assert build(exported, tmp_path / "c")[0].timing_cache.path is None


def test_parse_failures_name_the_unsupported_operators(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    options = FakeOptions(
        parse_errors=[ParserError("UNSUPPORTED_NODE: No importer registered for op: FFT.", 7)]
    )
    with pytest.raises(EngineBuildError, match=r"could not parse m\.onnx") as info:
        build(exported, tmp_path, options=options)
    assert info.value.details["unsupported_operators"] == ["FFT"]
    assert info.value.details["parser_errors"][0]["node"] == 7
    assert "FFT" in (info.value.hint or "")
    assert info.value.exit_code == 7
    assert list(tmp_path.iterdir()) == []


def test_parse_failures_without_a_known_operator_still_explain_themselves(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    with pytest.raises(EngineBuildError) as info:
        build(exported, tmp_path, options=FakeOptions(parse_errors=[ParserError("weird")]))
    assert info.value.details["unsupported_operators"] == []
    assert "validate the ONNX model first" in (info.value.hint or "")


def test_a_network_with_the_wrong_inputs_is_rejected(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    with pytest.raises(EngineBuildError, match=r"inputs \['other'\] differ"):
        build(exported, tmp_path, options=FakeOptions(network_inputs=["other"]))


def test_a_failed_build_reports_the_captured_builder_log(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    with pytest.raises(EngineBuildError, match="failed to build a fp32 engine") as info:
        build(exported, tmp_path, options=FakeOptions(build_fails=True))
    log_lines = info.value.details["builder_log"]
    assert any("conv1" in line and "ERROR" in line for line in log_lines)
    assert any("WARNING" in line for line in log_lines)
    assert not any("chatter" in line for line in log_lines)  # INFO is not captured
    assert list(tmp_path.iterdir()) == []


def test_invalid_profiles_are_reported(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    with pytest.raises(EngineBuildError, match="optimization profile 0 is not valid") as info:
        build(exported, tmp_path, options=FakeOptions(invalid_profile=True))
    assert info.value.details["shapes"]["x"]["max"] == [8, 16]


def test_engines_that_cannot_be_deserialized_are_an_error(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    with pytest.raises(EngineBuildError, match="could not be deserialized"):
        build(exported, tmp_path, options=FakeOptions(deserialize_fails=True))
    assert list(tmp_path.iterdir()) == []


def test_existing_engines_are_never_overwritten(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    target = tmp_path / "m.fp32.plan"
    target.write_bytes(b"precious")
    with pytest.raises(EngineBuildError, match="refusing to overwrite"):
        build(exported, tmp_path)
    assert target.read_bytes() == b"precious"


def test_unsupported_tensorrt_versions_are_rejected(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    with pytest.raises(EnvironmentUnavailableError, match=r"8\.5\.0 is not supported"):
        build(exported, tmp_path, options=FakeOptions(version="8.5.0"))


def test_selecting_another_gpu_needs_a_cuda_enabled_torch(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    with pytest.raises(EnvironmentUnavailableError, match="cannot select GPU 1"):
        build(exported, tmp_path, config=trt_config(device_index=1))
    result, _ = build(exported, tmp_path / "ok", config=trt_config(device_index=0))
    assert result.size_bytes > 0


def test_dynamic_models_need_profiles_before_tensorrt_is_touched(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    config = trt_config(profiles=[])
    with pytest.raises(ConfigError, match="dynamic dimensions"):
        build(exported, tmp_path, config=config)
    assert list(tmp_path.iterdir()) == []


def test_different_settings_produce_different_plans(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    fp32, _ = build(exported, tmp_path / "a")
    fp16, _ = build(exported, tmp_path / "b", Precision.FP16)
    assert fp32.sha256 != fp16.sha256
    flags = json.loads(Path(fp16.path).read_bytes().removeprefix(b"FAKEPLAN"))["flags"]
    assert flags == ["FP16"]


def test_build_result_round_trips_through_json(
    exported: tuple[Path, ModelSignature], tmp_path: Path
) -> None:
    result, _ = build(exported, tmp_path)
    assert BuildResult.model_validate_json(result.model_dump_json()) == result
