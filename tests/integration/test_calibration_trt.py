"""INT8 calibration against tests/fakes/fake_tensorrt.py: the data path, the calibrator protocol,
cache adoption/rejection, and the stages. Real TensorRT calibration is not exercised here."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from tests.fakes.fake_tensorrt import (
    DataType,
    FakeCalls,
    FakeOptions,
    HostBuffers,
    TensorSpec,
    make_fake_trt,
)
from tests.helpers import (
    MLP_INPUT,
    MLP_PROFILE,
    TOKEN_INPUTS,
    TOKEN_PROFILE,
    build_config,
    export_to,
    fake_environment,
)
from trtship.artifacts import ArtifactStore, ArtifactType, RunDirectory, StageStatus
from trtship.calibration import (
    TorchDeviceBuffers,
    calibrate,
    make_cache_calibrator,
    make_calibrator,
    read_cache_dir,
)
from trtship.cli.main import app
from trtship.config import TrtshipConfig
from trtship.errors import (
    CalibrationError,
    EngineBuildError,
    EnvironmentUnavailableError,
)
from trtship.models import ModelSignature, load_model
from trtship.pipeline import Pipeline
from trtship.pipeline.stages import default_stages
from trtship.tensorrt import build as trt_build
from trtship.utils import env

MakeRun = Callable[..., RunDirectory]
runner = CliRunner()


# --------------------------------------------------------------------------- calibrators


def test_calibrator_feeds_batches_and_collects_the_cache() -> None:
    trt, _ = make_fake_trt()
    device = HostBuffers()
    batches = iter([np.full((2, 4), i, dtype=np.float32) for i in (1, 2, 3)])
    other = {"mask": np.ones((2, 4), dtype=np.int64)}
    calibrator = make_calibrator(
        trt,
        "entropy2",
        batches=batches,
        batch_size=2,
        input_name="x",
        other_inputs=other,
        device=device,
    )
    assert isinstance(calibrator, trt.IInt8EntropyCalibrator2)
    assert calibrator.get_batch_size() == 2
    assert calibrator.read_calibration_cache() is None  # always calibrates from data

    pointers = calibrator.get_batch(["x", "mask"])
    assert pointers is not None
    assert len(pointers) == 2
    assert device.uploads[0].tolist() == np.full((2, 4), 1.0).tolist()  # the batch
    assert device.uploads[1].dtype == np.int64  # the constant for the other input
    assert calibrator.get_batch(["x", "mask"]) is not None
    assert calibrator.get_batch(["x", "mask"]) is not None
    assert calibrator.get_batch(["x", "mask"]) is None  # exhausted
    assert calibrator.batches_served == 3

    assert calibrator.cache_bytes is None
    calibrator.write_calibration_cache(bytearray(b"scales"))
    assert calibrator.cache_bytes == b"scales"


def test_calibrator_rejects_inputs_it_has_no_data_for() -> None:
    trt, _ = make_fake_trt()
    calibrator = make_calibrator(
        trt,
        "minmax",
        batches=iter([np.zeros((1, 2), dtype=np.float32)]),
        batch_size=1,
        input_name="x",
        other_inputs={},
        device=HostBuffers(),
    )
    assert isinstance(calibrator, trt.IInt8MinMaxCalibrator)
    with pytest.raises(CalibrationError, match=r"which has none"):
        calibrator.get_batch(["x", "surprise"])


def test_unknown_methods_are_rejected() -> None:
    trt, _ = make_fake_trt()
    with pytest.raises(CalibrationError, match="unknown calibration method"):
        make_cache_calibrator(trt, "histogram", b"c")


def test_cache_calibrator_only_serves_the_cache() -> None:
    trt, _ = make_fake_trt()
    calibrator = make_cache_calibrator(trt, "entropy2", b"known scales")
    assert calibrator.read_calibration_cache() == b"known scales"
    assert calibrator.get_batch(["x"]) is None
    calibrator.write_calibration_cache(b"ignored")  # nothing to store, and no error


def test_torch_device_buffers_refuse_a_cpu_only_torch() -> None:
    with pytest.raises(EnvironmentUnavailableError, match="CUDA build of PyTorch"):
        TorchDeviceBuffers()


# --------------------------------------------------------------------------- calibrate()


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, ModelSignature, str]:
    config = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)
    directory = tmp_path_factory.mktemp("cal")
    _, signature = export_to(config, directory / "m.onnx")
    weights = load_model(config.model).weights_sha256
    return directory / "m.onnx", signature, weights


def int8_config(
    tmp_path: Path, *, samples: int = 16, batch_size: int = 4, **calibration: Any
) -> TrtshipConfig:
    data = tmp_path / "calib.npy"
    if not data.exists():
        np.save(data, np.random.default_rng(0).standard_normal((samples, 16)).astype(np.float32))
    base = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE).model_dump(mode="json")
    base["tensorrt"]["precisions"] = ["fp32", "int8"]
    base["calibration"] = {
        "dataset": "numpy",
        "path": str(data),
        "num_samples": 12,
        "batch_size": batch_size,
        **calibration,
    }
    base["artifacts"] = {"root": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")}
    return TrtshipConfig.model_validate(base)


def run_calibrate(
    exported: tuple[Path, ModelSignature, str],
    config: TrtshipConfig,
    out: Path,
    options: FakeOptions | None = None,
) -> tuple[Any, FakeCalls]:
    onnx_path, signature, weights = exported
    trt, calls = make_fake_trt(options)
    metadata = calibrate(
        onnx_path,
        signature,
        config,
        out,
        model_weights_sha256=weights,
        trt=trt,
        device=HostBuffers(),
    )
    return metadata, calls


def test_calibrate_writes_a_cache_with_full_metadata(
    exported: tuple[Path, ModelSignature, str], tmp_path: Path
) -> None:
    config = int8_config(tmp_path)
    metadata, calls = run_calibrate(exported, config, tmp_path / "cache-dir")

    assert (metadata.requested_samples, metadata.sample_count) == (12, 12)
    assert (metadata.batch_size, metadata.num_batches) == (4, 3)
    assert metadata.method == "entropy2"
    assert metadata.input_name == "x"
    assert metadata.sample_shape == [16]
    assert metadata.tensorrt_version == "10.3.0.26"
    assert metadata.model_weights_sha256 == exported[2]
    assert metadata.representative is True
    assert metadata.dataset.kind == "numpy"
    assert metadata.dataset.items == 16

    cache, on_disk = read_cache_dir(tmp_path / "cache-dir")
    assert cache == b"FAKE-SCALES:3"
    assert on_disk == metadata
    (built,) = calls.configs
    assert [f.name for f in built.flags] == ["FP16", "INT8"]
    assert (
        built.calibration_profile is built.profiles[0]
    )  # dynamic-shape INT8 calibrates at a profile
    assert len(calls.calibration_batches) == 3
    assert not any(tmp_path.glob(".*build"))  # no scratch leftovers


def test_calibration_is_deterministic_for_a_seed(
    exported: tuple[Path, ModelSignature, str], tmp_path: Path
) -> None:
    config = int8_config(tmp_path, seed=5)
    first, _ = run_calibrate(exported, config, tmp_path / "a")
    second, _ = run_calibrate(exported, config, tmp_path / "b")
    assert first.cache_sha256 == second.cache_sha256
    assert first.dataset.fingerprint == second.dataset.fingerprint
    other, _ = run_calibrate(exported, int8_config(tmp_path, seed=6), tmp_path / "c")
    assert other.seed == 6


def test_synthetic_calibration_is_flagged_not_representative(
    exported: tuple[Path, ModelSignature, str], tmp_path: Path
) -> None:
    base = int8_config(tmp_path).model_dump(mode="json")
    base["calibration"] = {
        "dataset": "synthetic",
        "allow_synthetic": True,
        "num_samples": 8,
        "batch_size": 4,
    }
    metadata, _ = run_calibrate(exported, TrtshipConfig.model_validate(base), tmp_path / "s")
    assert metadata.representative is False
    assert metadata.dataset.kind == "synthetic"


def test_a_multi_input_model_calibrates_one_input_and_feeds_the_rest(tmp_path: Path) -> None:
    ids = tmp_path / "ids.npy"
    np.save(ids, np.random.default_rng(1).integers(0, 100, size=(8, 32)).astype(np.int64))
    base = build_config(
        "token_classifier", TOKEN_INPUTS, profile=TOKEN_PROFILE, output_names=["logits", "hidden"]
    ).model_dump(mode="json")
    base["tensorrt"]["precisions"] = ["int8"]
    base["calibration"] = {
        "dataset": "numpy",
        "path": str(ids),
        "input_name": "input_ids",
        "num_samples": 8,
        "batch_size": 4,
    }
    config = TrtshipConfig.model_validate(base)
    _, signature = export_to(config, tmp_path / "t.onnx")
    trt, calls = make_fake_trt(
        FakeOptions(
            network_inputs=["input_ids", "attention_mask"],
            engine_inputs=[
                TensorSpec("input_ids", (-1, -1), DataType.INT64),
                TensorSpec("attention_mask", (-1, -1), DataType.INT64),
            ],
            engine_outputs=[TensorSpec("logits", (-1, 3)), TensorSpec("hidden", (-1, -1, 8))],
        )
    )
    device = HostBuffers()
    metadata = calibrate(
        tmp_path / "t.onnx", signature, config, tmp_path / "out",
        model_weights_sha256="w" * 64, trt=trt, device=device,
    )  # fmt: skip
    assert metadata.input_name == "input_ids"
    assert metadata.sample_shape == [32]  # seq comes from the profile's opt shape
    assert all(len(pointers) == 2 for pointers in calls.calibration_batches)
    shapes = {u.shape for u in device.uploads}
    assert shapes == {(4, 32)}  # the other input is fabricated at the same batch shape


def test_calibration_configuration_errors(
    exported: tuple[Path, ModelSignature, str], tmp_path: Path
) -> None:
    with pytest.raises(CalibrationError, match="cannot fill one batch of 32"):
        run_calibrate(exported, int8_config(tmp_path, batch_size=32), tmp_path / "a")
    no_section = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)
    with pytest.raises(CalibrationError, match="not configured"):
        run_calibrate(exported, no_section, tmp_path / "b")


def test_fixed_batch_axes_must_match_the_calibration_batch_size(tmp_path: Path) -> None:
    np.save(tmp_path / "d.npy", np.zeros((8, 16), dtype=np.float32))
    base = build_config("tiny_mlp", [{"name": "x", "shape": [2, 16]}]).model_dump(mode="json")
    base["tensorrt"]["precisions"] = ["int8"]
    base["calibration"] = {"dataset": "numpy", "path": str(tmp_path / "d.npy"), "batch_size": 4}
    config = TrtshipConfig.model_validate(base)
    _, signature = export_to(config, tmp_path / "m.onnx")
    with pytest.raises(CalibrationError, match="fixed batch axis of 2"):
        calibrate(
            tmp_path / "m.onnx",
            signature,
            config,
            tmp_path / "o",
            model_weights_sha256="w",
            trt=make_fake_trt()[0],
            device=HostBuffers(),
        )


def test_tensorrt_producing_no_cache_is_an_error_and_leaves_nothing(
    exported: tuple[Path, ModelSignature, str], tmp_path: Path
) -> None:
    with pytest.raises(CalibrationError, match="without writing a calibration cache"):
        run_calibrate(
            exported,
            int8_config(tmp_path),
            tmp_path / "out",
            FakeOptions(calibration_writes_cache=False),
        )
    assert not (tmp_path / "out").exists()
    assert not any(tmp_path.glob(".*build"))


def test_an_existing_output_directory_is_never_overwritten(
    exported: tuple[Path, ModelSignature, str], tmp_path: Path
) -> None:
    (tmp_path / "out").mkdir()
    with pytest.raises(CalibrationError, match="refusing to overwrite"):
        run_calibrate(exported, int8_config(tmp_path), tmp_path / "out")


# --------------------------------------------------------------------------- adopting a cache


def test_a_matching_cache_is_adopted_without_calibrating(
    exported: tuple[Path, ModelSignature, str], tmp_path: Path
) -> None:
    config = int8_config(tmp_path)
    first, first_calls = run_calibrate(exported, config, tmp_path / "first")
    assert first_calls.builds == 1

    adopted_config = int8_config(tmp_path, cache_path=str(tmp_path / "first"))
    second, second_calls = run_calibrate(exported, adopted_config, tmp_path / "second")
    assert second_calls.builds == 0  # TensorRT was not asked to calibrate again
    assert second.cache_sha256 == first.cache_sha256
    assert read_cache_dir(tmp_path / "second")[0] == b"FAKE-SCALES:3"


@pytest.mark.parametrize(
    ("override", "mismatch"),
    [
        ({"batch_size": 2}, "batch_size"),
        ({"method": "minmax"}, "method"),
        ({"seed": 99}, "sample_count|seed|dataset_fingerprint"),
    ],
)
def test_a_cache_that_no_longer_matches_is_rejected_with_reasons(
    exported: tuple[Path, ModelSignature, str],
    tmp_path: Path,
    override: dict[str, Any],
    mismatch: str,
) -> None:
    run_calibrate(exported, int8_config(tmp_path), tmp_path / "first")
    config = int8_config(tmp_path, cache_path=str(tmp_path / "first"), **override)
    with pytest.raises(CalibrationError, match="does not match this run") as info:
        run_calibrate(exported, config, tmp_path / "second")
    assert any(
        re_key in p for p in info.value.details["mismatches"] for re_key in mismatch.split("|")
    )
    assert not (tmp_path / "second").exists()


def test_a_cache_for_different_data_or_model_is_rejected(
    exported: tuple[Path, ModelSignature, str], tmp_path: Path
) -> None:
    run_calibrate(exported, int8_config(tmp_path), tmp_path / "first")
    np.save(tmp_path / "calib.npy", np.zeros((16, 16), dtype=np.float32))  # different data
    config = int8_config(tmp_path, cache_path=str(tmp_path / "first"))
    with pytest.raises(CalibrationError) as info:
        run_calibrate(exported, config, tmp_path / "second")
    assert any(p.startswith("dataset_fingerprint") for p in info.value.details["mismatches"])

    onnx_path, signature, _ = exported
    trt, _ = make_fake_trt()
    with pytest.raises(CalibrationError) as model_info:
        calibrate(onnx_path, signature, config, tmp_path / "third",
                  model_weights_sha256="different" * 8, trt=trt, device=HostBuffers())  # fmt: skip
    assert any(p.startswith("model_weights_sha256") for p in model_info.value.details["mismatches"])


def test_a_tampered_cache_is_never_adopted(
    exported: tuple[Path, ModelSignature, str], tmp_path: Path
) -> None:
    run_calibrate(exported, int8_config(tmp_path), tmp_path / "first")
    (tmp_path / "first" / "calibration.cache").write_bytes(b"bit rot")
    config = int8_config(tmp_path, cache_path=str(tmp_path / "first"))
    with pytest.raises(CalibrationError, match="does not match its metadata"):
        run_calibrate(exported, config, tmp_path / "second")


# --------------------------------------------------------------------------- the pipeline


@pytest.fixture
def fake_trt(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeOptions | None], FakeCalls]:
    real = trt_build.build_engine

    def install(options: FakeOptions | None = None) -> FakeCalls:
        trt, calls = make_fake_trt(options)
        inject: Callable[..., Any] = lambda *a, **k: real(*a, **{**k, "trt": trt})  # noqa: E731
        monkeypatch.setattr("trtship.pipeline.stages.build_stage.build_engine", inject)
        monkeypatch.setattr("trtship.calibration.run.build_engine", inject)
        monkeypatch.setattr("trtship.calibration.run.load_tensorrt", lambda **k: trt)
        monkeypatch.setattr("trtship.calibration.run.TorchDeviceBuffers", HostBuffers)
        monkeypatch.setattr("trtship.pipeline.stages.build_stage.load_tensorrt", lambda **k: trt)
        return calls

    return install


def pipeline(config: TrtshipConfig, make_run: MakeRun, run_id: str = "r1") -> Pipeline:
    return Pipeline(config, make_run(run_id), default_stages(), fake_environment(gpu_ok=True))


def test_int8_flows_through_calibrate_then_build(
    tmp_path: Path, make_run: MakeRun, fake_trt: Callable[..., FakeCalls]
) -> None:
    calls = fake_trt()
    run = pipeline(int8_config(tmp_path), make_run)
    result = run.execute()
    statuses = {o.name: o.status for o in result.stages}
    assert statuses["calibrate"] is StageStatus.SUCCEEDED
    assert statuses["build"] is StageStatus.SUCCEEDED
    assert [o.name for o in result.stages].index("calibrate") < [
        o.name for o in result.stages
    ].index("build")

    store = ArtifactStore(run.run)
    (cache_record,) = store.records(ArtifactType.CALIBRATION_CACHE)
    assert cache_record.path == "artifacts/m.calibration"
    assert cache_record.metadata["dataset"]["kind"] == "numpy"
    assert cache_record.metadata["representative"] is True
    optimized = store.require(ArtifactType.ONNX_OPTIMIZED, needed_by="t")
    assert (
        cache_record.parents == [store.require(ArtifactType.ONNX, needed_by="t").id]
        or optimized.id in cache_record.parents
    )

    engines = {e.metadata["precision"]: e for e in store.records(ArtifactType.ENGINE)}
    assert set(engines) == {"fp32", "int8"}
    int8 = engines["int8"]
    assert int8.metadata["calibration_cache"] == cache_record.id
    assert int8.metadata["build"]["calibrator"] == "CacheCalibrator"
    assert int8.metadata["build"]["builder_flags"] == ["FP16", "INT8"]
    assert engines["fp32"].metadata["calibration_cache"] is None
    assert calls.used_cache  # the INT8 build used the cache
    assert len(calls.calibration_batches) == 3  # data was fed once, during calibrate only
    assert store.verify_all() == []


def test_calibration_is_skipped_without_int8(
    tmp_path: Path, make_run: MakeRun, fake_trt: Callable[..., FakeCalls]
) -> None:
    fake_trt()
    base = int8_config(tmp_path).model_dump(mode="json")
    base["tensorrt"]["precisions"] = ["fp32"]
    result = pipeline(TrtshipConfig.model_validate(base), make_run).execute()
    calibrate_outcome = next(o for o in result.stages if o.name == "calibrate")
    assert (calibrate_outcome.status, calibrate_outcome.reason) == (
        StageStatus.SKIPPED,
        "int8 is not in tensorrt.precisions",
    )


def test_calibration_results_are_cached_and_keyed_on_the_data(
    tmp_path: Path, make_run: MakeRun, fake_trt: Callable[..., FakeCalls]
) -> None:
    calls = fake_trt()
    config = int8_config(tmp_path)
    pipeline(config, make_run, "one").execute()
    batches = len(calls.calibration_batches)

    same = pipeline(config, make_run, "two").execute()
    assert {o.name: o.status for o in same.stages}["calibrate"] is StageStatus.CACHED
    assert len(calls.calibration_batches) == batches  # no recalibration

    np.save(tmp_path / "calib.npy", np.zeros((16, 16), dtype=np.float32))  # the data changed
    changed = pipeline(config, make_run, "three").execute()
    statuses = {o.name: o.status for o in changed.stages}
    assert statuses["calibrate"] is StageStatus.SUCCEEDED  # a new fingerprint is a new key
    assert statuses["build"] is StageStatus.SUCCEEDED  # scales changed, so engines are rebuilt
    assert statuses["optimize"] is StageStatus.CACHED


def test_synthetic_calibration_warns_in_the_pipeline(
    tmp_path: Path, make_run: MakeRun, fake_trt: Callable[..., FakeCalls]
) -> None:
    fake_trt()
    base = int8_config(tmp_path).model_dump(mode="json")
    base["calibration"] = {
        "dataset": "synthetic",
        "allow_synthetic": True,
        "num_samples": 8,
        "batch_size": 4,
    }
    result = pipeline(TrtshipConfig.model_validate(base), make_run).execute()
    warnings = next(o for o in result.stages if o.name == "calibrate").warnings
    assert any("not representative" in w for w in warnings)


def test_building_int8_without_a_cache_explains_what_to_do(
    tmp_path: Path, make_run: MakeRun, fake_trt: Callable[..., FakeCalls]
) -> None:
    fake_trt()
    run = pipeline(int8_config(tmp_path), make_run)
    run.execute(until="optimize")
    with pytest.raises(EngineBuildError, match="needs a calibration cache") as info:
        run.execute(only="build")
    assert "calibrate stage" in (info.value.hint or "")


# --------------------------------------------------------------------------- the command


def test_calibrate_command_needs_a_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "probe_nvidia_gpu", lambda: ([], env._missing(env.NVIDIA_GPU, "none")))
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(int8_config(tmp_path).model_dump(mode="json")))
    result = runner.invoke(
        app,
        ["calibrate", str(config), "m.onnx", "-o", str(tmp_path / "out")],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 3
    assert "INT8 calibration" in result.output
    assert not (tmp_path / "out").exists()


def test_calibrate_command_writes_the_cache(
    exported: tuple[Path, ModelSignature, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_trt: Callable[..., FakeCalls],
) -> None:
    fake_trt()
    monkeypatch.setattr(env, "require_tensorrt", lambda purpose: None)
    monkeypatch.setattr(env, "require", lambda name, purpose: None)
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(int8_config(tmp_path).model_dump(mode="json")))
    result = runner.invoke(
        app, ["calibrate", str(config), str(exported[0]), "-o", str(tmp_path / "out"), "--json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["sample_count"] == 12
    assert (tmp_path / "out" / "calibration.cache").is_file()
    assert (tmp_path / "out" / "metadata.json").is_file()

    again = runner.invoke(
        app, ["calibrate", str(config), str(exported[0]), "-o", str(tmp_path / "out")]
    )
    assert again.exit_code == 8  # never overwrites; CalibrationError
