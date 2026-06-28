"""Engine validation logic with stand-in executors, and the validate_engine stage/command.

The executors reproduce PyTorch, corrupt it, or fail on purpose; this tests trtship's tolerance
gating, shape-point coverage, and reporting. It does not test real TensorRT accuracy.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from tests.fakes.executors import (
    TorchBackedExecutor,
    Transform,
    noisy,
    reversed_logits,
    wrong_batch,
)
from tests.fakes.fake_tensorrt import FakeCalls, FakeOptions, make_fake_trt
from tests.helpers import MLP_INPUT, MLP_PROFILE, build_config, export_to, fake_environment
from trtship.artifacts import ArtifactStore, ArtifactType, RunDirectory, StageStatus
from trtship.cli.main import app
from trtship.config import Precision, TrtshipConfig
from trtship.errors import EngineRuntimeError, ValidationFailedError
from trtship.models import LoadedModel, ModelSignature, infer_signature, load_model
from trtship.pipeline import Pipeline
from trtship.pipeline.stages import default_stages
from trtship.pipeline.stages.validate_engine_stage import latest_engine_per_precision
from trtship.tensorrt import build as trt_build
from trtship.utils import env
from trtship.validation.engine import (
    EngineUnderTest,
    EngineValidationReport,
    validate_engines,
)

MakeRun = Callable[..., RunDirectory]
runner = CliRunner()


@pytest.fixture(scope="module")
def exported(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, TrtshipConfig, LoadedModel, ModelSignature]:
    config = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)
    directory = tmp_path_factory.mktemp("engval")
    export_to(config, directory / "m.onnx")
    model = load_model(config.model)
    signature = infer_signature(model, config.model, config.tensorrt.profiles)
    return directory / "m.onnx", config, model, signature


def run_validation(
    exported: tuple[Path, TrtshipConfig, LoadedModel, ModelSignature],
    tmp_path: Path,
    transforms: dict[Precision, Transform | None],
) -> tuple[EngineValidationReport, list[TorchBackedExecutor]]:
    onnx_path, config, model, signature = exported
    engines = []
    for precision in transforms:
        plan = tmp_path / f"{precision.value}.plan"
        plan.write_bytes(f"plan-{precision.value}".encode())
        engines.append(EngineUnderTest(precision=precision, path=str(plan)))
    made: list[TorchBackedExecutor] = []

    def factory(path: Path) -> TorchBackedExecutor:
        precision = Precision(path.stem)
        executor = TorchBackedExecutor(model, transforms[precision])
        made.append(executor)
        return executor

    report = validate_engines(engines, onnx_path, model, signature, config, factory)
    return report, made


# --------------------------------------------------------------------------- validation logic


def test_an_exact_engine_passes_at_every_shape_point(
    exported: tuple[Path, TrtshipConfig, LoadedModel, ModelSignature], tmp_path: Path
) -> None:
    report, made = run_validation(exported, tmp_path, {Precision.FP32: None})
    assert report.passed, report.failures
    (result,) = report.results
    assert result.precision is Precision.FP32
    assert [p.label for p in result.points] == ["min", "opt", "max"]
    assert result.engine_sha256
    assert result.tolerance.atol == 1e-4
    point = result.points[2]
    assert point.samples_passed == point.samples == 8
    assert point.vs_pytorch[0].max_abs_error == 0.0
    assert point.vs_onnx[0].max_abs_error is not None
    assert point.vs_onnx[0].max_abs_error < 1e-5  # ONNX Runtime agrees with PyTorch too
    assert made[0].closed  # engines are always released


def test_each_precision_is_judged_by_its_own_tolerance(
    exported: tuple[Path, TrtshipConfig, LoadedModel, ModelSignature], tmp_path: Path
) -> None:
    report, _ = run_validation(
        exported,
        tmp_path,
        {
            Precision.FP32: noisy(1e-3),  # too noisy for float32 (atol 1e-4)
            Precision.FP16: noisy(1e-3),  # fine for float16 (atol 5e-2)
            Precision.INT8: noisy(1e-3),
        },
    )
    by_precision = {r.precision: r for r in report.results}
    assert not by_precision[Precision.FP32].passed
    assert by_precision[Precision.FP16].passed
    assert by_precision[Precision.INT8].passed
    assert by_precision[Precision.FP16].tolerance.atol == 5e-2
    assert by_precision[Precision.INT8].tolerance.cosine_min == 0.98
    assert not report.passed
    assert all(f.startswith("fp32 ") for f in report.failures)  # only fp32 is blamed
    with pytest.raises(ValidationFailedError, match="engine validation failed") as info:
        report.raise_for_failure()
    assert info.value.exit_code == 6
    assert info.value.details["failures"] == report.failures


def test_flipped_predictions_fail_agreement_even_when_shapes_match(
    exported: tuple[Path, TrtshipConfig, LoadedModel, ModelSignature], tmp_path: Path
) -> None:
    report, _ = run_validation(exported, tmp_path, {Precision.FP16: reversed_logits})
    assert not report.passed
    text = " ".join(report.failures)
    assert "cosine similarity" in text or "top-1 agreement" in text or "differ beyond" in text


def test_a_wrong_output_shape_is_reported_per_point(
    exported: tuple[Path, TrtshipConfig, LoadedModel, ModelSignature], tmp_path: Path
) -> None:
    report, _ = run_validation(exported, tmp_path, {Precision.FP32: wrong_batch})
    (result,) = report.results
    by_label = {p.label: p for p in result.points}
    assert by_label["opt"].passed  # batch 4 happens to match
    assert not by_label["min"].passed
    assert not by_label["max"].passed
    assert any("shape mismatch" in f for f in report.failures)


def test_an_engine_that_cannot_run_a_shape_fails_that_point_only(
    exported: tuple[Path, TrtshipConfig, LoadedModel, ModelSignature], tmp_path: Path
) -> None:
    def refuses_large(outputs: dict[str, np.ndarray], inputs: Any) -> dict[str, np.ndarray]:
        if inputs["x"].shape[0] > 4:
            raise EngineRuntimeError("input 'x' of shape [8, 16] is outside the profile")
        return outputs

    report, _ = run_validation(exported, tmp_path, {Precision.FP32: refuses_large})
    (result,) = report.results
    by_label = {p.label: p for p in result.points}
    assert by_label["min"].passed
    assert by_label["opt"].passed
    assert by_label["max"].error is not None
    assert "outside the profile" in by_label["max"].error
    assert not report.passed


def test_multiple_engines_are_validated_and_all_released(
    exported: tuple[Path, TrtshipConfig, LoadedModel, ModelSignature], tmp_path: Path
) -> None:
    report, made = run_validation(
        exported, tmp_path, {Precision.FP32: None, Precision.FP16: noisy(1e-3)}
    )
    assert report.passed
    assert [r.precision for r in report.results] == [Precision.FP32, Precision.FP16]
    assert len(made) == 2
    assert all(e.closed for e in made)


def test_executors_are_released_even_when_validation_raises(
    exported: tuple[Path, TrtshipConfig, LoadedModel, ModelSignature], tmp_path: Path
) -> None:
    onnx_path, config, model, signature = exported
    plan = tmp_path / "fp32.plan"
    plan.write_bytes(b"x")
    executor = TorchBackedExecutor(model)

    def broken_sha(path: Path) -> TorchBackedExecutor:
        return executor

    plan.unlink()  # sha256 of a missing engine raises after execution
    with pytest.raises(FileNotFoundError):
        validate_engines(
            [EngineUnderTest(precision=Precision.FP32, path=str(plan))],
            onnx_path,
            model,
            signature,
            config,
            broken_sha,
        )
    assert executor.closed


def test_the_report_round_trips_through_json(
    exported: tuple[Path, TrtshipConfig, LoadedModel, ModelSignature], tmp_path: Path
) -> None:
    report, _ = run_validation(exported, tmp_path, {Precision.FP32: None})
    assert EngineValidationReport.model_validate_json(report.model_dump_json()) == report
    assert report.schema_version == 1


# --------------------------------------------------------------------------- the stage


@pytest.fixture
def fake_trt(monkeypatch: pytest.MonkeyPatch) -> Callable[[Transform | None], FakeCalls]:
    real = trt_build.build_engine

    def install(transform: Transform | None = None) -> FakeCalls:
        trt, calls = make_fake_trt(FakeOptions())
        monkeypatch.setattr(
            "trtship.pipeline.stages.build_stage.build_engine",
            lambda *a, **k: real(*a, **{**k, "trt": trt}),
        )
        model_holder: dict[str, LoadedModel] = {}

        def open_engine(path: Path) -> TorchBackedExecutor:
            if "model" not in model_holder:
                model_holder["model"] = load_model(engine_config.model)
            return TorchBackedExecutor(model_holder["model"], transform)

        monkeypatch.setattr(
            "trtship.pipeline.stages.validate_engine_stage._open_engine", open_engine
        )
        return calls

    engine_config = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE)
    return install


def pipeline_config(tmp_path: Path) -> TrtshipConfig:
    data = build_config("tiny_mlp", [MLP_INPUT], profile=MLP_PROFILE).model_dump(mode="json")
    data["tensorrt"]["precisions"] = ["fp32", "fp16"]
    data["artifacts"] = {"root": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")}
    return TrtshipConfig.model_validate(data)


def test_the_stage_validates_every_built_engine(
    tmp_path: Path, make_run: MakeRun, fake_trt: Callable[..., FakeCalls]
) -> None:
    fake_trt(None)
    run = Pipeline(
        pipeline_config(tmp_path), make_run(), default_stages(), fake_environment(gpu_ok=True)
    )
    result = run.execute()
    assert [o.name for o in result.stages][-2:] == ["build", "validate_engine"]
    assert all(o.status in (StageStatus.SUCCEEDED, StageStatus.SKIPPED) for o in result.stages)

    store = ArtifactStore(run.run)
    reports = [
        r for r in store.records(ArtifactType.VALIDATION_REPORT) if r.metadata["kind"] == "engine"
    ]
    (record,) = reports
    assert record.metadata["passed"] is True
    assert set(record.metadata["precisions"]) == {"fp32", "fp16"}
    engine_ids = {e.id for e in store.records(ArtifactType.ENGINE)}
    assert engine_ids <= set(record.parents)  # provenance: every engine is an input
    payload = json.loads(store.absolute(record).read_text())
    assert payload["passed"] is True
    assert {r["precision"] for r in payload["results"]} == {"fp32", "fp16"}


def test_an_inaccurate_engine_fails_the_stage_and_publishes_the_evidence(
    tmp_path: Path, make_run: MakeRun, fake_trt: Callable[..., FakeCalls]
) -> None:
    fake_trt(noisy(0.5))  # far too noisy for any precision
    run = Pipeline(
        pipeline_config(tmp_path), make_run(), default_stages(), fake_environment(gpu_ok=True)
    )
    with pytest.raises(ValidationFailedError, match="engine validation failed"):
        run.execute()
    manifest = run.run.read_manifest()
    assert manifest.stages["build"].status is StageStatus.SUCCEEDED  # engines are kept
    assert manifest.stages["validate_engine"].status is StageStatus.FAILED
    assert manifest.stages["validate_engine"].error is not None
    assert manifest.stages["validate_engine"].error["exit_code"] == 6
    store = ArtifactStore(run.run)
    (evidence,) = [
        r for r in store.records(ArtifactType.VALIDATION_REPORT) if r.metadata["kind"] == "engine"
    ]
    assert evidence.metadata["passed"] is False


def test_the_stage_cache_key_covers_every_precision(
    tmp_path: Path, make_run: MakeRun, fake_trt: Callable[..., FakeCalls]
) -> None:
    fake_trt(None)
    run = Pipeline(
        pipeline_config(tmp_path), make_run(), default_stages(), fake_environment(gpu_ok=True)
    )
    run.execute()
    records = ArtifactStore(run.run).records(ArtifactType.ENGINE)
    assert set(latest_engine_per_precision(records)) == {Precision.FP32, Precision.FP16}
    again = run.execute()
    assert (
        next(o for o in again.stages if o.name == "validate_engine").status is StageStatus.SKIPPED
    )


# --------------------------------------------------------------------------- the command


def test_validate_engine_command_needs_a_gpu(
    exported: tuple[Path, TrtshipConfig, LoadedModel, ModelSignature],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(env, "probe_nvidia_gpu", lambda: ([], env._missing(env.NVIDIA_GPU, "none")))
    onnx_path, config, _, _ = exported
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    engine = tmp_path / "m.plan"
    engine.write_bytes(b"x")
    result = runner.invoke(
        app,
        [
            "validate",
            "engine",
            str(cfg),
            str(engine),
            "--onnx",
            str(onnx_path),
            "--precision",
            "fp32",
        ],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 3
    assert "validating a TensorRT engine" in result.output


@pytest.mark.parametrize(("transform", "code"), [(None, 0), (noisy(0.5), 6)])
def test_validate_engine_command(
    exported: tuple[Path, TrtshipConfig, LoadedModel, ModelSignature],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transform: Transform | None,
    code: int,
) -> None:
    onnx_path, config, model, _ = exported
    monkeypatch.setattr(env, "require_tensorrt", lambda purpose: None)
    monkeypatch.setattr(env, "require", lambda name, purpose: None)
    monkeypatch.setattr(
        "trtship.cli.commands.validate_cmd.TensorRTExecutor",
        lambda path: TorchBackedExecutor(model, transform),
    )
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    engine = tmp_path / "m.plan"
    engine.write_bytes(b"plan")
    report_path = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "validate", "engine", str(cfg), str(engine), "--onnx", str(onnx_path),
            "--precision", "fp16", "-o", str(report_path), "--json",
        ],
    )  # fmt: skip
    assert result.exit_code == code, result.output
    assert json.loads(report_path.read_text())["passed"] is (code == 0)
    assert json.loads(result.stdout)["results"][0]["precision"] == "fp16"
