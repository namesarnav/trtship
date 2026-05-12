from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.helpers import fake_environment
from trtship.artifacts import ArtifactType, RunDirectory, RunStatus, StageStatus
from trtship.config import TrtshipConfig
from trtship.errors import (
    ArtifactError,
    ConfigError,
    EnvironmentUnavailableError,
    ValidationFailedError,
)
from trtship.pipeline import (
    Pipeline,
    Stage,
    StageContext,
    StageOutcome,
    StageResult,
    order_stages,
    select_stages,
)
from trtship.utils.env import EnvironmentReport

MakeRun = Callable[..., RunDirectory]


class FakeStageBase(Stage):
    """A stage that writes a small file. Class-level metadata comes from ``FakeStage``."""

    def __init__(
        self,
        *,
        content: bytes = b"data",
        skip: str | None = None,
        fail: BaseException | None = None,
    ) -> None:
        self.content = content
        self.skip = skip
        self.fail = fail
        self.calls = 0

    def skip_reason(self, config: TrtshipConfig) -> str | None:
        return self.skip

    def config_slice(self, config: TrtshipConfig) -> Any:
        return {"seed": config.seed, "content": self.content.decode()}

    def run(self, ctx: StageContext) -> StageResult:
        self.calls += 1
        for artifact_type in self.requires:
            ctx.input(artifact_type)
        if self.fail is not None:
            (ctx.scratch / "half-written.bin").write_bytes(b"partial")
            raise self.fail
        for artifact_type in self.produces:
            path = ctx.scratch / f"{self.name}-{artifact_type.value}.bin"
            path.write_bytes(self.content)
            ctx.publish(path, artifact_type, metadata={"by": self.name})
        return StageResult(metrics={"calls": self.calls}, warnings=["heads up"])


def FakeStage(
    name: str,
    *,
    requires: tuple[ArtifactType, ...] = (),
    uses: tuple[ArtifactType, ...] = (),
    produces: tuple[ArtifactType, ...] = (),
    capabilities: tuple[str, ...] = (),
    cacheable: bool = True,
    content: bytes = b"data",
    skip: str | None = None,
    fail: BaseException | None = None,
) -> FakeStageBase:
    cls = type(
        f"Fake_{name}",
        (FakeStageBase,),
        {
            "name": name,
            "requires": requires,
            "uses": uses,
            "produces": produces,
            "requires_capabilities": capabilities,
            "cacheable": cacheable,
        },
    )
    stage: FakeStageBase = cls(content=content, skip=skip, fail=fail)
    return stage


def config_for(tmp_path: Path, **extra: Any) -> TrtshipConfig:
    return TrtshipConfig.model_validate(
        {
            "model": {
                "name": "m",
                "kind": "module",
                "factory": "trtship_fixtures.models:tiny_mlp",
                "inputs": [{"name": "x", "shape": [2, 16]}],
            },
            "artifacts": {"root": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")},
            **extra,
        }
    )


def pipeline_for(
    tmp_path: Path,
    stages: list[Stage],
    make_run: MakeRun,
    *,
    run_id: str = "r1",
    environment: EnvironmentReport | None = None,
    progress: Callable[[StageOutcome], None] | None = None,
    **config_extra: Any,
) -> Pipeline:
    config = config_for(tmp_path, **config_extra)
    run = make_run(run_id)
    return Pipeline(
        config, run, stages, environment or fake_environment(gpu_ok=True), progress=progress
    )


A, B, C = ArtifactType.ONNX, ArtifactType.ONNX_OPTIMIZED, ArtifactType.ENGINE


def chain() -> list[Stage]:
    return [
        FakeStage("first", produces=(A,)),
        FakeStage("second", requires=(A,), produces=(B,)),
        FakeStage("third", requires=(A,), uses=(B,), produces=(C,)),
    ]


# --------------------------------------------------------------------------- ordering / selection


def test_stages_are_ordered_by_the_artifacts_they_consume() -> None:
    first, second, third = chain()
    assert [s.name for s in order_stages([third, second, first])] == ["first", "second", "third"]


def test_optional_inputs_also_create_ordering_edges() -> None:
    producer = FakeStage("producer", produces=(B,))
    consumer = FakeStage("consumer", uses=(B,))
    assert [s.name for s in order_stages([consumer, producer])] == ["producer", "consumer"]


def test_independent_stages_keep_their_given_order() -> None:
    stages = [FakeStage("x", produces=(A,)), FakeStage("y", produces=(B,)), FakeStage("z")]
    assert [s.name for s in order_stages(stages)] == ["x", "y", "z"]


def test_cycles_and_duplicate_names_are_rejected() -> None:
    loop_a = FakeStage("a", requires=(B,), produces=(A,))
    loop_b = FakeStage("b", requires=(A,), produces=(B,))
    with pytest.raises(ConfigError, match="cycle"):
        order_stages([loop_a, loop_b])
    with pytest.raises(ConfigError, match="duplicate stage names"):
        order_stages([FakeStage("a"), FakeStage("a")])


def test_stage_selection() -> None:
    stages = order_stages(chain())

    def names(**kw: str) -> list[str]:
        return [s.name for s in select_stages(stages, **kw)]

    assert names() == ["first", "second", "third"]
    assert names(from_stage="second") == ["second", "third"]
    assert names(until="second") == ["first", "second"]
    assert names(from_stage="second", until="second") == ["second"]
    assert names(only="third") == ["third"]


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"from_stage": "nope"}, "unknown stage 'nope' for --from"),
        ({"until": "nope"}, "unknown stage 'nope' for --until"),
        ({"only": "nope"}, "unknown stage 'nope' for --only"),
        ({"only": "first", "until": "second"}, "cannot be combined"),
        ({"only": "first", "from_stage": "first"}, "cannot be combined"),
        ({"from_stage": "third", "until": "first"}, "comes after"),
    ],
)
def test_bad_selections_are_config_errors(kwargs: dict[str, str], fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment):
        select_stages(order_stages(chain()), **kwargs)


def test_plan_reports_skips(tmp_path: Path, make_run: MakeRun) -> None:
    stages: list[Stage] = [FakeStage("a", produces=(A,)), FakeStage("b", skip="disabled in config")]
    pipeline = pipeline_for(tmp_path, stages, make_run)
    plan = pipeline.plan()
    assert [(p.name, p.action, p.reason) for p in plan] == [
        ("a", "run", None),
        ("b", "skip", "disabled in config"),
    ]


# --------------------------------------------------------------------------- execution


def test_a_successful_run_records_everything(tmp_path: Path, make_run: MakeRun) -> None:
    seen: list[str] = []
    pipeline = pipeline_for(tmp_path, chain(), make_run, progress=lambda o: seen.append(o.name))
    result = pipeline.execute()

    assert result.status is RunStatus.SUCCEEDED
    assert [s.name for s in result.stages] == ["first", "second", "third"] == seen
    assert all(s.status is StageStatus.SUCCEEDED for s in result.stages)
    assert result.stages[0].warnings == ["heads up"]
    manifest = pipeline.run.read_manifest()
    assert manifest.status is RunStatus.SUCCEEDED
    assert set(manifest.stages) == {"first", "second", "third"}
    record = manifest.stages["second"]
    assert record.status is StageStatus.SUCCEEDED
    assert record.cache_key
    assert record.started_at is not None
    assert record.finished_at is not None
    assert len(record.artifact_ids) == 1
    # provenance: the output's parent is the input it required
    first_id = manifest.stages["first"].artifact_ids[0]
    assert manifest.artifacts[record.artifact_ids[0]].parents == [first_id]
    assert not (pipeline.run.path / ".work").exists()  # scratch is cleaned up


def test_rerunning_a_finished_run_skips_everything_as_up_to_date(
    tmp_path: Path, make_run: MakeRun
) -> None:
    stages = chain()
    pipeline = pipeline_for(tmp_path, stages, make_run)
    first = pipeline.execute()
    second = pipeline.execute()
    assert all(s.status is StageStatus.SKIPPED and s.reason == "up to date" for s in second.stages)
    assert [s.artifacts for s in second.stages] == [s.artifacts for s in first.stages]
    assert [s.calls for s in stages if isinstance(s, FakeStageBase)] == [1, 1, 1]
    assert len(pipeline.run.read_manifest().artifacts) == 3


def test_a_tampered_artifact_is_regenerated_not_trusted(tmp_path: Path, make_run: MakeRun) -> None:
    stages = chain()
    pipeline = pipeline_for(tmp_path, stages, make_run)
    pipeline.execute()
    record = pipeline.store.latest(A)
    assert record is not None
    original = pipeline.store.absolute(record)
    original.write_bytes(b"corrupted")
    assert pipeline.store.verify_all()  # the damage is detectable

    result = pipeline.execute()
    by_name = {o.name: o for o in result.stages}
    assert by_name["first"].status is StageStatus.SUCCEEDED  # rebuilt, not skipped
    assert by_name["second"].status is StageStatus.SKIPPED  # its input content is unchanged
    assert stages[0].calls == 2  # type: ignore[attr-defined]
    healed = pipeline.store.latest(A)
    assert healed is not None
    assert pipeline.store.absolute(healed) != original  # a new version, nothing overwritten
    assert original.read_bytes() == b"corrupted"  # the evidence is left as it was
    pipeline.store.verify(healed)


def test_shared_cache_reuses_stage_outputs_in_a_new_run(tmp_path: Path, make_run: MakeRun) -> None:
    stages = chain()
    first = pipeline_for(tmp_path, stages, make_run, run_id="one")
    first.execute()
    second = pipeline_for(tmp_path, chain(), make_run, run_id="two")
    result = second.execute()
    assert all(s.status is StageStatus.CACHED for s in result.stages), result.stages
    assert all(s.reason == "reused from cache" for s in result.stages)
    for record in second.run.read_manifest().artifacts.values():
        assert "reused_from_cache" in record.metadata
    a = first.store.latest(A)
    b = second.store.latest(A)
    assert a is not None
    assert b is not None
    assert a.sha256 == b.sha256
    assert second.store.absolute(b) != first.store.absolute(a)  # a copy inside the second run


def test_changed_config_or_content_misses_the_cache(tmp_path: Path, make_run: MakeRun) -> None:
    pipeline_for(tmp_path, chain(), make_run, run_id="one").execute()
    changed = [FakeStage("first", produces=(A,), content=b"different"), *chain()[1:]]
    result = pipeline_for(tmp_path, changed, make_run, run_id="two").execute()
    assert result.stages[0].status is StageStatus.SUCCEEDED  # different config slice
    assert result.stages[1].status is StageStatus.SUCCEEDED  # its input hash changed too


def test_cache_can_be_disabled(tmp_path: Path, make_run: MakeRun) -> None:
    pipeline_for(
        tmp_path,
        chain(),
        make_run,
        run_id="one",
        artifacts={
            "root": str(tmp_path / "runs"),
            "cache_dir": str(tmp_path / "cache"),
            "reuse_cache": False,
        },
    ).execute()
    assert not (tmp_path / "cache").exists()


def test_non_cacheable_stages_always_run(tmp_path: Path, make_run: MakeRun) -> None:
    measure = FakeStage("measure", produces=(ArtifactType.BENCHMARK_REPORT,), cacheable=False)
    pipeline = pipeline_for(tmp_path, [measure], make_run)
    pipeline.execute()
    pipeline.execute()
    assert measure.calls == 2
    again = pipeline_for(
        tmp_path,
        [FakeStage("measure", produces=(ArtifactType.BENCHMARK_REPORT,), cacheable=False)],
        make_run,
        run_id="r2",
    )
    assert again.execute().stages[0].status is StageStatus.SUCCEEDED  # never CACHED


def test_disabled_stages_are_skipped_with_a_reason(tmp_path: Path, make_run: MakeRun) -> None:
    stages: list[Stage] = [FakeStage("a", produces=(A,)), FakeStage("b", skip="not applicable")]
    result = pipeline_for(tmp_path, stages, make_run).execute()
    assert [(s.name, s.status, s.reason) for s in result.stages] == [
        ("a", StageStatus.SUCCEEDED, None),
        ("b", StageStatus.SKIPPED, "not applicable"),
    ]


def test_publishing_never_overwrites_and_versions_conflicting_content(
    tmp_path: Path, make_run: MakeRun
) -> None:
    pipeline = pipeline_for(tmp_path, [FakeStage("s")], make_run)
    run = pipeline.run

    def publish(data: bytes) -> str:
        ctx = pipeline._context(pipeline.ordered[0])
        (ctx.scratch / "out.json").write_bytes(data)
        record = ctx.publish(ctx.scratch / "out.json", ArtifactType.REPORT)
        return record.path

    assert publish(b"v1") == "reports/out.json"
    assert publish(b"v1") == "reports/out.json"  # identical content is reused
    assert publish(b"v2") == "reports/out.2.json"
    assert publish(b"v3") == "reports/out.3.json"
    assert (run.path / "reports" / "out.json").read_bytes() == b"v1"  # the original is untouched


# --------------------------------------------------------------------------- failure / recovery


def test_a_failing_stage_is_recorded_and_stops_the_run(tmp_path: Path, make_run: MakeRun) -> None:
    boom = ValidationFailedError("outputs differ", details={"why": "test"})
    stages: list[Stage] = [
        FakeStage("first", produces=(A,)),
        FakeStage("second", requires=(A,), fail=boom),
        FakeStage("third", requires=(A,), produces=(C,)),
    ]
    seen: list[str] = []
    pipeline = pipeline_for(tmp_path, stages, make_run, progress=lambda o: seen.append(o.name))
    with pytest.raises(ValidationFailedError, match="outputs differ"):
        pipeline.execute()

    assert seen == ["first"]
    manifest = pipeline.run.read_manifest()
    assert manifest.status is RunStatus.FAILED
    assert manifest.stages["first"].status is StageStatus.SUCCEEDED
    failed = manifest.stages["second"]
    assert failed.status is StageStatus.FAILED
    assert failed.error is not None
    assert failed.error["error"] == "ValidationFailedError"
    assert failed.error["exit_code"] == 6
    assert failed.error["details"] == {"why": "test"}
    assert "third" not in manifest.stages
    assert not (pipeline.run.path / ".work").exists()  # the half-written file is gone
    assert list((pipeline.run.path / "artifacts").iterdir()) == [
        pipeline.store.absolute(pipeline.store.get(manifest.stages["first"].artifact_ids[0]))
    ]


def test_unexpected_exceptions_are_recorded_too(tmp_path: Path, make_run: MakeRun) -> None:
    pipeline = pipeline_for(tmp_path, [FakeStage("s", fail=RuntimeError("kaboom"))], make_run)
    with pytest.raises(RuntimeError, match="kaboom"):
        pipeline.execute()
    error = pipeline.run.read_manifest().stages["s"].error
    assert error == {"error": "RuntimeError", "message": "kaboom"}


def test_interrupts_are_recorded_and_propagate(tmp_path: Path, make_run: MakeRun) -> None:
    pipeline = pipeline_for(tmp_path, [FakeStage("s", fail=KeyboardInterrupt())], make_run)
    with pytest.raises(KeyboardInterrupt):
        pipeline.execute()
    manifest = pipeline.run.read_manifest()
    assert manifest.status is RunStatus.FAILED
    assert manifest.stages["s"].status is StageStatus.FAILED


def test_resuming_after_a_transient_failure_reruns_only_what_is_needed(
    tmp_path: Path, make_run: MakeRun
) -> None:
    flaky = FakeStage("second", requires=(A,), produces=(B,), fail=RuntimeError("GPU busy"))
    first_stage = FakeStage("first", produces=(A,))
    pipeline = pipeline_for(tmp_path, [first_stage, flaky], make_run)
    with pytest.raises(RuntimeError):
        pipeline.execute()
    assert first_stage.calls == 1

    flaky.fail = None  # the transient problem is fixed
    result = pipeline.execute()
    assert [(s.name, s.status) for s in result.stages] == [
        ("first", StageStatus.SKIPPED),  # reused: nothing to redo
        ("second", StageStatus.SUCCEEDED),
    ]
    assert first_stage.calls == 1
    assert pipeline.run.read_manifest().status is RunStatus.SUCCEEDED
    assert pipeline.run.read_manifest().stages["second"].error is None


def test_a_stage_whose_input_is_missing_fails_with_a_clear_error(
    tmp_path: Path, make_run: MakeRun
) -> None:
    pipeline = pipeline_for(tmp_path, chain(), make_run)
    with pytest.raises(ArtifactError, match="needs a onnx artifact") as info:
        pipeline.execute(only="second")
    assert "Run the stage that produces it" in (info.value.hint or "")
    assert pipeline.run.read_manifest().stages["second"].status is StageStatus.FAILED


def test_selection_runs_only_the_selected_stages(tmp_path: Path, make_run: MakeRun) -> None:
    pipeline = pipeline_for(tmp_path, chain(), make_run)
    assert [s.name for s in pipeline.execute(until="second").stages] == ["first", "second"]
    result = pipeline.execute(from_stage="second")
    assert [(s.name, s.status) for s in result.stages] == [
        ("second", StageStatus.SKIPPED),
        ("third", StageStatus.SUCCEEDED),
    ]
    assert [s.name for s in pipeline.execute(only="first").stages] == ["first"]


# --------------------------------------------------------------------------- preflight


def test_preflight_fails_before_any_stage_runs(tmp_path: Path, make_run: MakeRun) -> None:
    cpu = FakeStage("cpu-stage", produces=(A,))
    gpu = FakeStage(
        "gpu-stage", requires=(A,), produces=(C,), capabilities=("nvidia_gpu", "tensorrt")
    )
    pipeline = pipeline_for(
        tmp_path, [cpu, gpu], make_run, environment=fake_environment(gpu_ok=False)
    )
    with pytest.raises(EnvironmentUnavailableError) as info:
        pipeline.execute()
    assert cpu.calls == 0  # nothing ran
    assert pipeline.run.read_manifest().stages == {}
    assert "gpu-stage" in info.value.message
    assert "nvidia_gpu is missing" in info.value.message
    assert "tensorrt is missing" in info.value.message
    assert "--until cpu-stage" in (info.value.hint or "")
    assert info.value.exit_code == 3


def test_preflight_ignores_stages_that_are_not_selected_or_are_skipped(
    tmp_path: Path, make_run: MakeRun
) -> None:
    cpu = FakeStage("cpu-stage", produces=(A,))
    gpu = FakeStage("gpu-stage", requires=(A,), capabilities=("tensorrt",))
    off = FakeStage("disabled", capabilities=("tensorrt",), skip="off")
    env = fake_environment(gpu_ok=False)
    assert pipeline_for(tmp_path, [cpu, gpu], make_run, run_id="a", environment=env).execute(
        until="cpu-stage"
    )
    assert pipeline_for(tmp_path, [cpu, off], make_run, run_id="b", environment=env).execute()


def test_capabilities_are_part_of_gpu_stage_cache_keys(tmp_path: Path, make_run: MakeRun) -> None:
    gpu = FakeStage("g", produces=(A,), capabilities=("nvidia_gpu",))
    pipeline = pipeline_for(tmp_path, [gpu], make_run)
    assert "Test GPU" in pipeline._tools(gpu)["gpus"]  # type: ignore[operator]
    cpu = FakeStage("c", produces=(B,))
    assert "gpus" not in pipeline._tools(cpu)
