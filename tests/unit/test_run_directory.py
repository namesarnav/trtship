from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from trtship import __version__
from trtship.artifacts import (
    RunDirectory,
    RunStatus,
    StageStatus,
    new_run_id,
    validate_run_id,
)
from trtship.config import TrtshipConfig, config_hash, load_config
from trtship.errors import ArtifactError, ConfigError
from trtship.utils.env import EnvironmentReport, probe_all


@pytest.fixture(scope="module")
def environment() -> EnvironmentReport:
    return probe_all()


@pytest.fixture
def config(config_dict: dict[str, Any]) -> TrtshipConfig:
    return TrtshipConfig.model_validate(config_dict)


def test_new_run_id_uses_given_date_and_random_token() -> None:
    moment = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
    a, b = new_run_id(moment), new_run_id(moment)
    assert a.startswith("2026-09-19_")
    assert len(a.split("_")[1]) == 6
    assert a != b


def test_new_run_id_defaults_to_the_real_current_date() -> None:
    assert new_run_id().startswith(datetime.now(UTC).strftime("%Y-%m-%d"))


@pytest.mark.parametrize(
    "bad", ["", ".", "..", "../x", "a/b", "a b", "-lead", "a..b", "x" * 65, "a\\b"]
)
def test_invalid_run_ids_rejected(bad: str) -> None:
    with pytest.raises(ConfigError):
        validate_run_id(bad)


@pytest.mark.parametrize("good", ["2026-09-19_abc123", "my-run.1", "A_b-C", "x" * 64])
def test_valid_run_ids_accepted(good: str) -> None:
    assert validate_run_id(good) == good


def test_create_writes_full_layout_and_metadata(
    tmp_path: Path, config: TrtshipConfig, environment: EnvironmentReport
) -> None:
    run = RunDirectory.create(tmp_path / "runs", config, environment, run_id="r1")
    assert run.path == tmp_path / "runs" / "r1"
    for sub in ("artifacts", "validation", "benchmarks", "reports", "logs"):
        assert (run.path / sub).is_dir()
    assert run.config_path.is_file()
    env_json = json.loads(run.environment_path.read_text())
    assert env_json["trtship_version"] == __version__
    manifest = run.read_manifest()
    assert manifest.run_id == "r1"
    assert manifest.status is RunStatus.CREATED
    assert manifest.config_sha256 == config_hash(config)
    assert manifest.trtship_version == __version__
    assert manifest.created_at.tzinfo is not None


def test_create_refuses_to_overwrite_existing_run(
    tmp_path: Path, config: TrtshipConfig, environment: EnvironmentReport
) -> None:
    RunDirectory.create(tmp_path, config, environment, run_id="dup")
    with pytest.raises(ArtifactError, match="already exists"):
        RunDirectory.create(tmp_path, config, environment, run_id="dup")


def test_create_rejects_traversal_run_id(
    tmp_path: Path, config: TrtshipConfig, environment: EnvironmentReport
) -> None:
    with pytest.raises(ConfigError):
        RunDirectory.create(tmp_path / "runs", config, environment, run_id="../escape")
    assert not (tmp_path / "escape").exists()


def test_config_snapshot_reloads_to_the_same_config(
    tmp_path: Path, config: TrtshipConfig, environment: EnvironmentReport
) -> None:
    run = RunDirectory.create(tmp_path, config, environment, run_id="snap")
    assert load_config(run.config_path) == config


def test_open_requires_a_manifest(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="not a trtship run directory"):
        RunDirectory.open(tmp_path)


def test_update_stage_persists_and_bumps_updated_at(
    tmp_path: Path, config: TrtshipConfig, environment: EnvironmentReport
) -> None:
    run = RunDirectory.create(tmp_path, config, environment, run_id="u")
    before = run.read_manifest()
    record = run.update_stage(
        "export", status=StageStatus.SUCCEEDED, cache_key="k", artifact_ids=["onnx-1"]
    )
    assert record.status is StageStatus.SUCCEEDED
    after = RunDirectory.open(run.path).read_manifest()
    assert after.stages["export"].artifact_ids == ["onnx-1"]
    assert after.updated_at >= before.updated_at
    run.update_stage("export", status=StageStatus.CACHED)
    reread = run.read_manifest().stages["export"]
    assert reread.status is StageStatus.CACHED
    assert reread.cache_key == "k"  # untouched fields survive


def test_corrupt_manifest_is_an_artifact_error(
    tmp_path: Path, config: TrtshipConfig, environment: EnvironmentReport
) -> None:
    run = RunDirectory.create(tmp_path, config, environment, run_id="c")
    run.manifest_path.write_text("{not json")
    with pytest.raises(ArtifactError, match="corrupt"):
        run.read_manifest()
    run.manifest_path.write_text('{"run_id": "x"}')
    with pytest.raises(ArtifactError, match="corrupt"):
        run.read_manifest()


def test_latest_picks_most_recently_created(
    tmp_path: Path, config: TrtshipConfig, environment: EnvironmentReport
) -> None:
    RunDirectory.create(tmp_path, config, environment, run_id="first")
    second = RunDirectory.create(tmp_path, config, environment, run_id="second")
    (tmp_path / "stray").mkdir()  # not a run
    assert RunDirectory.latest(tmp_path).path == second.path
    with pytest.raises(ArtifactError, match="no runs"):
        RunDirectory.latest(tmp_path / "missing")
