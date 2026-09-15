from __future__ import annotations

from pathlib import Path

from tests.conftest import minimal_config_dict
from trtship.artifacts import RunDirectory
from trtship.config import TrtshipConfig
from trtship.utils.env import EnvironmentReport


def test_the_manifest_records_the_seed(
    tmp_path: Path, environment_report: EnvironmentReport
) -> None:
    data = minimal_config_dict()
    data["seed"] = 99
    config = TrtshipConfig.model_validate(data)
    run = RunDirectory.create(tmp_path / "runs", config, environment_report, run_id="s")
    manifest = run.read_manifest()
    assert manifest.seed == 99
    assert manifest.model_sha256 is None  # filled in when the model is loaded
