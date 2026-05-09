from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from tests.conftest import minimal_config_dict
from trtship.artifacts import RunDirectory
from trtship.config import TrtshipConfig
from trtship.utils.env import EnvironmentReport
from trtship.utils.seed import seed_everything


def draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_seeding_makes_all_generators_repeatable() -> None:
    seed_everything(123)
    first = draw()
    seed_everything(123)
    assert draw() == first
    seed_everything(124)
    assert draw() != first


def test_seed_everything_reports_what_it_seeded() -> None:
    info = seed_everything(7)
    assert info["seed"] == 7
    assert {"python.random", "numpy.random", "torch.cpu"} <= set(info["seeded"])


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
