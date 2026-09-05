"""The hardware CI job must never count skipped tests as a pass."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "require_no_skips.py"
_spec = importlib.util.spec_from_file_location("require_no_skips", _SCRIPT)
assert _spec is not None
assert _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
sys.modules["require_no_skips"] = gate
_spec.loader.exec_module(gate)


def _report(tmp_path: Path, **counts: int) -> Path:
    attrs = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0} | counts
    body = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    path = tmp_path / "r.xml"
    path.write_text(f'<testsuites><testsuite name="pytest" {body}/></testsuites>', "utf-8")
    return path


def test_all_passing_is_accepted(tmp_path: Path) -> None:
    assert gate.main(["x", str(_report(tmp_path, tests=5))]) == 0


@pytest.mark.parametrize(
    "counts",
    [
        {"tests": 0},  # nothing collected
        {"tests": 5, "skipped": 5},  # everything skipped
        {"tests": 5, "skipped": 1},  # partially skipped
        {"tests": 5, "failures": 1},
        {"tests": 5, "errors": 1},
    ],
)
def test_anything_but_a_clean_run_is_rejected(tmp_path: Path, counts: dict[str, int]) -> None:
    assert gate.main(["x", str(_report(tmp_path, **counts))]) == 1


def test_usage_error(tmp_path: Path) -> None:
    assert gate.main(["x"]) == 2
