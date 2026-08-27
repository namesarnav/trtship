"""Facts the documentation states about the pipeline must match the code."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from trtship import errors
from trtship.pipeline.stages import default_stages

OVERVIEW = (Path(__file__).resolve().parents[2] / "docs/pipeline/overview.md").read_text("utf-8")


def error_classes() -> list[type[errors.TrtshipError]]:
    return [
        cls
        for _, cls in inspect.getmembers(errors, inspect.isclass)
        if issubclass(cls, errors.TrtshipError)
    ]


def test_every_default_stage_is_named_in_the_overview_diagram() -> None:
    diagram = OVERVIEW.split("```")[1]
    for stage in default_stages():
        assert stage.name in diagram, f"stage {stage.name!r} is missing from the overview diagram"


def test_the_overview_diagram_names_no_stage_that_does_not_exist() -> None:
    diagram = OVERVIEW.split("```")[1]
    names = {s.name for s in default_stages()}
    assert set(re.findall(r"[a-z_]+", diagram)) <= names


def test_the_exit_code_table_matches_the_error_model() -> None:
    table = OVERVIEW.split("## Exit codes")[1]
    documented = {int(code) for code in re.findall(r"^\| (\d+) \|", table, re.MULTILINE)}
    assert documented == {0, errors.EXIT_UNEXPECTED, *(c.exit_code for c in error_classes())}


@pytest.mark.parametrize("cls", error_classes(), ids=lambda c: c.__name__)
def test_every_error_class_has_an_exit_code_and_a_docstring(cls: type[errors.TrtshipError]) -> None:
    assert cls.__doc__
    assert 1 <= cls.exit_code < errors.EXIT_UNEXPECTED
