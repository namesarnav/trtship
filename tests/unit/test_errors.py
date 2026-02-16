from __future__ import annotations

import pytest

from trtship import errors


@pytest.mark.parametrize(
    ("cls", "code"),
    [
        (errors.TrtshipError, 1),
        (errors.ConfigError, 2),
        (errors.EnvironmentUnavailableError, 3),
        (errors.ModelError, 4),
        (errors.ExportError, 5),
        (errors.ValidationFailedError, 6),
        (errors.EngineBuildError, 7),
        (errors.CalibrationError, 8),
        (errors.BenchmarkError, 9),
        (errors.TritonError, 10),
        (errors.ArtifactError, 11),
        (errors.ArtifactConflictError, 11),
    ],
)
def test_exit_codes_are_stable(cls: type[errors.TrtshipError], code: int) -> None:
    assert cls("boom").exit_code == code


def test_exit_codes_are_unique_per_category() -> None:
    codes = {
        c.exit_code
        for c in (
            errors.ConfigError,
            errors.EnvironmentUnavailableError,
            errors.ModelError,
            errors.ExportError,
            errors.ValidationFailedError,
            errors.EngineBuildError,
            errors.CalibrationError,
            errors.BenchmarkError,
            errors.TritonError,
            errors.ArtifactError,
        )
    }
    assert len(codes) == 10
    assert errors.EXIT_UNEXPECTED not in codes


def test_to_dict_round_trips_fields() -> None:
    exc = errors.ExportError("bad op", hint="use opset 17", details={"op": "Foo"})
    assert exc.to_dict() == {
        "error": "ExportError",
        "exit_code": 5,
        "message": "bad op",
        "hint": "use opset 17",
        "details": {"op": "Foo"},
    }
    assert str(exc) == "bad op"


def test_details_are_copied() -> None:
    details = {"a": 1}
    exc = errors.TrtshipError("x", details=details)
    details["a"] = 2
    assert exc.details == {"a": 1}
