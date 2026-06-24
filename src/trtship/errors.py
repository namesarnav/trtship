"""Structured error model.

Every expected failure derives from :class:`TrtshipError` and carries a stable process exit
code. Anything that is *not* a ``TrtshipError`` is a bug and exits with ``EXIT_UNEXPECTED``.
"""

from __future__ import annotations

from typing import Any, ClassVar

EXIT_OK = 0
EXIT_UNEXPECTED = 70


class TrtshipError(Exception):
    """Base class for expected, user-actionable failures."""

    exit_code: ClassVar[int] = 1

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": type(self).__name__,
            "exit_code": self.exit_code,
            "message": self.message,
            "hint": self.hint,
            "details": self.details,
        }


class ConfigError(TrtshipError):
    """Invalid configuration or CLI usage."""

    exit_code = 2


class EnvironmentUnavailableError(TrtshipError):
    """A required capability (GPU, TensorRT, Docker, Triton, ...) is missing or unusable."""

    exit_code = 3


class ModelError(TrtshipError):
    """Model loading or inspection failed."""

    exit_code = 4


class ExportError(TrtshipError):
    """ONNX export failed."""

    exit_code = 5


class ValidationFailedError(TrtshipError):
    """Graph or numerical validation exceeded its tolerance."""

    exit_code = 6


class EngineBuildError(TrtshipError):
    """TensorRT engine build failed (details include unsupported-operation reports)."""

    exit_code = 7


class EngineRuntimeError(EngineBuildError):
    """A TensorRT engine could not be loaded or executed."""


class CalibrationError(TrtshipError):
    """INT8 calibration failed or its data was invalid."""

    exit_code = 8


class BenchmarkError(TrtshipError):
    """A benchmark could not be run or produced unusable data."""

    exit_code = 9


class TritonError(TrtshipError):
    """Triton repository generation, server lifecycle, or client failure."""

    exit_code = 10


class ArtifactError(TrtshipError):
    """Artifact conflict, missing artifact, or hash mismatch."""

    exit_code = 11


class ArtifactConflictError(ArtifactError):
    """Attempted to overwrite an existing artifact with different content."""


class CommandError(TrtshipError):
    """An external command exited non-zero or could not be started."""

    exit_code = 1
