"""Access to the TensorRT Python bindings.

TensorRT is imported lazily and only after the GPU and the bindings have been verified, so importing
trtship (or running any CPU stage) never touches it.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from types import ModuleType
from typing import Final

from trtship.errors import EnvironmentUnavailableError
from trtship.utils import env

MIN_SUPPORTED: Final = (8, 6)


@dataclass(frozen=True, order=True)
class TrtVersion:
    major: int
    minor: int
    patch: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def parse_version(text: str) -> TrtVersion:
    """``'10.3.0.26'`` -> ``TrtVersion(10, 3, 0)``. Extra components (build numbers) are ignored."""
    match = re.match(r"^\s*(\d+)\.(\d+)(?:\.(\d+))?", text)
    if match is None:
        raise ValueError(f"cannot parse TensorRT version {text!r}")
    return TrtVersion(int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


def check_supported(version: TrtVersion) -> None:
    if (version.major, version.minor) < MIN_SUPPORTED:
        raise EnvironmentUnavailableError(
            f"TensorRT {version} is not supported; trtship needs "
            f"{MIN_SUPPORTED[0]}.{MIN_SUPPORTED[1]} or newer",
            hint="Upgrade TensorRT (`uv sync --extra trt`).",
        )


def load_tensorrt(*, purpose: str = "TensorRT") -> ModuleType:
    """Import ``tensorrt`` after verifying a usable GPU and working bindings."""
    env.require_tensorrt(purpose=purpose)
    module = importlib.import_module("tensorrt")
    check_supported(parse_version(str(module.__version__)))
    return module
