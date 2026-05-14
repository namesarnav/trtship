"""Helpers shared by the concrete stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trtship.config import TrtshipConfig
from trtship.utils.fs import atomic_write_json


def model_slice(config: TrtshipConfig) -> dict[str, Any]:
    """The model description for cache keys. The weights *path* is excluded: the weights hash,
    which is part of the key, identifies the content."""
    return config.model.model_dump(mode="json", exclude={"path"})


def profiles_slice(config: TrtshipConfig) -> list[Any]:
    return [p.model_dump(mode="json") for p in config.tensorrt.profiles]


def write_report(path: Path, payload: dict[str, Any]) -> Path:
    atomic_write_json(path, payload)
    return path
