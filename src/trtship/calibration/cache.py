"""The calibration cache artifact: TensorRT's opaque cache plus the metadata that justifies it.

A cache is a directory holding ``calibration.cache`` (written by TensorRT) and ``metadata.json``.
The metadata records what the scales were computed from. A cache is only reusable when the
things that determine the scales still match; :func:`compatibility_problems` says exactly which do
not, so a stale cache is never used silently.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, ValidationError

from trtship.calibration.dataset import DatasetIdentity
from trtship.errors import CalibrationError
from trtship.utils.fs import atomic_write_bytes, atomic_write_json
from trtship.utils.hashing import sha256_bytes

CACHE_FILE: Final = "calibration.cache"
METADATA_FILE: Final = "metadata.json"
SCHEMA_VERSION: Final = 1


class CalibrationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    created_at: datetime
    dataset: DatasetIdentity
    requested_samples: int
    sample_count: int  # samples actually used (a trailing partial batch is dropped)
    batch_size: int
    num_batches: int
    seed: int
    method: str
    preprocessing: dict[str, Any]
    input_name: str
    input_dtype: str
    sample_shape: list[int]
    tensorrt_version: str
    cuda_version: str | None
    gpu: str | None
    model_weights_sha256: str
    source_onnx_sha256: str
    cache_sha256: str
    representative: bool


def write_cache_dir(directory: Path, cache: bytes, metadata: CalibrationMetadata) -> None:
    """Write the cache directory. ``directory`` must not exist."""
    if directory.exists():
        raise CalibrationError(f"refusing to overwrite existing calibration cache: {directory}")
    if metadata.cache_sha256 != sha256_bytes(cache):
        raise CalibrationError("calibration metadata does not describe the cache bytes")
    directory.mkdir(parents=True)
    atomic_write_bytes(directory / CACHE_FILE, cache)
    atomic_write_json(directory / METADATA_FILE, metadata.model_dump(mode="json"))


def read_cache_dir(directory: Path) -> tuple[bytes, CalibrationMetadata]:
    """Read and integrity-check a cache directory."""
    cache_path, meta_path = directory / CACHE_FILE, directory / METADATA_FILE
    if not cache_path.is_file() or not meta_path.is_file():
        raise CalibrationError(
            f"{directory} is not a calibration cache (needs {CACHE_FILE} and {METADATA_FILE})"
        )
    try:
        metadata = CalibrationMetadata.model_validate(json.loads(meta_path.read_text("utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise CalibrationError(f"unreadable calibration metadata in {directory}: {exc}") from exc
    cache = cache_path.read_bytes()
    if sha256_bytes(cache) != metadata.cache_sha256:
        raise CalibrationError(
            f"calibration cache in {directory} does not match its metadata (modified or corrupt)"
        )
    return cache, metadata


def compatibility_problems(actual: CalibrationMetadata, expected: dict[str, Any]) -> list[str]:
    """Reasons ``actual`` cannot stand in for a fresh calibration described by ``expected``.

    ``expected`` maps a metadata field name (or ``dataset_fingerprint``) to the required value.
    """
    problems = []
    for key, wanted in expected.items():
        have: Any = (
            actual.dataset.fingerprint if key == "dataset_fingerprint" else getattr(actual, key)
        )
        if have != wanted:
            problems.append(f"{key}: cache has {have!r}, current run needs {wanted!r}")
    return problems
