"""Persistent records: artifacts, stage outcomes, and the run manifest."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

MANIFEST_SCHEMA_VERSION = 1


class ArtifactType(StrEnum):
    MODEL_REPORT = "model_report"
    ONNX = "onnx"
    ONNX_OPTIMIZED = "onnx_optimized"
    CALIBRATION_CACHE = "calibration_cache"
    ENGINE = "engine"
    VALIDATION_REPORT = "validation_report"
    BENCHMARK_REPORT = "benchmark_report"
    TRITON_REPOSITORY = "triton_repository"
    REPORT = "report"


class ArtifactRecord(BaseModel):
    """An immutable, hash-identified product of a pipeline stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    type: ArtifactType
    path: str  # POSIX path relative to the run directory; never absolute
    sha256: str
    size_bytes: int = Field(ge=0)
    created_at: datetime
    stage: str
    parents: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def make_id(artifact_type: ArtifactType, sha256: str) -> str:
        return f"{artifact_type.value}-{sha256[:12]}"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CACHED = "cached"


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class StageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: StageStatus = StageStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cache_key: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = MANIFEST_SCHEMA_VERSION
    run_id: str
    created_at: datetime
    updated_at: datetime
    trtship_version: str
    status: RunStatus = RunStatus.CREATED
    config_sha256: str
    model_sha256: str | None = None
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    artifacts: dict[str, ArtifactRecord] = Field(default_factory=dict)
