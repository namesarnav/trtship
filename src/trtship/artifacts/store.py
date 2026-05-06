"""Artifact store: register, look up, verify, and reuse the products of pipeline stages.

Rules:

* Artifacts are **immutable**. Registering a path again with the same content is a no-op;
  registering it with different content raises :class:`ArtifactConflictError`.
* Records store paths relative to the run directory, never absolute paths.
* Nothing is trusted on reuse: files are re-hashed and compared with the record before an artifact
  is handed to a stage or copied out of the shared cache.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from trtship.artifacts.records import ArtifactRecord, ArtifactType
from trtship.artifacts.run import RunDirectory
from trtship.errors import ArtifactConflictError, ArtifactError
from trtship.logging import get_logger
from trtship.utils.fs import atomic_write_json, path_size_bytes
from trtship.utils.hashing import sha256_path
from trtship.utils.timeutil import utc_now

log = get_logger(__name__)

# Where each artifact type lives inside a run directory.
_TYPE_DIRS: dict[ArtifactType, str] = {
    ArtifactType.MODEL_REPORT: "reports",
    ArtifactType.ONNX: "artifacts",
    ArtifactType.ONNX_OPTIMIZED: "artifacts",
    ArtifactType.CALIBRATION_CACHE: "artifacts",
    ArtifactType.ENGINE: "artifacts",
    ArtifactType.VALIDATION_REPORT: "validation",
    ArtifactType.BENCHMARK_REPORT: "benchmarks",
    ArtifactType.TRITON_REPOSITORY: "artifacts",
    ArtifactType.REPORT: "reports",
}


def _check_name(name: str) -> str:
    """Artifact file names are plain names: no separators, no traversal."""
    if not name or name in {".", ".."} or "/" in name or "\\" in name or name.startswith("."):
        raise ArtifactError(f"invalid artifact file name {name!r}")
    return name


class ArtifactStore:
    def __init__(self, run: RunDirectory) -> None:
        self.run = run

    # ------------------------------------------------------------------ paths

    def path_for(self, artifact_type: ArtifactType, filename: str) -> Path:
        """Where a stage should write an artifact of ``artifact_type`` named ``filename``."""
        directory = self.run.path / _TYPE_DIRS[artifact_type]
        directory.mkdir(parents=True, exist_ok=True)
        return directory / _check_name(filename)

    def absolute(self, record: ArtifactRecord) -> Path:
        path = (self.run.path / record.path).resolve()
        if not path.is_relative_to(self.run.path.resolve()):
            raise ArtifactError(f"artifact {record.id} points outside its run: {record.path}")
        return path

    def _relative(self, path: Path) -> str:
        resolved = path.resolve()
        root = self.run.path.resolve()
        if not resolved.is_relative_to(root):
            raise ArtifactError(
                f"artifact path is outside the run directory: {path}",
                hint="Write artifacts with ArtifactStore.path_for().",
            )
        return resolved.relative_to(root).as_posix()

    # ------------------------------------------------------------------ registration

    def register(
        self,
        path: Path,
        artifact_type: ArtifactType,
        *,
        stage: str,
        parents: Sequence[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Record a file or directory (already inside the run) as an artifact."""
        if not path.exists():
            raise ArtifactError(f"cannot register missing artifact: {path}")
        relative = self._relative(path)
        digest = sha256_path(path)
        record = ArtifactRecord(
            id=ArtifactRecord.make_id(artifact_type, digest),
            type=artifact_type,
            path=relative,
            sha256=digest,
            size_bytes=path_size_bytes(path),
            created_at=utc_now(),
            stage=stage,
            parents=list(parents),
            metadata=dict(metadata or {}),
        )
        existing = self.run.read_manifest().artifacts
        for other in existing.values():
            if other.path == relative:
                if other.sha256 == digest and other.type is artifact_type:
                    return other  # identical content at the same path: idempotent
                raise ArtifactConflictError(
                    f"artifact path {relative} is already registered with different content",
                    hint="Artifacts are immutable; write the new version to a new path.",
                    details={"registered": other.id, "new_sha256": digest},
                )
        for parent in parents:
            if parent not in existing:
                raise ArtifactError(f"unknown parent artifact {parent!r} for {relative}")

        def add(manifest: Any) -> None:
            manifest.artifacts[record.id] = record

        self.run.update_manifest(add)
        log.info("registered artifact %s (%s)", record.id, relative)
        return record

    # ------------------------------------------------------------------ lookup

    def get(self, artifact_id: str) -> ArtifactRecord:
        try:
            return self.run.read_manifest().artifacts[artifact_id]
        except KeyError:
            raise ArtifactError(f"unknown artifact {artifact_id!r}") from None

    def records(self, artifact_type: ArtifactType | None = None) -> list[ArtifactRecord]:
        records = list(self.run.read_manifest().artifacts.values())
        if artifact_type is not None:
            records = [r for r in records if r.type is artifact_type]
        return sorted(records, key=lambda r: r.created_at)

    def latest(self, artifact_type: ArtifactType) -> ArtifactRecord | None:
        found = self.records(artifact_type)
        return found[-1] if found else None

    def require(self, artifact_type: ArtifactType, *, needed_by: str) -> ArtifactRecord:
        record = self.latest(artifact_type)
        if record is None:
            raise ArtifactError(
                f"stage {needed_by!r} needs a {artifact_type.value} artifact, but the run has none",
                hint="Run the stage that produces it first (for example with `trtship run`), or "
                "use a run directory that contains it.",
                details={"missing": artifact_type.value, "run": str(self.run.path)},
            )
        self.verify(record)
        return record

    # ------------------------------------------------------------------ integrity

    def verify(self, record: ArtifactRecord) -> None:
        """Re-hash the artifact and compare with its record."""
        path = self.absolute(record)
        if not path.exists():
            raise ArtifactError(
                f"artifact {record.id} is missing: {record.path}",
                hint="The run directory was modified after the artifact was registered.",
            )
        actual = sha256_path(path)
        if actual != record.sha256:
            raise ArtifactError(
                f"artifact {record.id} is corrupt or was modified: {record.path}",
                details={"expected_sha256": record.sha256, "actual_sha256": actual},
            )

    def verify_all(self) -> list[str]:
        """Problems found across every registered artifact (empty when all are intact)."""
        problems = []
        for record in self.records():
            try:
                self.verify(record)
            except ArtifactError as exc:
                problems.append(exc.message)
        return problems


# --------------------------------------------------------------------------- shared cache


@dataclass(frozen=True)
class CachedArtifact:
    record: ArtifactRecord
    path: Path


class ArtifactCache:
    """Content-verified reuse of stage outputs across runs.

    Layout: ``<cache_dir>/<stage-cache-key>/index.json`` plus one entry per artifact. A stage's
    cache key covers its inputs (artifact hashes), config slice, and tool versions, so a hit means
    the stage would produce equivalent output.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _entry(self, key: str) -> Path:
        if not key or "/" in key or ".." in key:
            raise ArtifactError(f"invalid cache key {key!r}")
        return self.directory / key

    def store(self, key: str, stage: str, produced: Sequence[CachedArtifact]) -> None:
        """Copy a stage's artifacts into the cache under ``key`` (first writer wins)."""
        entry = self._entry(key)
        if (entry / "index.json").is_file():
            return
        staging = self.directory / f".{key}.staging"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        index = []
        for item in produced:
            name = Path(item.record.path).name
            target = staging / name
            if item.path.is_dir():
                shutil.copytree(item.path, target)
            else:
                shutil.copy2(item.path, target)
            index.append({"name": name, "record": item.record.model_dump(mode="json")})
        atomic_write_json(staging / "index.json", {"stage": stage, "artifacts": index})
        try:
            staging.rename(entry)
        except OSError:  # another writer published the same key first; theirs is equivalent
            shutil.rmtree(staging, ignore_errors=True)

    def load(self, key: str) -> list[CachedArtifact] | None:
        """Cached artifacts for ``key``, or ``None`` on a miss or if anything fails verification."""
        entry = self._entry(key)
        index_path = entry / "index.json"
        if not index_path.is_file():
            return None
        try:
            payload = json.loads(index_path.read_text("utf-8"))
            items = [
                CachedArtifact(ArtifactRecord.model_validate(item["record"]), entry / item["name"])
                for item in payload["artifacts"]
            ]
        except (OSError, json.JSONDecodeError, KeyError, ValidationError) as exc:
            log.warning("ignoring unreadable cache entry %s: %s", key[:12], exc)
            return None
        for item in items:
            if not item.path.exists() or sha256_path(item.path) != item.record.sha256:
                log.warning("cache entry %s failed verification; treating as a miss", key[:12])
                return None
        return items

    def adopt(
        self,
        key: str,
        stage: str,
        destination: ArtifactStore,
        *,
        parents: Sequence[str] = (),
    ) -> list[ArtifactRecord] | None:
        """Copy a verified cache hit into ``destination``'s run and register it there."""
        cached = self.load(key)
        if cached is None:
            return None
        adopted = []
        for item in cached:
            target = destination.path_for(item.record.type, Path(item.record.path).name)
            if target.exists():
                return None  # would clobber something already in the run; treat as a miss
            if item.path.is_dir():
                shutil.copytree(item.path, target)
            else:
                shutil.copy2(item.path, target)
            adopted.append(
                destination.register(
                    target,
                    item.record.type,
                    stage=stage,
                    parents=parents,
                    metadata={**item.record.metadata, "reused_from_cache": key},
                )
            )
        return adopted
