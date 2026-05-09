"""Run directories: one self-describing directory per pipeline run.

Layout::

    runs/<YYYY-MM-DD>_<token>/
      config.yaml  manifest.json  environment.json
      artifacts/  validation/  benchmarks/  reports/  logs/
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from trtship import __version__
from trtship.artifacts.records import RunManifest, StageRecord
from trtship.config import TrtshipConfig, config_hash, dump_config_yaml
from trtship.errors import ArtifactError, ConfigError
from trtship.utils.env import EnvironmentReport
from trtship.utils.fs import atomic_write_json, atomic_write_text
from trtship.utils.timeutil import utc_now

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SUBDIRS = ("artifacts", "validation", "benchmarks", "reports", "logs")


def new_run_id(now: datetime | None = None) -> str:
    """``YYYY-MM-DD_<6 hex>`` using the real wall-clock date."""
    moment = now or utc_now()
    return f"{moment:%Y-%m-%d}_{secrets.token_hex(3)}"


def validate_run_id(run_id: str) -> str:
    """Reject ids that could escape the run root or collide with special names."""
    if not _RUN_ID.match(run_id) or run_id in {".", ".."} or ".." in run_id:
        raise ConfigError(
            f"invalid run id {run_id!r}",
            hint="Run ids may contain letters, digits, '_', '.', '-' and must not contain '..'.",
        )
    return run_id


class RunDirectory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.run_id = path.name

    # ------------------------------------------------------------------ construction

    @classmethod
    def create(
        cls,
        root: Path,
        config: TrtshipConfig,
        environment: EnvironmentReport,
        *,
        run_id: str | None = None,
    ) -> RunDirectory:
        """Create a new run directory and write its initial metadata files."""
        run_id = validate_run_id(run_id) if run_id is not None else new_run_id()
        path = root / run_id
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise ArtifactError(
                f"run directory already exists: {path}",
                hint="Choose a different --run-id or resume the existing run.",
            ) from None
        for name in SUBDIRS:
            (path / name).mkdir()
        run = cls(path)
        atomic_write_text(path / "config.yaml", dump_config_yaml(config))
        atomic_write_json(path / "environment.json", environment.model_dump(mode="json"))
        now = utc_now()
        run.write_manifest(
            RunManifest(
                run_id=run_id,
                created_at=now,
                updated_at=now,
                trtship_version=__version__,
                config_sha256=config_hash(config),
                seed=config.seed,
            )
        )
        return run

    @classmethod
    def open(cls, path: Path) -> RunDirectory:
        if not (path / "manifest.json").is_file():
            raise ArtifactError(
                f"not a trtship run directory (no manifest.json): {path}",
                hint="Pass a directory created by `trtship run`, e.g. runs/2026-01-01_ab12cd.",
            )
        return cls(path)

    @classmethod
    def latest(cls, root: Path) -> RunDirectory:
        """Most recently created run under ``root`` (by manifest creation time)."""
        candidates = (
            [
                (RunDirectory(p).read_manifest().created_at, p)
                for p in root.iterdir()
                if p.is_dir() and (p / "manifest.json").is_file()
            ]
            if root.is_dir()
            else []
        )
        if not candidates:
            raise ArtifactError(f"no runs found under {root}")
        return cls(max(candidates)[1])

    # ------------------------------------------------------------------ layout

    @property
    def config_path(self) -> Path:
        return self.path / "config.yaml"

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    @property
    def environment_path(self) -> Path:
        return self.path / "environment.json"

    @property
    def artifacts_dir(self) -> Path:
        return self.path / "artifacts"

    @property
    def validation_dir(self) -> Path:
        return self.path / "validation"

    @property
    def benchmarks_dir(self) -> Path:
        return self.path / "benchmarks"

    @property
    def reports_dir(self) -> Path:
        return self.path / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.path / "logs"

    # ------------------------------------------------------------------ manifest

    def read_manifest(self) -> RunManifest:
        try:
            return RunManifest.model_validate(json.loads(self.manifest_path.read_text("utf-8")))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise ArtifactError(
                f"corrupt or unreadable manifest: {self.manifest_path}", details={"cause": str(exc)}
            ) from exc

    def write_manifest(self, manifest: RunManifest) -> None:
        atomic_write_json(self.manifest_path, manifest.model_dump(mode="json"))

    def update_manifest(self, mutate: Callable[[RunManifest], None]) -> RunManifest:
        """Read-modify-write the manifest. Single-writer: a run is driven by one process."""
        manifest = self.read_manifest()
        mutate(manifest)
        manifest.updated_at = utc_now()
        self.write_manifest(manifest)
        return manifest

    def update_stage(self, name: str, **fields: object) -> StageRecord:
        def apply(manifest: RunManifest) -> None:
            record = manifest.stages.get(name, StageRecord())
            manifest.stages[name] = record.model_copy(update=fields)

        return self.update_manifest(apply).stages[name]
