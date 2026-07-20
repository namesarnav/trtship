"""Building a Triton model repository directory."""

from __future__ import annotations

import json
import secrets
import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from trtship.config import TritonConfig
from trtship.errors import ArtifactConflictError, TritonError
from trtship.logging import get_logger
from trtship.tensorrt import EngineInfo
from trtship.triton.config import derive_max_batch_size, render_config_pbtxt
from trtship.utils.fs import atomic_write_text
from trtship.utils.hashing import sha256_directory, sha256_file

log = get_logger(__name__)


class RepositoryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str
    model_name: str
    version: int
    config_path: str
    plan_path: str
    plan_sha256: str
    max_batch_size: int
    repository_sha256: str


def build_repository(
    destination: Path,
    model_name: str,
    plan: Path,
    info: EngineInfo,
    settings: TritonConfig,
) -> RepositoryResult:
    """Create ``destination`` (which must not exist) as a Triton repository holding one model::

        destination/
          <model_name>/
            config.pbtxt
            <version>/model.plan

    The repository is assembled next to ``destination`` and moved into place, so a failure leaves no
    partial repository behind.
    """
    if destination.exists():
        raise ArtifactConflictError(
            f"refusing to overwrite existing model repository: {destination}",
            hint="Repositories are immutable artifacts. Choose a new path.",
        )
    if not plan.is_file():
        raise TritonError(f"engine plan not found: {plan}")
    config_text = render_config_pbtxt(model_name, info, settings)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{secrets.token_hex(4)}.tmp")
    try:
        model_dir = staging / model_name
        version_dir = model_dir / str(settings.model_version)
        version_dir.mkdir(parents=True)
        shutil.copy2(plan, version_dir / "model.plan")
        atomic_write_text(model_dir / "config.pbtxt", config_text)
        digest = sha256_directory(staging)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    log.info("built model repository %s (model %s)", destination, model_name)
    final_dir = destination / model_name
    return RepositoryResult(
        repository=str(destination),
        model_name=model_name,
        version=settings.model_version,
        config_path=str(final_dir / "config.pbtxt"),
        plan_path=str(final_dir / str(settings.model_version) / "model.plan"),
        plan_sha256=sha256_file(final_dir / str(settings.model_version) / "model.plan"),
        max_batch_size=derive_max_batch_size(info, settings),
        repository_sha256=digest,
    )


def load_engine_info(path: Path) -> EngineInfo:
    """Read engine metadata from JSON: an ``EngineInfo``, or a ``BuildResult`` holding one."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TritonError(f"cannot read engine metadata {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TritonError(f"engine metadata {path} is not valid JSON: {exc}") from exc
    if isinstance(payload, dict) and isinstance(payload.get("engine"), dict):
        payload = payload["engine"]
    try:
        return EngineInfo.model_validate(payload)
    except ValidationError as exc:
        raise TritonError(f"engine metadata {path} is not valid: {exc}") from exc
