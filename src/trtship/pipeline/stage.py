"""The stage contract and the context a stage runs in."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

from trtship.artifacts import ArtifactRecord, ArtifactStore, ArtifactType, RunDirectory
from trtship.config import TrtshipConfig
from trtship.errors import ArtifactConflictError
from trtship.models import LoadedModel, ModelSignature, infer_signature, load_model
from trtship.utils.env import EnvironmentReport
from trtship.utils.hashing import sha256_json, sha256_path


class Resources:
    """Expensive objects shared by the stages of one pipeline run, created on first use."""

    def __init__(self, config: TrtshipConfig, run: RunDirectory) -> None:
        self._config = config
        self._run = run

    @cached_property
    def model(self) -> LoadedModel:
        loaded = load_model(self._config.model)
        if self._run.read_manifest().model_sha256 != loaded.weights_sha256:

            def record(manifest: Any) -> None:
                manifest.model_sha256 = loaded.weights_sha256

            self._run.update_manifest(record)
        return loaded

    @cached_property
    def signature(self) -> ModelSignature:
        return infer_signature(
            self.model,
            self._config.model,
            self._config.tensorrt.profiles,
            seed=self._config.seed,
        )


@dataclass
class StageContext:
    config: TrtshipConfig
    run: RunDirectory
    store: ArtifactStore
    environment: EnvironmentReport
    resources: Resources
    stage: str
    scratch: Path
    inputs_used: list[ArtifactRecord] = field(default_factory=list)
    published: list[ArtifactRecord] = field(default_factory=list)

    def input(
        self, artifact_type: ArtifactType, *, optional: bool = False
    ) -> ArtifactRecord | None:
        """The newest verified artifact of ``artifact_type``; recorded as a parent of outputs."""
        record = (
            self.store.latest(artifact_type)
            if optional
            else self.store.require(artifact_type, needed_by=self.stage)
        )
        if record is None:
            return None
        self.store.verify(record)
        if record.id not in {r.id for r in self.inputs_used}:
            self.inputs_used.append(record)
        return record

    def input_path(self, artifact_type: ArtifactType) -> Path:
        record = self.input(artifact_type)
        assert record is not None  # not optional
        return self.store.absolute(record)

    def publish(
        self,
        path: Path,
        artifact_type: ArtifactType,
        filename: str | None = None,
        *,
        parents: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Move ``path`` (written in the scratch directory) into the run and register it.

        Never overwrites: if the destination exists with different content the artifact is
        published as ``name.2.ext``, ``name.3.ext``, ...; identical content is reused as is.
        """
        name = filename or path.name
        parent_ids = parents if parents is not None else [r.id for r in self.inputs_used]
        destination = self.store.path_for(artifact_type, name)
        digest = sha256_path(path)
        counter = 2
        while destination.exists():
            if sha256_path(destination) == digest:
                _discard(path)
                break
            stem, dot, suffix = name.partition(".")
            destination = self.store.path_for(artifact_type, f"{stem}.{counter}{dot}{suffix}")
            counter += 1
            if counter > 1000:
                raise ArtifactConflictError(f"too many versions of artifact {name}")
        else:
            shutil.move(str(path), destination)
        record = self.store.register(
            destination, artifact_type, stage=self.stage, parents=parent_ids, metadata=metadata
        )
        self.published.append(record)
        return record


def _discard(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


@dataclass(frozen=True)
class StageResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class Stage(ABC):
    """One step of the pipeline.

    Order is derived from artifact types: a stage runs after the stages that ``produces`` what it
    ``requires`` or ``uses``. ``requires_capabilities`` are checked for every selected stage before
    any stage starts.
    """

    name: ClassVar[str]
    requires: ClassVar[tuple[ArtifactType, ...]] = ()
    uses: ClassVar[tuple[ArtifactType, ...]] = ()  # optional inputs
    produces: ClassVar[tuple[ArtifactType, ...]] = ()
    requires_capabilities: ClassVar[tuple[str, ...]] = ()
    depends_on_model: ClassVar[bool] = False  # the weights hash is part of the cache key
    cacheable: ClassVar[bool] = True
    version: ClassVar[int] = 1  # bump when a stage's behavior changes

    def skip_reason(self, config: TrtshipConfig) -> str | None:
        """A reason this stage does not apply to ``config``, or ``None`` if it does."""
        return None

    @abstractmethod
    def config_slice(self, config: TrtshipConfig) -> Any:
        """The JSON-serializable part of the config this stage's output depends on."""

    @abstractmethod
    def run(self, ctx: StageContext) -> StageResult:
        """Do the work: write files under ``ctx.scratch``, then ``ctx.publish`` them."""

    def cache_key(self, ctx: StageContext, tools: dict[str, str | None]) -> str:
        inputs = {}
        for artifact_type in (*self.requires, *self.uses):
            record = ctx.store.latest(artifact_type)
            if record is not None:
                inputs[artifact_type.value] = record.sha256
        payload = {
            "stage": self.name,
            "version": self.version,
            "config": self.config_slice(ctx.config),
            "seed": ctx.config.seed,
            "inputs": inputs,
            "weights": ctx.resources.model.weights_sha256 if self.depends_on_model else None,
            "tools": tools,
        }
        return sha256_json(payload)
