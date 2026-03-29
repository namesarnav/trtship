"""Filesystem helpers: atomic writes and size accounting."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from trtship.errors import ArtifactConflictError


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a same-directory temp file and an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def path_size_bytes(path: Path) -> int:
    if path.is_dir():
        return sum(member.stat().st_size for member in path.rglob("*") if member.is_file())
    return path.stat().st_size


def publish_new(tmp: Path, dest: Path) -> None:
    """Atomically move ``tmp`` to ``dest``, refusing to replace an existing file.

    ``tmp`` must be on the same filesystem as ``dest`` (write it next to the destination). The
    hard link fails atomically if ``dest`` exists, so two writers can never silently clobber each
    other. ``tmp`` is removed either way.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(tmp, dest)
    except FileExistsError:
        raise ArtifactConflictError(
            f"refusing to overwrite existing artifact: {dest}",
            hint="Artifacts are immutable. Choose a new output path or remove the file yourself.",
        ) from None
    finally:
        tmp.unlink(missing_ok=True)
