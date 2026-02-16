"""Content hashing used for artifact identity, cache keys, and reproducibility records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_CHUNK = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, ASCII-safe."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("ascii"))


def sha256_directory(path: Path) -> str:
    """Hash a directory as the hash of its sorted ``relative-path -> file-hash`` listing.

    Empty directories and file modes are deliberately ignored; only names and contents count.
    """
    if not path.is_dir():
        raise NotADirectoryError(str(path))
    listing: Mapping[str, str] = {
        member.relative_to(path).as_posix(): sha256_file(member)
        for member in sorted(path.rglob("*"))
        if member.is_file()
    }
    return sha256_json(dict(listing))


def sha256_path(path: Path) -> str:
    """Hash a file or a directory."""
    return sha256_directory(path) if path.is_dir() else sha256_file(path)
