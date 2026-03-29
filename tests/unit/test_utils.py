from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from trtship.errors import ArtifactConflictError, CommandError
from trtship.utils.fs import atomic_write_bytes, atomic_write_json, path_size_bytes, publish_new
from trtship.utils.hashing import (
    canonical_json,
    sha256_bytes,
    sha256_directory,
    sha256_file,
    sha256_json,
    sha256_path,
)
from trtship.utils.subprocess import run_command


def test_sha256_known_vector() -> None:
    assert (
        sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_sha256_file_matches_bytes(tmp_path: Path) -> None:
    payload = os.urandom(3 * 1024 * 1024 + 17)  # spans several read chunks
    path = tmp_path / "blob"
    path.write_bytes(payload)
    assert sha256_file(path) == sha256_bytes(payload)


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json({"b": 1, "a": [2, 3]}) == canonical_json({"a": [2, 3], "b": 1})
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})
    assert sha256_json({"a": 1}) != sha256_json({"a": 2})


def test_directory_hash_depends_on_names_and_contents(tmp_path: Path) -> None:
    a = tmp_path / "a"
    (a / "sub").mkdir(parents=True)
    (a / "x.txt").write_text("one")
    (a / "sub" / "y.txt").write_text("two")
    base = sha256_directory(a)

    b = tmp_path / "b"
    (b / "sub").mkdir(parents=True)
    (b / "x.txt").write_text("one")
    (b / "sub" / "y.txt").write_text("two")
    assert sha256_directory(b) == base  # location-independent

    (b / "sub" / "y.txt").write_text("changed")
    assert sha256_directory(b) != base
    (b / "sub" / "y.txt").write_text("two")
    (b / "sub" / "y.txt").rename(b / "sub" / "z.txt")
    assert sha256_directory(b) != base  # renames count


def test_sha256_path_dispatches(tmp_path: Path) -> None:
    f = tmp_path / "f"
    f.write_bytes(b"data")
    assert sha256_path(f) == sha256_file(f)
    assert sha256_path(tmp_path) == sha256_directory(tmp_path)
    with pytest.raises(NotADirectoryError):
        sha256_directory(f)


def test_atomic_write_creates_parents_and_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "dir" / "file.bin"
    atomic_write_bytes(target, b"hello")
    assert target.read_bytes() == b"hello"
    assert [p.name for p in target.parent.iterdir()] == ["file.bin"]


def test_atomic_write_failure_keeps_original_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "file.txt"
    atomic_write_bytes(target, b"original")

    def boom(self: Path, *_: object) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_bytes(target, b"new")
    monkeypatch.undo()
    assert target.read_bytes() == b"original"
    assert [p.name for p in tmp_path.iterdir()] == ["file.txt"]


def test_atomic_write_json_is_sorted_and_newline_terminated(tmp_path: Path) -> None:
    target = tmp_path / "x.json"
    atomic_write_json(target, {"b": 1, "a": 2})
    text = target.read_text()
    assert text.endswith("\n")
    assert text.index('"a"') < text.index('"b"')


def test_path_size_bytes(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"12345")
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "b").write_bytes(b"123")
    assert path_size_bytes(tmp_path / "a") == 5
    assert path_size_bytes(tmp_path) == 8


def test_run_command_captures_output() -> None:
    result = run_command(
        [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"]
    )
    assert result.ok
    assert result.stdout.strip() == "out"
    assert result.output == "out\nerr"


def test_run_command_nonzero_only_raises_with_check() -> None:
    argv = [sys.executable, "-c", "import sys; sys.exit(3)"]
    assert run_command(argv).returncode == 3
    with pytest.raises(CommandError) as info:
        run_command(argv, check=True)
    assert info.value.details["returncode"] == 3


def test_run_command_missing_executable_and_timeout() -> None:
    with pytest.raises(CommandError, match="not found"):
        run_command(["definitely-not-a-real-binary-xyz"])
    with pytest.raises(CommandError, match="timed out"):
        run_command([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.2)
    with pytest.raises(CommandError, match="empty"):
        run_command([])


def test_run_command_does_not_use_a_shell() -> None:
    # A shell would expand this; with argv semantics it is a literal argument.
    result = run_command(
        [sys.executable, "-c", "import sys; print(sys.argv[1])", "$(echo pwned); ls"]
    )
    assert result.stdout.strip() == "$(echo pwned); ls"


def test_publish_new_moves_the_file_and_refuses_to_clobber(tmp_path: Path) -> None:
    tmp = tmp_path / ".x.tmp"
    tmp.write_bytes(b"new")
    dest = tmp_path / "sub" / "x.bin"
    publish_new(tmp, dest)
    assert dest.read_bytes() == b"new"
    assert not tmp.exists()

    again = tmp_path / ".y.tmp"
    again.write_bytes(b"other")
    with pytest.raises(ArtifactConflictError, match="refusing to overwrite"):
        publish_new(again, dest)
    assert dest.read_bytes() == b"new"  # untouched
    assert not again.exists()  # temp is cleaned up either way
