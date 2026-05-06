from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trtship.artifacts import (
    ArtifactCache,
    ArtifactRecord,
    ArtifactStore,
    ArtifactType,
    CachedArtifact,
    RunDirectory,
)
from trtship.errors import ArtifactConflictError, ArtifactError
from trtship.utils.hashing import sha256_directory, sha256_file

MakeRun = Callable[..., RunDirectory]


@pytest.fixture
def store(make_run: MakeRun) -> ArtifactStore:
    return ArtifactStore(make_run())


def write(store: ArtifactStore, artifact_type: ArtifactType, name: str, data: bytes) -> Path:
    path = store.path_for(artifact_type, name)
    path.write_bytes(data)
    return path


# --------------------------------------------------------------------------- paths


@pytest.mark.parametrize(
    ("artifact_type", "directory"),
    [
        (ArtifactType.ONNX, "artifacts"),
        (ArtifactType.ONNX_OPTIMIZED, "artifacts"),
        (ArtifactType.ENGINE, "artifacts"),
        (ArtifactType.CALIBRATION_CACHE, "artifacts"),
        (ArtifactType.TRITON_REPOSITORY, "artifacts"),
        (ArtifactType.VALIDATION_REPORT, "validation"),
        (ArtifactType.BENCHMARK_REPORT, "benchmarks"),
        (ArtifactType.MODEL_REPORT, "reports"),
        (ArtifactType.REPORT, "reports"),
    ],
)
def test_path_for_places_each_type_in_its_directory(
    store: ArtifactStore, artifact_type: ArtifactType, directory: str
) -> None:
    assert store.path_for(artifact_type, "x.bin") == store.run.path / directory / "x.bin"


@pytest.mark.parametrize("bad", ["", ".", "..", "../x", "a/b", "a\\b", ".hidden"])
def test_path_for_rejects_unsafe_names(store: ArtifactStore, bad: str) -> None:
    with pytest.raises(ArtifactError, match="invalid artifact file name"):
        store.path_for(ArtifactType.ONNX, bad)


# --------------------------------------------------------------------------- registration


def test_register_a_file(store: ArtifactStore) -> None:
    path = write(store, ArtifactType.ONNX, "m.onnx", b"onnx bytes")
    record = store.register(path, ArtifactType.ONNX, stage="export", metadata={"opset": 17})
    digest = sha256_file(path)
    assert record.id == f"onnx-{digest[:12]}"
    assert record.path == "artifacts/m.onnx"  # relative, POSIX
    assert record.sha256 == digest
    assert record.size_bytes == len(b"onnx bytes")
    assert record.stage == "export"
    assert record.metadata == {"opset": 17}
    assert record.parents == []
    assert record.created_at.tzinfo is not None
    assert store.run.read_manifest().artifacts[record.id] == record
    assert store.get(record.id) == record
    assert store.absolute(record) == path.resolve()


def test_register_a_directory_artifact(store: ArtifactStore) -> None:
    repo = store.path_for(ArtifactType.TRITON_REPOSITORY, "model_repository")
    (repo / "m" / "1").mkdir(parents=True)
    (repo / "m" / "config.pbtxt").write_text("name: 'm'")
    (repo / "m" / "1" / "model.plan").write_bytes(b"plan")
    record = store.register(repo, ArtifactType.TRITON_REPOSITORY, stage="package")
    assert record.sha256 == sha256_directory(repo)
    assert record.size_bytes == len("name: 'm'") + len(b"plan")


def test_registering_identical_content_at_the_same_path_is_idempotent(
    store: ArtifactStore,
) -> None:
    path = write(store, ArtifactType.ONNX, "m.onnx", b"same")
    first = store.register(path, ArtifactType.ONNX, stage="export")
    second = store.register(path, ArtifactType.ONNX, stage="export")
    assert first == second
    assert len(store.records()) == 1


def test_changed_content_at_a_registered_path_is_a_conflict(store: ArtifactStore) -> None:
    path = write(store, ArtifactType.ONNX, "m.onnx", b"v1")
    store.register(path, ArtifactType.ONNX, stage="export")
    path.write_bytes(b"v2")
    with pytest.raises(ArtifactConflictError, match="already registered with different content"):
        store.register(path, ArtifactType.ONNX, stage="export")
    assert len(store.records()) == 1


def test_register_rejects_missing_paths_and_paths_outside_the_run(
    store: ArtifactStore, tmp_path: Path
) -> None:
    with pytest.raises(ArtifactError, match="missing artifact"):
        store.register(store.run.path / "artifacts" / "nope.onnx", ArtifactType.ONNX, stage="s")
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"x")
    with pytest.raises(ArtifactError, match="outside the run directory"):
        store.register(outside, ArtifactType.ONNX, stage="s")


def test_register_rejects_symlinks_that_escape_the_run(
    store: ArtifactStore, tmp_path: Path
) -> None:
    target = tmp_path / "secret.bin"
    target.write_bytes(b"secret")
    link = store.run.path / "artifacts" / "link.onnx"
    link.parent.mkdir(exist_ok=True)
    os.symlink(target, link)
    with pytest.raises(ArtifactError, match="outside the run directory"):
        store.register(link, ArtifactType.ONNX, stage="s")


def test_parents_must_exist_and_are_recorded(store: ArtifactStore) -> None:
    onnx = store.register(
        write(store, ArtifactType.ONNX, "m.onnx", b"a"), ArtifactType.ONNX, stage="export"
    )
    child = store.register(
        write(store, ArtifactType.ONNX_OPTIMIZED, "o.onnx", b"b"),
        ArtifactType.ONNX_OPTIMIZED,
        stage="optimize",
        parents=[onnx.id],
    )
    assert child.parents == [onnx.id]
    with pytest.raises(ArtifactError, match="unknown parent"):
        store.register(
            write(store, ArtifactType.ENGINE, "e.plan", b"c"),
            ArtifactType.ENGINE,
            stage="build",
            parents=["onnx-ffffffffffff"],
        )


# --------------------------------------------------------------------------- lookup


def test_lookup_by_type_and_latest(store: ArtifactStore) -> None:
    first = store.register(
        write(store, ArtifactType.ONNX, "a.onnx", b"1"), ArtifactType.ONNX, stage="export"
    )
    second = store.register(
        write(store, ArtifactType.ONNX, "b.onnx", b"2"), ArtifactType.ONNX, stage="export"
    )
    other = store.register(
        write(store, ArtifactType.ENGINE, "e.plan", b"3"), ArtifactType.ENGINE, stage="build"
    )
    assert [r.id for r in store.records(ArtifactType.ONNX)] == [first.id, second.id]
    assert store.latest(ArtifactType.ONNX) == second
    assert store.latest(ArtifactType.ENGINE) == other
    assert store.latest(ArtifactType.REPORT) is None
    assert len(store.records()) == 3
    with pytest.raises(ArtifactError, match="unknown artifact"):
        store.get("nope")


def test_require_explains_what_is_missing(store: ArtifactStore) -> None:
    with pytest.raises(ArtifactError, match="needs a engine artifact") as info:
        store.require(ArtifactType.ENGINE, needed_by="benchmark")
    assert "Run the stage that produces it" in (info.value.hint or "")
    assert info.value.details["missing"] == "engine"


def test_require_returns_a_verified_record(store: ArtifactStore) -> None:
    record = store.register(
        write(store, ArtifactType.ONNX, "m.onnx", b"ok"), ArtifactType.ONNX, stage="export"
    )
    assert store.require(ArtifactType.ONNX, needed_by="validate") == record


# --------------------------------------------------------------------------- integrity


def test_verify_detects_modification_and_deletion(store: ArtifactStore) -> None:
    path = write(store, ArtifactType.ONNX, "m.onnx", b"original")
    record = store.register(path, ArtifactType.ONNX, stage="export")
    store.verify(record)  # intact

    path.write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="corrupt or was modified") as info:
        store.verify(record)
    assert info.value.details["expected_sha256"] == record.sha256
    with pytest.raises(ArtifactError, match="corrupt"):
        store.require(ArtifactType.ONNX, needed_by="validate")

    path.unlink()
    with pytest.raises(ArtifactError, match="is missing"):
        store.verify(record)


def test_verify_all_reports_every_problem(store: ArtifactStore) -> None:
    good = write(store, ArtifactType.ONNX, "good.onnx", b"g")
    bad = write(store, ArtifactType.ENGINE, "bad.plan", b"b")
    store.register(good, ArtifactType.ONNX, stage="s")
    store.register(bad, ArtifactType.ENGINE, stage="s")
    assert store.verify_all() == []
    bad.write_bytes(b"changed")
    problems = store.verify_all()
    assert len(problems) == 1
    assert "bad.plan" in problems[0]


def test_absolute_rejects_records_that_point_outside_the_run(store: ArtifactStore) -> None:
    forged = ArtifactRecord(
        id="onnx-000000000000",
        type=ArtifactType.ONNX,
        path="../../etc/passwd",
        sha256="0" * 64,
        size_bytes=0,
        created_at=datetime.now(UTC),
        stage="x",
    )
    with pytest.raises(ArtifactError, match="outside its run"):
        store.absolute(forged)


# --------------------------------------------------------------------------- shared cache


def produce(store: ArtifactStore, name: str, data: bytes) -> CachedArtifact:
    path = write(store, ArtifactType.ONNX, name, data)
    record = store.register(path, ArtifactType.ONNX, stage="export", metadata={"opset": 17})
    return CachedArtifact(record, path)


def test_cache_round_trip_into_another_run(make_run: MakeRun, tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path / "cache")
    source = ArtifactStore(make_run("a"))
    item = produce(source, "m.onnx", b"model bytes")
    cache.store("k" * 64, "export", [item])
    assert not list((tmp_path / "cache").glob(".*staging"))  # no staging leftovers

    other = ArtifactStore(make_run("b"))
    parent = other.register(
        write(other, ArtifactType.REPORT, "p.json", b"{}"), ArtifactType.REPORT, stage="inspect"
    )
    adopted = cache.adopt("k" * 64, "export", other, parents=[parent.id])
    assert adopted is not None
    (record,) = adopted
    assert record.sha256 == item.record.sha256
    assert record.metadata == {"opset": 17, "reused_from_cache": "k" * 64}
    assert record.parents == [parent.id]
    assert other.absolute(record).read_bytes() == b"model bytes"
    assert other.absolute(record) != source.absolute(item.record)  # a copy, not a shared file


def test_cache_miss_and_invalid_keys(tmp_path: Path, make_run: MakeRun) -> None:
    cache = ArtifactCache(tmp_path / "cache")
    assert cache.load("a" * 64) is None
    assert cache.adopt("a" * 64, "export", ArtifactStore(make_run())) is None
    for bad in ("", "../x", "a/b"):
        with pytest.raises(ArtifactError, match="invalid cache key"):
            cache.load(bad)


def test_first_writer_wins(make_run: MakeRun, tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path / "cache")
    first = produce(ArtifactStore(make_run("a")), "m.onnx", b"first")
    second = produce(ArtifactStore(make_run("b")), "m.onnx", b"second")
    cache.store("k" * 64, "export", [first])
    cache.store("k" * 64, "export", [second])
    loaded = cache.load("k" * 64)
    assert loaded is not None
    assert loaded[0].record.sha256 == first.record.sha256


def test_a_corrupted_cache_entry_is_a_miss(make_run: MakeRun, tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path / "cache")
    cache.store("k" * 64, "export", [produce(ArtifactStore(make_run("a")), "m.onnx", b"good")])
    (tmp_path / "cache" / ("k" * 64) / "m.onnx").write_bytes(b"bit rot")
    assert cache.load("k" * 64) is None
    assert cache.adopt("k" * 64, "export", ArtifactStore(make_run("b"))) is None


def test_an_unreadable_index_is_a_miss(make_run: MakeRun, tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path / "cache")
    cache.store("k" * 64, "export", [produce(ArtifactStore(make_run("a")), "m.onnx", b"x")])
    (tmp_path / "cache" / ("k" * 64) / "index.json").write_text("{not json")
    assert cache.load("k" * 64) is None


def test_adopt_never_clobbers_an_existing_file(make_run: MakeRun, tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path / "cache")
    cache.store("k" * 64, "export", [produce(ArtifactStore(make_run("a")), "m.onnx", b"cached")])
    destination = ArtifactStore(make_run("b"))
    existing = write(destination, ArtifactType.ONNX, "m.onnx", b"already here")
    assert cache.adopt("k" * 64, "export", destination) is None
    assert existing.read_bytes() == b"already here"


def test_directory_artifacts_round_trip_through_the_cache(
    make_run: MakeRun, tmp_path: Path
) -> None:
    source = ArtifactStore(make_run("a"))
    repo = source.path_for(ArtifactType.TRITON_REPOSITORY, "model_repository")
    (repo / "m" / "1").mkdir(parents=True)
    (repo / "m" / "config.pbtxt").write_text("cfg")
    (repo / "m" / "1" / "model.plan").write_bytes(b"plan")
    record = source.register(repo, ArtifactType.TRITON_REPOSITORY, stage="package")

    cache = ArtifactCache(tmp_path / "cache")
    cache.store("d" * 64, "package", [CachedArtifact(record, repo)])
    destination = ArtifactStore(make_run("b"))
    adopted = cache.adopt("d" * 64, "package", destination)
    assert adopted is not None
    assert sha256_directory(destination.absolute(adopted[0])) == record.sha256
