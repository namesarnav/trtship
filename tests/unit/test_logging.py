from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.console import Console

from trtship.logging import (
    attach_log_file,
    configure_logging,
    detach_log_file,
    get_logger,
    log_context,
)
from trtship.logging.setup import _context


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    logger = logging.getLogger("trtship")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_file_handler_writes_json_lines_with_context(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "trtship.jsonl"
    configure_logging(level="INFO", log_file=log_file)
    log = get_logger("trtship.test")
    with log_context(run_id="r1", stage="export"):
        log.info("exporting %s", "model", extra={"opset": 17})
    log.info("outside")
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert lines[0]["message"] == "exporting model"
    assert lines[0]["run_id"] == "r1"
    assert lines[0]["stage"] == "export"
    assert lines[0]["opset"] == 17
    assert lines[0]["level"] == "INFO"
    assert "run_id" not in lines[1]


def test_context_nests_and_restores() -> None:
    with log_context(run_id="a"):
        with log_context(stage="build"):
            assert _context.get() == {"run_id": "a", "stage": "build"}
        assert _context.get() == {"run_id": "a"}
    assert _context.get() == {}


def test_level_filtering(tmp_path: Path) -> None:
    log_file = tmp_path / "l.jsonl"
    configure_logging(level="WARNING", log_file=log_file)
    log = get_logger("x")
    log.info("hidden")
    log.warning("shown")
    assert [json.loads(line)["message"] for line in log_file.read_text().splitlines()] == ["shown"]


def test_exceptions_are_serialized(tmp_path: Path) -> None:
    log_file = tmp_path / "l.jsonl"
    configure_logging(level="INFO", log_file=log_file)
    try:
        raise RuntimeError("kaboom")
    except RuntimeError:
        get_logger("x").exception("failed")
    record = json.loads(log_file.read_text().splitlines()[0])
    assert "RuntimeError: kaboom" in record["exception"]


def test_reconfiguring_replaces_handlers(tmp_path: Path) -> None:
    configure_logging(level="INFO", log_file=tmp_path / "a.jsonl")
    configure_logging(level="INFO", log_file=tmp_path / "b.jsonl")
    assert len(logging.getLogger("trtship").handlers) == 2  # console + one file


def test_invalid_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging(level="LOUD")


def test_console_output_goes_to_stderr_not_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(level="INFO", json_logs=True)
    get_logger("x").info("hello")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err.strip())["message"] == "hello"


def test_get_logger_namespacing() -> None:
    assert get_logger("foo").name == "trtship.foo"
    assert get_logger("trtship.bar").name == "trtship.bar"


def test_rich_console_is_used_by_default() -> None:
    buffer = io.StringIO()
    configure_logging(level="INFO", console=Console(file=buffer, width=100))
    get_logger("x").info("plain message")
    assert "plain message" in buffer.getvalue()


def test_attach_log_file_records_info_even_when_the_console_is_quiet(tmp_path: Path) -> None:
    configure_logging(level="WARNING", json_logs=True)
    log_file = tmp_path / "logs" / "run.jsonl"
    handler = attach_log_file(log_file)
    try:
        with log_context(run_id="r9", stage="export"):
            get_logger("t").info("stage detail")
            get_logger("t").debug("too verbose")
    finally:
        detach_log_file(handler)
    get_logger("t").info("after detach")
    records = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert [r["message"] for r in records] == ["stage detail"]
    assert records[0]["run_id"] == "r9"
    assert records[0]["stage"] == "export"
