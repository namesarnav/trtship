"""Logging configuration.

Records carry ``run_id`` and ``stage`` from a context variable so that every line emitted while a
stage runs is attributable without threading identifiers through call signatures.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from trtship.utils.timeutil import isoformat, utc_now

ROOT_LOGGER = "trtship"

_context: contextvars.ContextVar[Mapping[str, str]] = contextvars.ContextVar(
    "trtship_log_context", default=MappingProxyType({})
)

# Attributes present on every LogRecord; anything else was passed through ``extra=``.
_STANDARD_ATTRS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}


@contextmanager
def log_context(**fields: str) -> Iterator[None]:
    """Attach ``fields`` (e.g. ``run_id``, ``stage``) to every record emitted inside the block."""
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in _context.get().items():
            setattr(record, key, value)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": isoformat(utc_now()),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


def configure_logging(
    *,
    level: str = "INFO",
    json_logs: bool = False,
    log_file: Path | None = None,
    console: Console | None = None,
) -> logging.Logger:
    """Configure the ``trtship`` logger tree. Safe to call repeatedly; handlers are replaced.

    Console output goes to stderr so that stdout stays reserved for command results
    (including ``--json`` output).
    """
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(f"unknown log level: {level!r}")

    logger = logging.getLogger(ROOT_LOGGER)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    # The logger passes INFO and above so run log files can capture it; each handler filters to its
    # own level (the console honors --log-level).
    logger.setLevel(min(numeric, logging.INFO))
    logger.propagate = False

    console_handler: logging.Handler
    if json_logs:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler = RichHandler(
            console=console or Console(stderr=True),
            show_path=False,
            rich_tracebacks=False,
            markup=False,
        )
    console_handler.setLevel(numeric)
    console_handler.addFilter(_ContextFilter())
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric)
        file_handler.setFormatter(JsonFormatter())
        file_handler.addFilter(_ContextFilter())
        logger.addHandler(file_handler)
    return logger


def attach_log_file(path: Path, *, level: int = logging.INFO) -> logging.Handler:
    """Also write JSON-lines logs to ``path`` (e.g. a run's log file). Pair with
    :func:`detach_log_file`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(_ContextFilter())
    logger = logging.getLogger(ROOT_LOGGER)
    if logger.level > level or logger.level == logging.NOTSET:
        logger.setLevel(level)
    logger.addHandler(handler)
    return handler


def detach_log_file(handler: logging.Handler) -> None:
    logging.getLogger(ROOT_LOGGER).removeHandler(handler)
    handler.close()


def get_logger(name: str) -> logging.Logger:
    """Logger under the ``trtship`` tree, e.g. ``get_logger(__name__)``."""
    return logging.getLogger(name if name.startswith(ROOT_LOGGER) else f"{ROOT_LOGGER}.{name}")
