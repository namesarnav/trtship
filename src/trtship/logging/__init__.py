"""Structured logging: Rich console output for humans, JSON lines for machines."""

from trtship.logging.setup import (
    attach_log_file,
    configure_logging,
    detach_log_file,
    get_logger,
    log_context,
)

__all__ = [
    "attach_log_file",
    "configure_logging",
    "detach_log_file",
    "get_logger",
    "log_context",
]
