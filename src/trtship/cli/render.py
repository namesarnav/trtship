"""Shared terminal rendering. stdout carries results; stderr carries diagnostics."""

from __future__ import annotations

import json
import sys
from typing import Any

from rich.console import Console
from rich.markup import escape

from trtship.errors import TrtshipError

console = Console()
err_console = Console(stderr=True)


def render_error(exc: TrtshipError) -> None:
    err_console.print(f"[bold red]error:[/] {escape(exc.message)}", highlight=False)
    if exc.hint:
        err_console.print(f"[bold cyan]hint:[/] {escape(exc.hint)}", highlight=False)


def render_unexpected() -> None:
    err_console.print(
        "[bold red]unexpected error[/] (this is a bug in trtship; please report it):",
        highlight=False,
    )
    err_console.print_exception(show_locals=False)


def emit_json(value: Any) -> None:
    """Write machine-readable output to stdout, uncolored and unwrapped."""
    sys.stdout.write(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
