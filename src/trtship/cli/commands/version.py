"""`trtship version`."""

from __future__ import annotations

import platform
from typing import Annotated

import typer

from trtship import __version__
from trtship.cli.render import console, emit_json


def version_command(
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    info = {
        "trtship": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if json_output:
        emit_json(info)
    else:
        console.print(f"trtship {info['trtship']} (python {info['python']}, {info['platform']})")
