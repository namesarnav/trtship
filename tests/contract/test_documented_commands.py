"""Every ``trtship`` command line shown in the documentation must exist in the real CLI.

The docs are parsed, not hand-copied: a renamed command or option, or a documented command that was
never built, fails here rather than in a user's terminal.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import typer.core
import typer.main

from trtship.cli.main import app

REPO = Path(__file__).resolve().parents[2]
DOCS = [REPO / "README.md", *sorted((REPO / "docs").rglob("*.md"))]
FENCE = re.compile(r"```(?:bash|sh|console|shell)?\n(.*?)```", re.DOTALL)
ROOT = typer.main.get_command(app)


def documented_commands() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for path in DOCS:
        for block in FENCE.findall(path.read_text("utf-8")):
            for raw in block.replace("\\\n", " ").splitlines():
                line = raw.split("  #")[0].strip().removeprefix("$ ")
                if line.startswith("trtship ") and "|" not in line:
                    found.append((str(path.relative_to(REPO)), line))
    return found


COMMANDS = documented_commands()


def resolve(words: list[str]) -> tuple[Any, list[str]]:
    """Walk the command tree; returns the leaf command and the words left over."""
    command = ROOT
    rest = list(words)
    while isinstance(command, typer.core.TyperGroup) and rest and rest[0] in command.commands:
        command = command.commands[rest.pop(0)]
    return command, rest


def option_names(command: Any) -> set[str]:
    names = {"--help", "-h"}
    for param in command.params:
        if isinstance(param, typer.core.TyperOption):
            names.update(param.opts)
            names.update(param.secondary_opts)
    return names


def test_the_documentation_contains_commands_to_check() -> None:
    assert len(COMMANDS) > 20, "the docs parser found suspiciously few commands"


@pytest.mark.parametrize(("source", "line"), COMMANDS, ids=[f"{s}: {c}" for s, c in COMMANDS])
def test_documented_command_exists(source: str, line: str) -> None:
    words = [w for w in shlex.split(line)[1:] if not w.startswith("<")]
    command, rest = resolve(words)
    assert command is not ROOT, f"{source}: `{line}` is not a trtship command"
    if isinstance(command, typer.core.TyperGroup):
        pytest.fail(f"{source}: `{line}` needs a subcommand; {sorted(command.commands)} exist")
    known = option_names(command)
    for word in rest:
        if word.startswith("-"):
            flag = word.split("=")[0]
            assert flag in known, f"{source}: `{line}` uses {flag}, which {command.name} lacks"


@pytest.mark.parametrize(("source", "line"), COMMANDS, ids=[f"{s}: {c}" for s, c in COMMANDS])
def test_documented_config_files_exist(source: str, line: str) -> None:
    for word in shlex.split(line):
        if word.startswith("configs/") and word.endswith((".yaml", ".yml")):
            assert (REPO / word).is_file(), f"{source}: `{line}` names {word}, which does not exist"
