"""External command execution. Always argv lists, never a shell."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from trtship.errors import CommandError


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def output(self) -> str:
        """stdout and stderr combined; some tools (e.g. nvidia-smi) report errors on stdout."""
        return "\n".join(part for part in (self.stdout.strip(), self.stderr.strip()) if part)


def run_command(
    argv: Sequence[str],
    *,
    timeout: float = 30.0,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = False,
) -> CommandResult:
    """Run ``argv`` and capture its output.

    A missing executable or a timeout is reported as :class:`CommandError` regardless of ``check``;
    a non-zero exit code raises only when ``check`` is true.
    """
    command = tuple(argv)
    if not command:
        raise CommandError("cannot run an empty command")
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            check=False,
        )
    except FileNotFoundError as exc:
        raise CommandError(
            f"executable not found: {command[0]}", details={"argv": list(command)}
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise CommandError(
            f"command timed out after {timeout:g}s: {' '.join(command)}",
            details={"argv": list(command)},
        ) from exc
    result = CommandResult(command, completed.returncode, completed.stdout, completed.stderr)
    if check and not result.ok:
        raise CommandError(
            f"command failed with exit code {result.returncode}: {' '.join(command)}",
            hint=result.output[-2000:] or None,
            details={"argv": list(command), "returncode": result.returncode},
        )
    return result
