"""Git metadata for reproducibility records."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from trtship.errors import CommandError
from trtship.utils.subprocess import run_command


class GitInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    commit: str
    branch: str | None
    dirty: bool


def git_info(cwd: Path) -> GitInfo | None:
    """Return the commit/branch/dirty state of the repository containing ``cwd``, if any."""
    try:
        head = run_command(["git", "rev-parse", "HEAD"], cwd=cwd, timeout=10)
        if not head.ok:
            return None
        branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, timeout=10)
        status = run_command(["git", "status", "--porcelain"], cwd=cwd, timeout=10)
    except CommandError:
        return None
    branch_name = branch.stdout.strip() if branch.ok else None
    return GitInfo(
        commit=head.stdout.strip(),
        branch=None if branch_name in (None, "", "HEAD") else branch_name,
        dirty=bool(status.stdout.strip()) if status.ok else False,
    )
