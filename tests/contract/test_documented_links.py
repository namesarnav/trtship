"""Relative links and anchors in the Markdown documentation must point at something real."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PAGES = [
    REPO / name
    for name in ("README.md", "ARCHITECTURE.md", "DECISIONS.md", "PROJECT_PLAN.md")
    if (REPO / name).exists()
] + sorted((REPO / "docs").rglob("*.md"))
FENCE = re.compile(r"```.*?```", re.DOTALL)
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$", re.MULTILINE)


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"[`*_]", "", heading.lower())
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    body = FENCE.sub("", path.read_text("utf-8"))
    return {_slug(h) for h in HEADING.findall(body)}


def _links() -> list[tuple[Path, str]]:
    found = []
    for page in PAGES:
        body = FENCE.sub("", page.read_text("utf-8"))
        found += [(page, target) for target in LINK.findall(body)]
    return found


@pytest.mark.parametrize(
    ("page", "target"), _links(), ids=lambda v: v.name if isinstance(v, Path) else v
)
def test_link_resolves(page: Path, target: str) -> None:
    if re.match(r"[a-z][a-z0-9+.-]*:", target):  # http(s), mailto, ...
        return
    file_part, _, anchor = target.partition("#")
    destination = (page.parent / file_part).resolve() if file_part else page
    assert destination.exists(), f"{page.relative_to(REPO)}: {target} does not exist"
    if anchor and destination.suffix == ".md":
        assert anchor in _anchors(destination), (
            f"{page.relative_to(REPO)}: {target} has no such heading"
        )
