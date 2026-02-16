"""Wall-clock helpers. All persisted timestamps are timezone-aware UTC."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


def isoformat(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds")
