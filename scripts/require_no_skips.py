"""Fail unless a JUnit report shows tests that ran and none that were skipped.

The hardware job exists to run GPU and TensorRT tests. A run in which they were all skipped (a
runner without a working driver, say) must not look like a pass, so this exits non-zero when the
report has no executed tests or any skipped, failed or errored ones.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def summarize(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()  # noqa: S314
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for suite in suites:
        for key in totals:
            totals[key] += int(suite.get(key, 0))
    return totals


def verdict(totals: dict[str, int]) -> str | None:
    """A reason the report is unacceptable, or None."""
    executed = totals["tests"] - totals["skipped"]
    if totals["tests"] == 0:
        return "no tests were collected"
    if totals["failures"] or totals["errors"]:
        return f"{totals['failures']} failed and {totals['errors']} errored"
    if totals["skipped"]:
        return f"{totals['skipped']} of {totals['tests']} tests were skipped, not run"
    if executed == 0:
        return "no tests were executed"
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: require_no_skips.py JUNIT_XML", file=sys.stderr)
        return 2
    totals = summarize(Path(argv[1]))
    reason = verdict(totals)
    if reason:
        print(f"hardware tests did not pass: {reason}", file=sys.stderr)
        return 1
    print(f"{totals['tests']} hardware tests ran and passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
