from __future__ import annotations

import pytest

from trtship.utils.units import format_bytes, format_count


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0 B"),
        (1023, "1023 B"),
        (1024, "1.00 KiB"),
        (1536, "1.50 KiB"),
        (5 * 1024**2, "5.00 MiB"),
        (3 * 1024**3, "3.00 GiB"),
        (2 * 1024**4, "2.00 TiB"),
        (5 * 1024**5, "5120.00 TiB"),  # beyond TiB stays in TiB
    ],
)
def test_format_bytes(value: int, expected: str) -> None:
    assert format_bytes(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (999, "999"),
        (1000, "1.00K"),
        (1_234_567, "1.23M"),
        (25_557_032, "25.56M"),
        (1_500_000_000, "1.50B"),
        (2_000_000_000_000, "2.00T"),
    ],
)
def test_format_count(value: int, expected: str) -> None:
    assert format_count(value) == expected
