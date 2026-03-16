"""Human-readable number formatting."""

from __future__ import annotations

_BYTE_UNITS = ("B", "KiB", "MiB", "GiB", "TiB")


def format_bytes(value: int | float) -> str:
    size = float(value)
    for unit in _BYTE_UNITS:
        if abs(size) < 1024 or unit == _BYTE_UNITS[-1]:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")  # pragma: no cover


def format_count(value: int) -> str:
    """1234567 -> '1.23M' (decimal magnitudes, as parameter counts are conventionally quoted)."""
    number = float(value)
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(number) >= divisor:
            return f"{number / divisor:.2f}{suffix}"
    return str(value)
