"""The cost of serving: a Triton measurement set against the same engine run directly.

Rows are matched on (precision, batch size) between a direct ``tensorrt`` measurement and a served
``triton-*`` measurement at concurrency 1, the only setting where the two do the same work. The
difference of the end-to-end means is the measured serving overhead. The server's own statistics
split the served time into queue and compute; whatever remains (client serialization, network,
response parsing) is *derived* by subtraction and labelled that way.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from trtship.benchmark import BenchmarkMeasurement, BenchmarkReport
from trtship.reporting.benchmark_report import environment_warnings

DIRECT_BACKEND = "tensorrt"
SERVED_PREFIX = "triton-"


class ServingOverhead(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: str
    precision: str
    batch_size: int
    direct_mean_ms: float
    served_mean_ms: float
    overhead_ms: float
    overhead_pct: float
    direct_p50_ms: float
    served_p50_ms: float
    direct_p99_ms: float
    served_p99_ms: float
    # Server-side means per request; None when the server's statistics were unavailable.
    queue_ms: float | None
    compute_input_ms: float | None
    compute_infer_ms: float | None
    compute_output_ms: float | None
    server_total_ms: float | None
    # served mean minus server_total: client-side and network time, derived, not measured directly.
    client_and_network_ms: float | None


class ServingComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    rows: list[ServingOverhead]
    unmatched: list[str]
    warnings: list[str]


def _served(m: BenchmarkMeasurement) -> bool:
    return m.backend.startswith(SERVED_PREFIX)


def _describe(m: BenchmarkMeasurement) -> str:
    return f"{m.backend}/{m.precision or '-'} batch={m.batch_size} conc={m.concurrency}"


def compare_serving(
    direct: list[BenchmarkReport], served: list[BenchmarkReport]
) -> ServingComparison:
    """Pair direct engine measurements with served ones and derive the serving overhead."""
    direct_by_key: dict[tuple[str, int], BenchmarkMeasurement] = {}
    for report in direct:
        for m in report.measurements:
            if m.backend == DIRECT_BACKEND and m.concurrency == 1:
                direct_by_key[(m.precision or "-", m.batch_size)] = m

    rows: list[ServingOverhead] = []
    unmatched: list[str] = []
    seen: set[tuple[str, str, int]] = set()
    for report in served:
        for m in report.measurements:
            if not _served(m):
                continue
            if m.concurrency != 1:
                unmatched.append(
                    f"{_describe(m)}: direct measurements are single-stream, so there is "
                    "nothing at this concurrency to compare against"
                )
                continue
            key = (m.precision or "-", m.batch_size)
            base = direct_by_key.get(key)
            if base is None:
                unmatched.append(f"{_describe(m)}: no direct tensorrt measurement to compare with")
                continue
            identity = (m.backend, *key)
            if identity in seen:
                unmatched.append(f"{_describe(m)}: duplicate served measurement ignored")
                continue
            seen.add(identity)
            rows.append(_row(m, base))

    matched_direct = {(p, b) for _, p, b in seen}
    for (precision, batch), m in sorted(direct_by_key.items()):
        if (precision, batch) not in matched_direct:
            unmatched.append(f"{_describe(m)}: no served measurement to compare with")

    warnings = environment_warnings(direct, served)
    if not rows:
        warnings.append("no direct and served measurements could be paired")
    return ServingComparison(
        rows=sorted(rows, key=lambda r: (r.backend, r.precision, r.batch_size)),
        unmatched=unmatched,
        warnings=warnings,
    )


def _row(served: BenchmarkMeasurement, direct: BenchmarkMeasurement) -> ServingOverhead:
    d, s = direct.phases["end_to_end"], served.phases["end_to_end"]
    side = served.server_side
    overhead = s.mean_ms - d.mean_ms
    return ServingOverhead(
        backend=served.backend,
        precision=served.precision or "-",
        batch_size=served.batch_size,
        direct_mean_ms=d.mean_ms,
        served_mean_ms=s.mean_ms,
        overhead_ms=overhead,
        overhead_pct=overhead / d.mean_ms * 100.0 if d.mean_ms else 0.0,
        direct_p50_ms=d.p50_ms,
        served_p50_ms=s.p50_ms,
        direct_p99_ms=d.p99_ms,
        served_p99_ms=s.p99_ms,
        queue_ms=side.queue_ms if side else None,
        compute_input_ms=side.compute_input_ms if side else None,
        compute_infer_ms=side.compute_infer_ms if side else None,
        compute_output_ms=side.compute_output_ms if side else None,
        server_total_ms=side.total_ms if side else None,
        client_and_network_ms=s.mean_ms - side.total_ms if side else None,
    )


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def render_serving(comparison: ServingComparison, console: Console) -> None:
    for warning in comparison.warnings:
        console.print(f"[yellow]warning:[/] {escape(warning)}", highlight=False)
    table = Table(title="Serving overhead (means, ms)", title_justify="left")
    for column in (
        "Backend",
        "Prec",
        "Batch",
        "Direct",
        "Served",
        "Overhead",
        "Queue",
        "Input",
        "Infer",
        "Output",
        "Client+net*",
    ):
        table.add_column(column, justify="left" if column in ("Backend", "Prec") else "right")
    for r in comparison.rows:
        table.add_row(
            r.backend,
            r.precision,
            str(r.batch_size),
            _ms(r.direct_mean_ms),
            _ms(r.served_mean_ms),
            f"{r.overhead_ms:+.3f} ({r.overhead_pct:+.1f}%)",
            _ms(r.queue_ms),
            _ms(r.compute_input_ms),
            _ms(r.compute_infer_ms),
            _ms(r.compute_output_ms),
            _ms(r.client_and_network_ms),
        )
    console.print(table)
    console.print(
        "* derived: served mean minus the server's own queue and compute means. "
        "Queue/Input/Infer/Output come from Triton's statistics."
    )
    for line in comparison.unmatched:
        console.print(f"[dim]not compared:[/] {escape(line)}", highlight=False)


def render_serving_markdown(comparison: ServingComparison) -> str:
    lines = ["# Serving overhead", ""]
    lines += [f"> Warning: {w}" for w in comparison.warnings]
    if comparison.warnings:
        lines.append("")
    lines += [
        "| Backend | Precision | Batch | Direct mean (ms) | Served mean (ms) | Overhead (ms) | "
        "Overhead % | Queue | Compute input | Compute infer | Compute output | "
        "Client + network (derived) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in comparison.rows:
        lines.append(
            f"| {r.backend} | {r.precision} | {r.batch_size} | {r.direct_mean_ms:.3f} | "
            f"{r.served_mean_ms:.3f} | {r.overhead_ms:+.3f} | {r.overhead_pct:+.1f}% | "
            f"{_ms(r.queue_ms)} | {_ms(r.compute_input_ms)} | {_ms(r.compute_infer_ms)} | "
            f"{_ms(r.compute_output_ms)} | {_ms(r.client_and_network_ms)} |"
        )
    if comparison.unmatched:
        lines += ["", "## Not compared", ""] + [f"- {u}" for u in comparison.unmatched]
    lines += [
        "",
        "Queue and compute figures are Triton's own statistics. Client + network is derived as "
        "the served mean minus the server total, so it also absorbs any measurement error.",
        "",
    ]
    return "\n".join(lines)
