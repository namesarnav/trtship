"""Rendering and comparison of benchmark reports.

Comparison never implies more than the data supports: it flags runs from different machines or
model weights, lists measurements present in only one run, and reports changes as measured deltas.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from trtship.artifacts import ArtifactStore, ArtifactType, RunDirectory
from trtship.benchmark import BenchmarkMeasurement, BenchmarkReport
from trtship.errors import ArtifactError, BenchmarkError

Key = tuple[str, str, int, int]  # backend, precision, batch, concurrency


def _key(m: BenchmarkMeasurement) -> Key:
    return (m.backend, m.precision or "-", m.batch_size, m.concurrency)


def _label(key: Key) -> str:
    backend, precision, batch, concurrency = key
    return f"{backend}/{precision} batch={batch} conc={concurrency}"


def _mb(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f}"


# --------------------------------------------------------------------------- rendering


def render_benchmark(report: BenchmarkReport, console: Console) -> None:
    subject = report.subject
    console.print(
        f"Benchmark [bold]{escape(report.model_name)}[/]  subject: {subject.kind}"
        + (f" ({escape(subject.path)})" if subject.path else "")
    )
    env = report.environment
    console.print(
        f"gpus: {', '.join(env.get('gpus', [])) or 'none'}; seed {report.seed}; "
        f"measured {report.generated_at.isoformat(timespec='seconds')}"
    )
    table = Table(title="Latency (ms) and throughput", title_justify="left")
    for column in (
        "Backend",
        "Prec",
        "Batch",
        "Conc",
        "e2e p50",
        "e2e p95",
        "e2e p99",
        "exec p50",
        "Samples/s",
        "GPU MB",
        "First call",
    ):
        table.add_column(column, justify="right" if column not in ("Backend", "Prec") else "left")
    for m in report.measurements:
        e2e, execute = m.phases["end_to_end"], m.phases["execute"]
        table.add_row(
            m.backend,
            m.precision or "-",
            str(m.batch_size),
            str(m.concurrency),
            f"{e2e.p50_ms:.3f}",
            f"{e2e.p95_ms:.3f}",
            f"{e2e.p99_ms:.3f}",
            f"{execute.p50_ms:.3f}",
            f"{m.throughput_samples_per_s:,.1f}",
            _mb(m.memory.gpu_mb),
            f"{m.first_call_ms:.2f}",
        )
    console.print(table)
    served = [m for m in report.measurements if m.server_side is not None]
    if served:
        side = Table(title="Server-side means per request (ms)", title_justify="left")
        for column in ("Backend", "Batch", "Conc", "Queue", "Input", "Infer", "Output", "Requests"):
            side.add_column(column, justify="left" if column == "Backend" else "right")
        for m in served:
            t = m.server_side
            assert t is not None
            side.add_row(
                m.backend,
                str(m.batch_size),
                str(m.concurrency),
                f"{t.queue_ms:.3f}",
                f"{t.compute_input_ms:.3f}",
                f"{t.compute_infer_ms:.3f}",
                f"{t.compute_output_ms:.3f}",
                str(t.requests),
            )
        console.print(side)
    for m in report.measurements:
        for note in m.notes:
            console.print(f"[dim]{escape(_label(_key(m)))}:[/] {escape(note)}", highlight=False)
    for skipped in report.skipped:
        console.print(f"[yellow]not measured:[/] {escape(skipped)}", highlight=False)
    console.print(f"\n[dim]{escape(report.methodology)}[/]")


def render_benchmark_markdown(report: BenchmarkReport) -> str:
    lines = [
        f"# Benchmark: {report.model_name}",
        "",
        f"- Subject: {report.subject.kind}" + (f" `{p}`" if (p := report.subject.path) else ""),
        f"- Measured: {report.generated_at.isoformat(timespec='seconds')}",
        f"- GPUs: {', '.join(report.environment.get('gpus', [])) or 'none'}",
        f"- Seed: {report.seed}",
        "",
        "| Backend | Precision | Batch | Concurrency | e2e p50 (ms) | e2e p95 (ms) | "
        "e2e p99 (ms) | execute p50 (ms) | Samples/s | GPU MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in report.measurements:
        e2e, execute = m.phases["end_to_end"], m.phases["execute"]
        lines.append(
            f"| {m.backend} | {m.precision or '-'} | {m.batch_size} | {m.concurrency} | "
            f"{e2e.p50_ms:.3f} | {e2e.p95_ms:.3f} | {e2e.p99_ms:.3f} | {execute.p50_ms:.3f} | "
            f"{m.throughput_samples_per_s:,.1f} | {_mb(m.memory.gpu_mb)} |"
        )
    served = [m for m in report.measurements if m.server_side is not None]
    if served:
        lines += [
            "",
            "## Server-side means per request",
            "",
            "| Backend | Batch | Concurrency | Queue (ms) | Compute input (ms) | "
            "Compute infer (ms) | Compute output (ms) | Requests |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for m in served:
            t = m.server_side
            assert t is not None
            lines.append(
                f"| {m.backend} | {m.batch_size} | {m.concurrency} | {t.queue_ms:.3f} | "
                f"{t.compute_input_ms:.3f} | {t.compute_infer_ms:.3f} | "
                f"{t.compute_output_ms:.3f} | {t.requests} |"
            )
    notes = sorted({note for m in report.measurements for note in m.notes})
    if notes or report.skipped:
        lines += ["", "## Notes", ""]
        lines += [f"- {n}" for n in notes] + [f"- Not measured: {s}" for s in report.skipped]
    lines += ["", "## Methodology", "", report.methodology, ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- comparison


class MeasurementDelta(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    p50_a_ms: float
    p50_b_ms: float
    p50_change_pct: float  # negative = B is faster
    p95_change_pct: float
    p99_change_pct: float
    throughput_a: float
    throughput_b: float
    throughput_ratio: float  # B / A


class Comparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    deltas: list[MeasurementDelta]
    only_in_a: list[str]
    only_in_b: list[str]
    warnings: list[str]


def _pct(a: float, b: float) -> float:
    return (b - a) / a * 100.0 if a else 0.0


def environment_warnings(a: list[BenchmarkReport], b: list[BenchmarkReport]) -> list[str]:
    """Reasons two sets of reports may not be comparable: hardware, tool versions, or weights."""
    warnings: list[str] = []
    gpus_a = {g for r in a for g in r.environment.get("gpus", [])}
    gpus_b = {g for r in b for g in r.environment.get("gpus", [])}
    if gpus_a != gpus_b:
        warnings.append(
            f"the runs were measured on different GPUs ({sorted(gpus_a) or 'none'} vs "
            f"{sorted(gpus_b) or 'none'}); differences may reflect hardware, not the change"
        )
    versions_a = {k: v for r in a for k, v in r.environment.get("versions", {}).items()}
    versions_b = {k: v for r in b for k, v in r.environment.get("versions", {}).items()}
    changed = sorted(
        k
        for k in {*versions_a, *versions_b} - {"platform"}
        if versions_a.get(k) != versions_b.get(k)
    )
    if changed:
        warnings.append(f"tool versions differ between the runs: {', '.join(changed)}")
    weights_a = {r.weights_sha256 for r in a}
    weights_b = {r.weights_sha256 for r in b}
    if weights_a != weights_b:
        warnings.append("the runs benchmarked different model weights")
    return warnings


def compare_reports(a: list[BenchmarkReport], b: list[BenchmarkReport]) -> Comparison:
    """Compare two sets of measurements matched on (backend, precision, batch, concurrency)."""
    left = {_key(m): m for r in a for m in r.measurements}
    right = {_key(m): m for r in b for m in r.measurements}
    warnings = environment_warnings(a, b)
    deltas = []
    for key in sorted(left.keys() & right.keys()):
        ma, mb = left[key], right[key]
        ea, eb = ma.phases["end_to_end"], mb.phases["end_to_end"]
        deltas.append(
            MeasurementDelta(
                label=_label(key),
                p50_a_ms=ea.p50_ms,
                p50_b_ms=eb.p50_ms,
                p50_change_pct=_pct(ea.p50_ms, eb.p50_ms),
                p95_change_pct=_pct(ea.p95_ms, eb.p95_ms),
                p99_change_pct=_pct(ea.p99_ms, eb.p99_ms),
                throughput_a=ma.throughput_samples_per_s,
                throughput_b=mb.throughput_samples_per_s,
                throughput_ratio=(
                    mb.throughput_samples_per_s / ma.throughput_samples_per_s
                    if ma.throughput_samples_per_s
                    else 0.0
                ),
            )
        )
    if not deltas:
        warnings.append("the runs share no comparable measurements")
    return Comparison(
        deltas=deltas,
        only_in_a=[_label(k) for k in sorted(left.keys() - right.keys())],
        only_in_b=[_label(k) for k in sorted(right.keys() - left.keys())],
        warnings=warnings,
    )


def render_comparison(comparison: Comparison, console: Console, name_a: str, name_b: str) -> None:
    for warning in comparison.warnings:
        console.print(f"[yellow]warning:[/] {escape(warning)}", highlight=False)
    table = Table(title=f"{escape(name_a)} -> {escape(name_b)} (end-to-end)", title_justify="left")
    for column in ("Measurement", "p50 A", "p50 B", "p50", "p95", "p99", "Samples/s x"):
        table.add_column(column, justify="left" if column == "Measurement" else "right")
    for d in comparison.deltas:
        table.add_row(
            escape(d.label),
            f"{d.p50_a_ms:.3f}",
            f"{d.p50_b_ms:.3f}",
            f"{d.p50_change_pct:+.1f}%",
            f"{d.p95_change_pct:+.1f}%",
            f"{d.p99_change_pct:+.1f}%",
            f"{d.throughput_ratio:.2f}x",
        )
    console.print(table)
    console.print("Negative latency changes mean B is faster; throughput x is B relative to A.")
    for label in comparison.only_in_a:
        console.print(f"[dim]only in A:[/] {escape(label)}", highlight=False)
    for label in comparison.only_in_b:
        console.print(f"[dim]only in B:[/] {escape(label)}", highlight=False)


# --------------------------------------------------------------------------- loading


def load_benchmark_reports(path: Path) -> list[BenchmarkReport]:
    """Benchmark reports from a run directory (all registered ones) or a single JSON file."""
    if path.is_file():
        try:
            return [BenchmarkReport.model_validate(json.loads(path.read_text("utf-8")))]
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise BenchmarkError(f"{path} is not a benchmark report: {exc}") from exc
    run = RunDirectory.open(path)
    store = ArtifactStore(run)
    records = store.records(ArtifactType.BENCHMARK_REPORT)
    if not records:
        raise ArtifactError(
            f"run {run.run_id} has no benchmark report", hint="Run the benchmark stage first."
        )
    return [load_benchmark_reports(store.absolute(r))[0] for r in records]
