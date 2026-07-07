"""The measurement loop: warmup, timed iterations, concurrency, and phase accounting.

A :class:`BenchmarkTarget` is anything that can perform one request and report how long each phase
took. The runner owns scheduling and statistics; targets own the work and its device timing.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from trtship.benchmark.schema import PHASES, BenchmarkMeasurement, MemoryUsage
from trtship.benchmark.stats import summarize
from trtship.errors import BenchmarkError, TrtshipError
from trtship.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class PhaseTimes:
    """Milliseconds spent in each phase of one request (end-to-end is measured by the runner)."""

    preprocess_ms: float
    execute_ms: float
    postprocess_ms: float


class BenchmarkTarget(Protocol):
    @property
    def backend(self) -> str: ...

    @property
    def device(self) -> str: ...

    @property
    def precision(self) -> str | None: ...

    @property
    def supports_concurrency(self) -> bool: ...

    def prepare(self, batch_size: int) -> None:
        """Set up inputs for ``batch_size`` (called once per measurement, before any timing)."""

    def make_worker(self) -> Callable[[], PhaseTimes]:
        """A callable performing one request. One worker per concurrent thread."""

    def memory(self) -> MemoryUsage: ...

    def notes(self) -> list[str]: ...

    def close(self) -> None: ...


def measure(
    target: BenchmarkTarget,
    *,
    batch_size: int,
    concurrency: int,
    warmup_iters: int,
    iters: int,
    include_raw: bool = True,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchmarkMeasurement:
    """Measure ``target`` at one (batch size, concurrency) point.

    Raises :class:`BenchmarkError` for impossible requests (for example concurrency on a backend
    that cannot run requests in parallel); nothing is estimated or extrapolated.
    """
    if iters < 1 or concurrency < 1 or batch_size < 1 or warmup_iters < 0:
        raise BenchmarkError("iters, concurrency and batch_size must be >= 1 and warmup_iters >= 0")
    if concurrency > 1 and not target.supports_concurrency:
        raise BenchmarkError(
            f"the {target.backend} backend cannot run requests concurrently",
            hint="Use concurrency 1 for this backend, or benchmark through Triton, which schedules "
            "concurrent requests.",
        )
    target.prepare(batch_size)
    workers = [target.make_worker() for _ in range(concurrency)]

    try:
        first_call_ms = _timed_call(workers[0], clock)[1]  # the cold call, kept out of statistics
    except Exception as exc:
        raise _wrap(exc) from exc
    errors: list[BaseException] = []

    def warm(worker: Callable[[], PhaseTimes], index: int) -> None:
        try:
            for _ in range(warmup_iters):
                worker()
        except BaseException as exc:
            errors.append(exc)

    _run_parallel(workers, warm)
    if errors:
        raise _wrap(errors[0])

    results: list[list[tuple[PhaseTimes, float]]] = [[] for _ in workers]

    def timed(worker: Callable[[], PhaseTimes], index: int) -> None:
        try:
            for _ in range(iters):
                results[index].append(_timed_call(worker, clock))
        except BaseException as exc:
            errors.append(exc)

    started = clock()
    _run_parallel(workers, timed)
    duration = clock() - started
    if errors:
        raise _wrap(errors[0])

    samples = [item for per_worker in results for item in per_worker]
    columns = {
        "preprocess": [t.preprocess_ms for t, _ in samples],
        "execute": [t.execute_ms for t, _ in samples],
        "postprocess": [t.postprocess_ms for t, _ in samples],
        "end_to_end": [wall for _, wall in samples],
    }
    total_requests = len(samples)
    return BenchmarkMeasurement(
        backend=target.backend,
        precision=target.precision,
        device=target.device,
        batch_size=batch_size,
        concurrency=concurrency,
        warmup_iters=warmup_iters,
        iters=iters,
        first_call_ms=first_call_ms,
        phases={phase: summarize(columns[phase]) for phase in PHASES},
        throughput_samples_per_s=total_requests * batch_size / duration if duration > 0 else 0.0,
        requests_per_s=total_requests / duration if duration > 0 else 0.0,
        duration_s=duration,
        memory=target.memory(),
        notes=_noise_notes(columns["end_to_end"], iters) + target.notes(),
        raw_ms={k: [round(v, 6) for v in vals] for k, vals in columns.items()}
        if include_raw
        else {},
    )


def _timed_call(
    worker: Callable[[], PhaseTimes], clock: Callable[[], float]
) -> tuple[PhaseTimes, float]:
    start = clock()
    phases = worker()
    return phases, (clock() - start) * 1000.0


def _run_parallel(
    workers: list[Callable[[], PhaseTimes]], body: Callable[[Callable[[], PhaseTimes], int], None]
) -> None:
    """Run ``body(worker, index)`` for every worker, on threads when there is more than one."""
    if len(workers) == 1:
        body(workers[0], 0)
        return
    threads = [threading.Thread(target=body, args=(w, i)) for i, w in enumerate(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def _wrap(exc: BaseException) -> BaseException:
    """Structured trtship errors (with their own exit code and hint) pass through unchanged;
    anything else becomes a BenchmarkError naming the failure."""
    if isinstance(exc, TrtshipError):
        return exc
    return BenchmarkError(f"benchmark request failed: {type(exc).__name__}: {exc}")


def _noise_notes(end_to_end_ms: list[float], iters: int) -> list[str]:
    notes = []
    summary = summarize(end_to_end_ms)
    if len(end_to_end_ms) < 100:
        notes.append(
            f"only {len(end_to_end_ms)} timed requests: p95/p99 are estimates dominated by the "
            "largest samples"
        )
    if summary.cv > 0.25:
        notes.append(
            f"high variance (coefficient of variation {summary.cv:.2f}); the machine may have been "
            "busy or clocks were changing"
        )
    return notes
