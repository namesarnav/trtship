from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterator

import numpy as np
import pytest

from trtship.benchmark import PhaseTimes, measure, summarize
from trtship.benchmark.schema import MemoryUsage, ServerSideTimes
from trtship.errors import BenchmarkError

# --------------------------------------------------------------------------- statistics


def test_summary_of_one_to_one_hundred() -> None:
    data = [float(i) for i in range(1, 101)]
    s = summarize(data)
    assert (s.count, s.min_ms, s.max_ms) == (100, 1.0, 100.0)
    assert s.mean_ms == pytest.approx(50.5)
    assert s.p50_ms == pytest.approx(50.5)  # linear interpolation between 50 and 51
    assert s.p90_ms == pytest.approx(90.1)
    assert s.p95_ms == pytest.approx(95.05)
    assert s.p99_ms == pytest.approx(99.01)
    assert s.stdev_ms == pytest.approx(float(np.std(data, ddof=1)))
    assert s.cv == pytest.approx(s.stdev_ms / s.mean_ms)


def test_a_single_sample_has_no_spread() -> None:
    s = summarize([3.5])
    assert (s.count, s.stdev_ms, s.p50_ms, s.p99_ms, s.min_ms, s.max_ms) == (
        1,
        0.0,
        3.5,
        3.5,
        3.5,
        3.5,
    )
    assert s.cv == 0.0


def test_order_does_not_matter() -> None:
    assert summarize([5.0, 1.0, 3.0, 2.0, 4.0]) == summarize([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.mark.parametrize("bad", [[], [1.0, float("nan")], [1.0, math.inf], [-0.5, 1.0]])
def test_invalid_samples_are_rejected(bad: list[float]) -> None:
    with pytest.raises(BenchmarkError):
        summarize(bad)


# --------------------------------------------------------------------------- the runner


class FakeTarget:
    backend = "fake"
    device = "cpu"
    precision: str | None = "fp32"

    def __init__(self, *, concurrent: bool = True, phases: PhaseTimes | None = None) -> None:
        self.supports_concurrency = concurrent
        self.phases = phases or PhaseTimes(1.0, 5.0, 2.0)
        self.prepared: list[int] = []
        self.workers_made = 0
        self.calls = 0
        self.fail_on_call: int | None = None
        self.closed = False

    def prepare(self, batch_size: int) -> None:
        self.prepared.append(batch_size)

    def make_worker(self) -> Callable[[], PhaseTimes]:
        self.workers_made += 1

        def once() -> PhaseTimes:
            self.calls += 1
            if self.fail_on_call is not None and self.calls == self.fail_on_call:
                raise RuntimeError("device lost")
            return self.phases

        return once

    def memory(self) -> MemoryUsage:
        return MemoryUsage(gpu_mb=None, cpu_rss_mb=1.0)

    def notes(self) -> list[str]:
        return ["target note"]

    def close(self) -> None:
        self.closed = True


class TickClock:
    """Advances 2 ms per call, so wall-clock measurements are exact and machine independent."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.002
        return self.now


def sequence_clock(values: list[float]) -> Callable[[], float]:
    source: Iterator[float] = iter(values)
    return lambda: next(source)


def test_a_single_worker_measurement_is_exact_with_an_injected_clock() -> None:
    target = FakeTarget()
    m = measure(target, batch_size=4, concurrency=1, warmup_iters=3, iters=10, clock=TickClock())
    assert target.prepared == [4]
    assert target.calls == 1 + 3 + 10  # the cold call, warmup, then the timed iterations
    assert m.phases["end_to_end"].count == 10  # warmup and the first call are excluded
    assert m.phases["end_to_end"].p50_ms == pytest.approx(2.0)
    assert m.phases["preprocess"].p50_ms == 1.0
    assert m.phases["execute"].p50_ms == 5.0
    assert m.phases["postprocess"].p50_ms == 2.0
    assert m.first_call_ms == pytest.approx(2.0)
    duration = (1 + 2 * 10) * 0.002  # the start mark plus two clock reads per timed request
    assert m.duration_s == pytest.approx(duration)
    assert m.requests_per_s == pytest.approx(10 / duration)
    assert m.throughput_samples_per_s == pytest.approx(10 * 4 / duration)
    assert (m.backend, m.precision, m.device, m.batch_size, m.concurrency) == (
        "fake",
        "fp32",
        "cpu",
        4,
        1,
    )
    assert (m.warmup_iters, m.iters) == (3, 10)
    assert "target note" in m.notes


def test_the_first_call_is_kept_apart_from_the_statistics() -> None:
    # Clock reads, in order: the cold call (start, end), the start of the timed run, then a
    # (start, end) pair per timed request, then the end of the timed run.
    pairs = [x for i in range(3) for x in (float(i), i + 0.001)]
    clock = sequence_clock([0.0, 0.050, 5.0, *pairs, 9.0])
    m = measure(FakeTarget(), batch_size=1, concurrency=1, warmup_iters=0, iters=3, clock=clock)
    assert m.first_call_ms == pytest.approx(50.0)  # the slow cold call
    assert m.phases["end_to_end"].max_ms == pytest.approx(1.0)  # ...is not in the statistics
    assert m.duration_s == pytest.approx(4.0)


def test_concurrent_workers_multiply_the_request_count() -> None:
    target = FakeTarget()
    m = measure(target, batch_size=2, concurrency=4, warmup_iters=1, iters=5)
    assert target.workers_made == 4
    assert m.concurrency == 4
    assert m.phases["end_to_end"].count == 4 * 5
    assert m.requests_per_s == pytest.approx(20 / m.duration_s)
    assert m.throughput_samples_per_s == pytest.approx(40 / m.duration_s)


def test_concurrency_on_a_serial_backend_is_refused_not_faked() -> None:
    target = FakeTarget(concurrent=False)
    with pytest.raises(BenchmarkError, match="cannot run requests concurrently") as info:
        measure(target, batch_size=1, concurrency=2, warmup_iters=0, iters=1)
    assert "Triton" in (info.value.hint or "")
    assert target.prepared == []  # nothing was set up for an impossible request


@pytest.mark.parametrize("call", [3, 12])  # during warmup, and during the timed run
def test_request_failures_surface_as_benchmark_errors(call: int) -> None:
    target = FakeTarget()
    target.fail_on_call = call
    with pytest.raises(BenchmarkError, match="benchmark request failed: RuntimeError: device lost"):
        measure(target, batch_size=1, concurrency=1, warmup_iters=4, iters=10)


def test_benchmark_errors_pass_through_unchanged() -> None:
    class Failing(FakeTarget):
        def make_worker(self) -> Callable[[], PhaseTimes]:
            def once() -> PhaseTimes:
                raise BenchmarkError("specific problem", hint="specific hint")

            return once

    with pytest.raises(BenchmarkError, match="specific problem") as info:
        measure(Failing(), batch_size=1, concurrency=1, warmup_iters=1, iters=1)
    assert info.value.hint == "specific hint"
    # ...but only after the cold call raised: that one propagates directly
    assert isinstance(info.value, BenchmarkError)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iters": 0},
        {"concurrency": 0},
        {"batch_size": 0},
        {"warmup_iters": -1},
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict[str, int]) -> None:
    def call(
        iters: int = 1, concurrency: int = 1, batch_size: int = 1, warmup_iters: int = 0
    ) -> None:
        measure(
            FakeTarget(),
            batch_size=batch_size,
            concurrency=concurrency,
            warmup_iters=warmup_iters,
            iters=iters,
        )

    with pytest.raises(BenchmarkError, match="must be >= 1"):
        call(**kwargs)


def test_few_samples_and_high_variance_are_called_out() -> None:
    walls_ms = [1.0, 1.0, 1.0, 100.0]  # one slow timed request among fast ones
    reads: list[float] = [0.0, 0.001, 1.0]  # cold call (start, end), then the timed run's start
    for index, wall in enumerate(walls_ms):
        reads += [10.0 + index, 10.0 + index + wall / 1000.0]
    reads.append(99.0)
    m = measure(
        FakeTarget(),
        batch_size=1,
        concurrency=1,
        warmup_iters=0,
        iters=4,
        clock=sequence_clock(reads),
    )
    joined = " ".join(m.notes)
    assert "only 4 timed requests" in joined
    assert "high variance" in joined


def test_raw_samples_can_be_omitted() -> None:
    kept = measure(FakeTarget(), batch_size=1, concurrency=1, warmup_iters=0, iters=6)
    assert {k: len(v) for k, v in kept.raw_ms.items()} == {
        "preprocess": 6,
        "execute": 6,
        "postprocess": 6,
        "end_to_end": 6,
    }
    dropped = measure(
        FakeTarget(), batch_size=1, concurrency=1, warmup_iters=0, iters=6, include_raw=False
    )
    assert dropped.raw_ms == {}


def test_each_worker_warms_and_is_timed_on_a_single_thread() -> None:
    threads: dict[int, set[int]] = {}

    class Recording(FakeTarget):
        def make_worker(self) -> Callable[[], PhaseTimes]:
            index = self.workers_made
            self.workers_made += 1

            def once() -> PhaseTimes:
                threads.setdefault(index, set()).add(threading.get_ident())
                return self.phases

            return once

    measure(Recording(), batch_size=1, concurrency=3, warmup_iters=2, iters=3)
    # Worker 0 also makes the cold call on the calling thread; warmup and timing share one thread.
    for index in (1, 2):
        assert len(threads[index]) == 1
    assert len({next(iter(t)) for i, t in threads.items() if i != 0}) == 2  # distinct threads


def test_a_failing_worker_does_not_leave_the_others_waiting() -> None:
    target = FakeTarget()
    target.fail_on_call = 4  # inside the warmup of one worker, others are still running
    result: list[BaseException] = []

    def attempt() -> None:
        try:
            measure(target, batch_size=1, concurrency=3, warmup_iters=5, iters=5)
        except BaseException as exc:
            result.append(exc)

    thread = threading.Thread(target=attempt, daemon=True)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive(), "measure() hung after a worker failed"
    assert isinstance(result[0], BenchmarkError)
    assert "device lost" in str(result[0])


class ObservedTarget(FakeTarget):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def make_worker(self) -> Callable[[], PhaseTimes]:
        inner = super().make_worker()

        def once() -> PhaseTimes:
            self.events.append("call")
            return inner()

        return once

    def timed_started(self) -> None:
        self.events.append("started")

    def timed_finished(self, requests_sent: int) -> ServerSideTimes | None:
        self.events.append(f"finished:{requests_sent}")
        return ServerSideTimes(
            source="test",
            requests=requests_sent,
            queue_ms=1.0,
            compute_input_ms=2.0,
            compute_infer_ms=3.0,
            compute_output_ms=4.0,
        )


def test_the_observer_brackets_exactly_the_timed_calls() -> None:
    target = ObservedTarget()
    m = measure(target, batch_size=1, concurrency=1, warmup_iters=2, iters=3)
    # cold call + 2 warmup calls, then the timed section, then the closing read
    assert target.events == ["call"] * 3 + ["started"] + ["call"] * 3 + ["finished:3"]
    assert m.server_side is not None
    assert m.server_side.total_ms == pytest.approx(10.0)


def test_with_concurrency_the_observer_starts_once_after_all_warmup() -> None:
    target = ObservedTarget()
    measure(target, batch_size=1, concurrency=3, warmup_iters=2, iters=2)
    started = target.events.index("started")
    assert target.events.count("started") == 1
    assert target.events[:started].count("call") == 1 + 3 * 2  # cold call + every warmup call
    assert target.events[started + 1 : -1] == ["call"] * 6
    assert target.events[-1] == "finished:6"


def test_a_failed_measurement_does_not_read_the_server_again() -> None:
    target = ObservedTarget()
    target.fail_on_call = 2
    with pytest.raises(BenchmarkError):
        measure(target, batch_size=1, concurrency=1, warmup_iters=3, iters=3)
    assert not any(e.startswith("finished") for e in target.events)


def test_targets_without_an_observer_report_no_server_side_times() -> None:
    m = measure(FakeTarget(), batch_size=1, concurrency=1, warmup_iters=0, iters=2)
    assert m.server_side is None
