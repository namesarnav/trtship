"""Benchmarking a model served by Triton, over HTTP or gRPC.

Phases as measured from the client: preprocess = building the request tensors, execute = the
request call (serialization, network, queueing, compute and response parsing all happen inside it;
the client cannot separate them), postprocess = decoding the response to numpy. The separation
comes from Triton's own statistics, read before and after the timed section (``timed_started`` /
``timed_finished``), which give the server-side queue and compute means for exactly those requests.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy.typing as npt

from trtship.benchmark.runner import PhaseTimes
from trtship.benchmark.schema import MemoryUsage, ServerSideTimes
from trtship.benchmark.targets import batch_inputs, cpu_rss_mb
from trtship.config import TrtshipConfig
from trtship.errors import BenchmarkError, TritonError
from trtship.triton.client import ModelStatistics, TritonClient

_NS_PER_MS = 1_000_000.0
STATISTICS_SOURCE = "Triton inference statistics"


def server_side_delta(
    before: ModelStatistics, after: ModelStatistics, requests_sent: int
) -> tuple[ServerSideTimes | None, str | None]:
    """Per-request server-side means for the requests between two snapshots.

    Returns ``(times, note)``. The times are omitted when the server counted no requests; a note is
    returned when the server's count differs from what the benchmark sent (other clients, or
    requests that failed), because then the means describe a different population.
    """
    counted = after.success.count - before.success.count
    if counted <= 0:
        return None, "the server counted no successful requests in the timed section"

    def mean_ms(stage: str) -> float:
        delta = getattr(after, stage).ns - getattr(before, stage).ns
        return float(delta) / counted / _NS_PER_MS

    times = ServerSideTimes(
        source=STATISTICS_SOURCE,
        requests=counted,
        queue_ms=mean_ms("queue"),
        compute_input_ms=mean_ms("compute_input"),
        compute_infer_ms=mean_ms("compute_infer"),
        compute_output_ms=mean_ms("compute_output"),
    )
    note = None
    if counted != requests_sent:
        note = (
            f"the server counted {counted} requests but {requests_sent} were sent; the "
            "server-side means may include other traffic or exclude failed requests"
        )
    return times, note


class TritonTarget:
    """A model served by Triton, driven through ``protocol`` with one client per worker."""

    supports_concurrency = True

    def __init__(
        self,
        client_factory: Callable[[], TritonClient],
        model: str,
        config: TrtshipConfig,
        seed: int,
        *,
        protocol: str,
        precision: str | None,
        device: str,
    ) -> None:
        self._factory = client_factory
        self._model = model
        self._config = config
        self._seed = seed
        self.backend = f"triton-{protocol}"
        self.precision = precision
        self.device = device
        self._inputs: dict[str, npt.NDArray[Any]] = {}
        # tritonclient's HTTP client is bound to the thread that created it (it runs on gevent),
        # so every thread that talks to the server gets its own client, created on that thread.
        self._clients: dict[int, TritonClient] = {}
        self._lock = threading.Lock()
        self._before: ModelStatistics | None = None
        self._notes: list[str] = []

    def _client(self) -> TritonClient:
        """The calling thread's client, opened on first use."""
        ident = threading.get_ident()
        with self._lock:
            client = self._clients.get(ident)
        if client is None:
            client = self._factory()
            with self._lock:
                self._clients[ident] = client
        return client

    def prepare(self, batch_size: int) -> None:
        self._inputs = batch_inputs(self._config, batch_size, self._seed)
        self._notes = []
        self._before = None
        client = self._client()
        if not client.model_ready(self._model):
            raise BenchmarkError(
                f"model {self._model!r} is not ready on the server",
                hint="Start it with `trtship serve` and check `trtship status`.",
            )

    def make_worker(self) -> Callable[[], PhaseTimes]:
        model, inputs = self._model, self._inputs

        def once() -> PhaseTimes:
            _, timing = self._client().infer_timed(model, inputs)
            return PhaseTimes(timing.prepare_ms, timing.request_ms, timing.decode_ms)

        return once

    # ------------------------------------------------------------------ server-side timing

    def timed_started(self) -> None:
        try:
            self._before = self._client().inference_statistics(self._model)
        except TritonError as exc:
            self._before = None
            self._notes.append(f"server-side timing unavailable: {exc.message}")

    def timed_finished(self, requests_sent: int) -> ServerSideTimes | None:
        if self._before is None:
            return None
        try:
            after = self._client().inference_statistics(self._model)
        except TritonError as exc:
            self._notes.append(f"server-side timing unavailable: {exc.message}")
            return None
        times, note = server_side_delta(self._before, after, requests_sent)
        if note:
            self._notes.append(note)
        return times

    # ------------------------------------------------------------------ reporting

    def memory(self) -> MemoryUsage:
        return MemoryUsage(gpu_mb=None, cpu_rss_mb=cpu_rss_mb())

    def notes(self) -> list[str]:
        return [
            "execute is the whole request call (serialization, network, server queue and compute, "
            "response parsing); see server_side for the server's own breakdown",
            "gpu_mb is not measured: the GPU belongs to the server process",
            "gpu and tool versions in the environment describe the machine running trtship, "
            "which may not be the server's host",
            *self._notes,
        ]

    def close(self) -> None:
        with self._lock:
            clients, self._clients = list(self._clients.values()), {}
        for client in clients:
            client.close()
