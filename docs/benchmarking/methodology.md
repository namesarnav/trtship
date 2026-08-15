# Benchmarking

```bash
trtship benchmark onnx my-config.yaml build/model.onnx                       # ONNX Runtime on the CPU
trtship benchmark engine my-config.yaml --engine fp16:build/model.fp16.plan  # TensorRT direct (GPU)
trtship benchmark compare runs/run_a runs/run_b                              # or two report files
trtship run my-config.yaml                                                   # includes the benchmark stage
```

> **Verification status.** The ONNX Runtime CPU benchmark is real and runs anywhere. The statistics
> and the measurement loop are tested exactly with injected clocks. The TensorRT backend is written
> against the real API (CUDA-event timing, GPU memory by device-memory growth) and tested with a
> labelled fake; **it has not run on a GPU**, and this repository contains no TensorRT or Triton
> numbers. Nothing is ever reported that was not measured, and every measurement is labelled with its
> backend and device.

## What is measured

For each `(batch size, concurrency)` pair in `benchmark.batch_sizes` x `benchmark.concurrency`:

| step | meaning |
|---|---|
| first call | the cold call (allocations, lazy initialization), recorded as `first_call_ms` and kept out of the statistics |
| warmup | `benchmark.warmup_iters` further iterations, discarded |
| timed run | `benchmark.iters` iterations **per worker**, all recorded |

Each request is split into phases:

| phase | ONNX Runtime (CPU) | TensorRT (direct) |
|---|---|---|
| `preprocess` | preparing the input feed | host-to-device copy of the inputs |
| `execute` | `session.run` | GPU time between two CUDA events around `execute_async_v3` |
| `postprocess` | converting outputs to arrays | device-to-host copy of the outputs |
| `end_to_end` | wall clock of the whole request, with synchronization | same |

Reported per phase: count, min, mean, stdev, p50, p90, p95, p99, max (milliseconds, linear
interpolation between order statistics). Also reported: throughput in samples per second (samples
completed divided by wall-clock time across all workers), requests per second, peak CPU RSS of the
process, GPU memory (TensorRT: growth in device memory in use between before the engine was loaded
and after the run, all processes; `null` when not measured), and raw per-request samples.

Inputs are deterministic (seeded) and use the profile's `opt` shape with the batch axis replaced. A
batch size outside the profile's range is an error, so an engine is only benchmarked at shapes it was
built for.

## Reading the numbers honestly

- **Concurrency.** ONNX Runtime runs concurrent requests on threads. A directly executed TensorRT
  engine runs one request at a time; combinations with `concurrency > 1` are listed under
  `skipped` with the reason, never silently dropped or faked. Concurrency for TensorRT is a serving
  concern and is measured through Triton (see [benchmarking a served model](../triton/benchmarking.md)).
- **Noise.** A note is attached when a measurement has fewer than 100 timed requests (tail
  percentiles are estimates there) or a coefficient of variation above 0.25 (the machine was likely
  busy or clocks were changing).
- **Comparability.** Results depend on the machine, clocks, power state and other load. Compare
  runs only from the same machine. `benchmark compare` warns when GPUs, tool versions, or model
  weights differ, and lists measurements present in only one run.
- **Baseline.** The ONNX Runtime CPU baseline (`benchmark.include_onnx_baseline`, default true) is a
  CPU measurement; comparing it with a GPU engine shows the speed-up of moving to the GPU, not a
  like-for-like framework comparison.

## Reports and comparison

Each measurement is a JSON `BenchmarkReport` (versioned; schema in `trtship.benchmark.schema`),
published as `benchmark_onnx_cpu.json` and `benchmark_engines.json` artifacts, with the methodology
text embedded. `trtship benchmark compare A B` accepts run directories (all their benchmark reports)
or report files and matches measurements on backend, precision, batch size, and concurrency; it
prints the change in end-to-end p50/p95/p99 (negative means B is faster) and the throughput ratio.
`trtship.reporting.render_benchmark_markdown` renders a report as Markdown.

The `benchmark` stage is never cached or skipped as up to date: a benchmark is a measurement of this
machine at this moment, so each run performs it again.
