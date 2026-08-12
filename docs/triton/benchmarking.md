# Benchmarking a served model

`trtship benchmark triton` measures the model as clients see it: through Triton, over HTTP or gRPC,
using the same warmup, cold-call, percentile and noise rules as the other benchmarks (see
[methodology](../benchmarking/methodology.md)). `trtship benchmark overhead` then sets the result
against the same engine run directly, so the cost of serving is a measured number.

Nothing here has run against a real Triton server. It is tested against a protocol stub that uses the
real `tritonclient` over real sockets and serves synthetic statistics; see
[clients](clients.md#verification-status).

## Measuring

```bash
trtship serve    configs/resnet50.yaml
trtship benchmark triton configs/resnet50.yaml --precision fp16 \
    --protocol http --protocol grpc -o build/served.json
```

Batch sizes and concurrency levels come from the `benchmark` section of the configuration. Unlike a
directly executed TensorRT engine, a served model is measured at every configured concurrency: each
concurrent worker thread has its own client, warms up and is timed on that same thread, and all
workers start timing together.

`--precision` is recorded in the report (the server does not expose it); `--repository` locates the
served plan so its SHA-256 is recorded. Without the plan file the hash is simply omitted.

### Phases

The client cannot look inside a request, so the phases are these:

| Phase | What it covers |
|---|---|
| `preprocess` | building the request tensors |
| `execute` | the request call: serialization, network, server queue and compute, response parsing |
| `postprocess` | decoding the response to numpy arrays |
| `end_to_end` | the wall time of all of the above |

### Server-side times

Triton keeps cumulative per-model statistics. trtship reads them immediately before and after the
timed section and reports, per request, the mean queue, compute-input, compute-infer and
compute-output time (`server_side` in the report, a table in the rendered output). Because they are
differences of cumulative counters:

- warmup and the cold call are excluded;
- the figures are the server's own clock, not something trtship infers;
- if the server counted a different number of requests than trtship sent (other clients, failed
  requests) the report says so in the measurement's notes, since the means then describe a
  different population;
- if the statistics cannot be read, `server_side` is left empty and a note gives the reason. The
  client-side measurement is unaffected.

### What is not measured

- `gpu_mb` is empty: the GPU belongs to the server process, and trtship does not attribute memory
  to it.
- The GPU and tool versions in `environment` describe the machine running trtship, which may not be
  the server's host.

## Serving overhead

```bash
trtship benchmark engine  configs/resnet50.yaml --engine fp16:build/resnet50.fp16.plan -o build/direct.json
trtship benchmark overhead build/direct.json build/served.json
```

Both arguments accept a run directory or a report file. Rows are paired on precision and batch size
between a direct `tensorrt` measurement and a served `triton-*` measurement at concurrency 1, the
only setting where both do the same work. For each pair:

- **Overhead** is the served end-to-end mean minus the direct one, in milliseconds and percent.
- **Queue / Input / Infer / Output** are the server-side means above.
- **Client + net** is *derived*: the served mean minus the server's total. It contains request
  serialization, the network, and response parsing, and it also absorbs any timing error, so treat
  it as an estimate of where the rest of the time went, not a measurement.

Measurements with no counterpart (a concurrency above 1, a precision or batch size measured on only
one side) are listed under "not compared" with the reason rather than dropped. Reports from
different GPUs, different tool versions or different weights produce warnings, as with
`benchmark compare`; overhead between numbers taken on different machines is not meaningful.
