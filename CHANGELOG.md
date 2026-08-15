# Changelog

All notable changes are recorded here with the real date the work was completed.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### 2026-09-19

#### Added
- Project foundation: `pyproject.toml` (uv, hatchling, src layout), Ruff, mypy (strict), pytest and
  coverage configuration, Makefile, pre-commit configuration.
- Structured error model with stable exit codes (`trtship.errors`).
- Environment and capability detection for Python, PyTorch/CUDA, NVIDIA driver and GPUs, CUDA
  toolkit, TensorRT, ONNX, ONNX Runtime, Docker and its NVIDIA runtime, and the Triton client.
- Structured logging: Rich console output and JSON-lines files, with run/stage context.
- Strictly validated YAML configuration with environment and CLI overrides, cross-field checks
  (profiles, calibration, precisions), and resolved-config snapshots.
- Run directories with run IDs, `manifest.json`, `environment.json`, and `config.yaml` snapshots;
  atomic writes.
- CLI: `trtship version`, `trtship doctor`, `trtship config validate`.

- Model abstraction: `module`, `checkpoint` (weights-only by default, common wrapper formats,
  DataParallel prefix stripping) and `torchscript` (opt-in) loaders; weight-identity hashing;
  deterministic example-input generation; output normalization; output signature inference with
  dynamic-dimension attribution.
- Model inspection (`trtship inspect`): parameter counts (total/trainable/frozen, shared weights
  counted once), buffers, exact weight bytes, an activation upper bound measured from a real
  forward pass, module tree, and I/O signature, as versioned JSON and a human-readable report.
- ONNX export (`trtship export`): TorchScript-tracing and `torch.export` exporters, dynamic axes
  from the model signature, opset/constant-folding settings, structural verification of the graph
  against the signature, embedded traceability metadata, atomic no-overwrite publishing, captured
  exporter warnings, and clean unsupported-operator errors.
- ONNX validation (`trtship validate onnx`): official checker with strict shape inference, graph
  statistics, ONNX Runtime execution pinned to the CPU provider with graph optimizations off, and a
  PyTorch-vs-ONNX comparison at the profile's min/opt/max shapes (absolute/relative error, cosine
  similarity, top-1/top-k agreement, KL divergence, distribution stats) with configurable tolerances.
  Catches shapes frozen into the graph during tracing.
- ONNX optimization (`trtship optimize`): non-destructive, subgraph-safe passes (constant
  extraction, identity/dead-node elimination, initializer de-duplication and cleanup, shape
  inference) that write a new file with recorded hashes, sizes and before/after graph statistics,
  reject any pass that changes the model interface, and validate the result against PyTorch.
- Pipeline orchestration (`trtship run`): stages ordered by the artifacts they produce and consume,
  `--from/--only/--until/--dry-run`, capability preflight, scratch-then-publish so failures leave
  nothing half-written, per-stage cache keys with within-run reuse and a content-verified shared
  cache, failure recording and resume (`--run`), per-run structured logs.
- Immutable artifact store with hash verification; `trtship report` (run summary with integrity
  check); `trtship init` (starter config); `trtship config schema` and a checked-in JSON Schema.
- `model.python_path` so a project's model code is importable without touching `PYTHONPATH`;
  example model and config (`configs/examples/custom_model.yaml`).
- TensorRT engine building (`trtship build`, `build` pipeline stage): parser error and unsupported
  operator reporting, FP32/FP16/INT8 flags (INT8 with optional FP16 fallback), optimization
  profiles, workspace and optimization level, timing cache, engine description (`EngineInfo`) for
  Triton configuration, TensorRT 8.6+ compatibility. **Not yet run against real TensorRT**; unit
  tests use a labelled fake and GPU acceptance tests are in `tests/gpu`.
- INT8 calibration (`trtship calibrate`, `calibrate` stage): image/numpy/synthetic (opt-in)
  datasets with content fingerprints, configurable preprocessing, deterministic seeded sampling and
  batching, a calibration cache directory with full metadata, verified adoption of an existing cache
  (every mismatch listed), and TensorRT calibrators (data-feeding and cache-serving) built against
  the real API. The build stage consumes the cache for INT8. **TensorRT calibrator not yet run on
  hardware.**
- Engine execution and validation (`trtship validate engine`, `validate_engine` stage): a
  `TensorRTExecutor` (real API; GPU memory via torch, injectable) and three-way comparison of
  PyTorch, ONNX Runtime, and TensorRT at the profile's min/opt/max shapes, gated per precision by
  the configured tolerance. Logic tested with stand-in executors and a fake TensorRT; **not yet run
  on hardware**.
- Benchmarking (`trtship benchmark onnx|engine|compare`, `benchmark` stage): warmup and cold-call
  separation, p50/p90/p95/p99, throughput, CPU/GPU memory where measurable, batch size and
  concurrency matrices, preprocess/execute/postprocess/end-to-end phases (CUDA-event GPU timing for
  TensorRT), an ONNX Runtime CPU baseline, versioned JSON reports with embedded methodology,
  Markdown rendering, and run-to-run comparison that flags different machines, tool versions, or
  weights. The ONNX Runtime backend is real; the TensorRT backend is **not yet run on hardware**.
- Triton model repository (`trtship package`, `package` stage): `config.pbtxt` generated from the
  engine's real tensors and profile (batching, dtypes, instance groups, dynamic batching), the
  `<model>/<version>/model.plan` layout, refusal to package an engine that failed validation, and
  an engine description file written by `trtship build`. Tests parse the output with Triton's
  protobuf schema; **no Triton server has loaded it**.
- Triton server management (`trtship serve|status|stop`): Docker CLI (argv only), read-only
  repository mount, ports bound to `triton.bind_address` (default `127.0.0.1`), readiness polling
  of the server and every model, container logs in failures, cleanup of a failed start. No real
  Triton container has been started; `status`/`stop` were run against real Docker.
- Triton clients (`trtship.triton.TritonClient`, HTTP and gRPC over `tritonclient`): health, metadata,
  binary-tensor inference, structured errors, readiness waiting; `trtship validate triton` compares a
  served model with PyTorch using the engine-validation gates. Tested against a KServe v2 stub with
  the real client library; **not yet run against a real Triton server**.
- Capability preflight now runs before a run directory is created.
- Shared comparison metrics (`trtship.validation`) reused by later TensorRT validation.
- Shared `TensorSpec`/`DType` (`trtship.specs`) with `value_range` for generated data.

- Triton benchmarking (`trtship benchmark triton`, `trtship benchmark overhead`): measures a served
  model over HTTP and gRPC through the real client, reads Triton's own queue/compute statistics
  around the timed section (`server_side`), and reports serving overhead against a directly
  executed engine with the client + network share labelled as derived. Tested against a protocol
  stub; **not yet run against a real Triton server**.

#### Changed
- `measure()` warms up and times each concurrent worker on a single thread, with a barrier between
  the phases, so clients that are bound to their creating thread (tritonclient over HTTP) work.
