# trtship Architecture

trtship turns a trained PyTorch model into a benchmarked, calibrated, TensorRT-optimized,
Triton-servable deployment. This document defines the contracts every module is written against.
Where implementation and this document disagree, fix one of them in the same change.

## 1. Pipeline

```
inspect -> export -> validate -> optimize -> build -> calibrate -> validate_engine -> benchmark -> package -> serve
```

`calibrate` produces the INT8 calibration cache and is consumed by `build` when `int8` is enabled;
in execution order it runs before `build` for INT8 configs. The stage graph is therefore resolved
from declared `requires`/`produces` artifact types, not from list position (see 3.2).

Every stage is:

- **modular** - a class implementing `Stage`, no imports of other stages;
- **independently executable** - `trtship <stage>` runs it against a run directory, resolving inputs
  from that run's artifact store or from explicit CLI paths;
- **idempotent and cacheable** - a stage's cache key is the hash of its input artifact hashes, its
  config slice, and the relevant tool versions. An identical key reuses the existing artifact.

## 2. Layers

```
cli/          Typer commands. Thin: parse args, load config, call pipeline/stages, render with Rich.
pipeline/     Stage protocol, stage registry, DAG resolution, orchestrator, resume/reuse logic.
config/       Pydantic v2 models, YAML loading, env/CLI override merging, JSON-schema export.
models/       Model abstraction: loaders (nn.Module, TorchScript, checkpoint), IOSpec, inspection.
export/       PyTorch -> ONNX.
onnx/         ONNX checker, shape inference, ORT runner, graph optimization.
tensorrt/     Engine builder, optimization profiles, runtime/executor, engine inspection.
calibration/  Dataset abstraction, preprocessing, batching, calibrators, cache, metadata.
validation/   Metrics and PyTorch/ONNX/TensorRT comparison, per-precision tolerances.
benchmark/    Measurement engine, timing, memory sampling, schema, reports, comparison.
triton/       Model repository + config.pbtxt generation, server lifecycle, HTTP/gRPC clients.
artifacts/    Artifact records, content-addressed store, run directory, manifest.
reporting/    Human/JSON/Markdown report rendering.
logging/      Structured logging setup.
utils/        Hashing, environment detection, subprocess, paths, time.
```

Dependency direction is strictly downward: `cli -> pipeline -> {domain packages} -> {artifacts, config, utils}`.
Domain packages do not import from `pipeline` or `cli`. Domain packages may not import each other
except through the shared types in `models/` (IOSpec) and `artifacts/`.

## 3. Contracts

### 3.1 Artifacts

Every product of a stage is an `ArtifactRecord`:

| field | meaning |
|---|---|
| `id` | `<type>-<sha256[:12]>` |
| `type` | `ArtifactType` enum: `model_report`, `onnx`, `onnx_optimized`, `calibration_cache`, `engine`, `validation_report`, `benchmark_report`, `triton_repository`, `report` |
| `path` | path relative to the run directory (never absolute) |
| `sha256` | content hash (directory artifacts hash a sorted manifest of member hashes) |
| `size_bytes` | on-disk size |
| `created_at` | UTC ISO-8601, real wall clock |
| `stage` | producing stage name |
| `parents` | artifact ids this was derived from |
| `metadata` | stage-specific, JSON-serializable |

Rules:

- Artifacts are **immutable once written**. Writing to an existing path with different content raises
  `ArtifactConflictError`; identical content is a cache hit.
- Files are written to a temp path in the same directory and atomically renamed into place.
- The store is per-run (`runs/<run_id>/artifacts/`) with an optional shared content-addressed cache
  (`<cache_dir>/<sha256>`), used for cache reuse across runs.
- Original artifacts are never modified: `optimize` writes `onnx_optimized` beside `onnx`.

### 3.2 Stage protocol

```python
class Stage(Protocol):
    name: ClassVar[str]
    requires: ClassVar[tuple[ArtifactType, ...]]
    produces: ClassVar[tuple[ArtifactType, ...]]
    def cache_key(self, ctx: StageContext) -> str: ...
    def run(self, ctx: StageContext) -> StageResult: ...
```

`StageContext` carries the validated config, the run directory/artifact store, the environment
snapshot, and the logger. `StageResult` carries produced `ArtifactRecord`s, structured metrics, and
warnings. Stages are pure with respect to the artifact store: all inputs come from `ctx`, all outputs
are registered through it.

The orchestrator orders stages topologically from `requires`/`produces`, supports
`--from`, `--only`, `--until`, records per-stage status (`pending|running|succeeded|failed|skipped|cached`)
in the run manifest, and on re-run skips stages whose cache key is unchanged (failure recovery).

### 3.3 Run directory

```
runs/<YYYY-MM-DD>_<shortid>/
  config.yaml          resolved config snapshot
  manifest.json        run id, status, per-stage records, artifact records, versions
  environment.json     python/torch/cuda/tensorrt/onnx/ort versions, GPU, driver, git commit
  artifacts/
  validation/
  benchmarks/
  reports/
  logs/
```

The date component is the real wall-clock date at run creation.

### 3.4 Configuration

Strictly validated Pydantic v2 models (`extra="forbid"`), loaded from YAML. Top-level sections:
`model`, `export`, `optimize`, `tensorrt` (precision, workspace, profiles), `calibration`,
`validation` (tolerances per precision), `benchmark`, `triton`, `artifacts`. Merge order:
defaults < YAML < environment (`TRTSHIP_*`) < CLI flags. Relative paths resolve against the config
file's directory. A JSON Schema is exported to `configs/schemas/` and checked in CI so it cannot drift.

### 3.5 Model abstraction

`ModelSource` loads a `LoadedModel` from `nn.Module` factories (`module:callable`), TorchScript, or
checkpoints (state_dict into a factory-built module). `IOSpec` describes each tensor: name, dtype,
shape with symbolic dims (`"batch"`, `"seq"`), and optional dynamic-axis ranges. Input specs are
declared in config (they cannot be soundly inferred from an arbitrary `nn.Module`) and verified by a
real forward pass. Nothing assumes image classification.

Loading pickled checkpoints/TorchScript executes arbitrary code. Loading uses
`torch.load(weights_only=True)` by default; disabling it requires an explicit
`model.trust_source: true` and logs a warning.

### 3.6 Error model

All expected failures derive from `TrtshipError(message, hint, details)`; each subclass carries a
stable exit code:

| exit | class | meaning |
|---|---|---|
| 0 | - | success |
| 1 | `TrtshipError` | unclassified trtship error |
| 2 | `ConfigError` | invalid config / CLI usage |
| 3 | `EnvironmentUnavailableError` | required capability (GPU, TensorRT, Docker, Triton) missing |
| 4 | `ModelError` | model loading / inspection failed |
| 5 | `ExportError` | ONNX export failed |
| 6 | `ValidationFailedError` | numerical/graph validation exceeded tolerance |
| 7 | `EngineBuildError` | TensorRT build failed (includes unsupported-op report) |
| 8 | `CalibrationError` | calibration failed or data invalid |
| 9 | `BenchmarkError` | benchmark could not run |
| 10 | `TritonError` | repository/server/client failure |
| 11 | `ArtifactError` | conflict, missing, or corrupt artifact |

Unexpected exceptions are bugs: they surface with a traceback and exit code 70. GPU work never falls
back to CPU: if a GPU stage cannot get a GPU it raises `EnvironmentUnavailableError`.

### 3.7 Logging

`logging` stdlib with a Rich console handler for humans and a JSON-lines file handler at
`runs/<id>/logs/trtship.jsonl`. Every record carries `run_id` and `stage`. `--log-level` and
`--json-logs` are global CLI options.

## 4. GPU / CPU boundary

| Needs a GPU | CPU-testable |
|---|---|
| TensorRT engine build and execution, INT8 calibration run, CUDA timing, GPU memory sampling | config, model loading/inspection, ONNX export, checker, shape inference, ORT-on-CPU comparison, graph optimization, metrics, artifact store, run manifest, orchestrator (with test stages), calibration dataset/preprocessing/cache handling, `config.pbtxt` generation from engine metadata, CLI, reports, benchmark statistics |

TensorRT and `tritonclient` are optional extras imported lazily behind capability probes in
`utils/env.py`. Absence yields `EnvironmentUnavailableError` at the point of use, never an import-time
crash of the CLI. GPU tests are marked `@pytest.mark.gpu`, are skipped with an explicit reason when no
usable GPU/TensorRT exists, and CI reports them separately so a skip is never counted as a pass.

CPU benchmarking of ONNX Runtime is allowed but is labelled with its backend in every report; it is
never presented as TensorRT data.

## 5. Reproducibility

`environment.json` captures: run id, git commit (and dirty flag), Python/OS, versions of torch, onnx,
onnxruntime, tensorrt, tritonclient, CUDA runtime, driver version, GPU name/compute capability/memory.
`manifest.json` captures the config hash, model hash, and every artifact hash. Seeds
(`python`, `numpy`, `torch`) are set from config; calibration batch selection is a deterministic
function of the seed and dataset identity.

## 6. Benchmark schema

`BenchmarkReport` (versioned `schema_version`): `backend` (`tensorrt|onnxruntime|triton-http|triton-grpc`),
`precision`, `device`, `batch_size`, `concurrency`, `warmup_iters`, `measured_iters`, per-phase
latency summaries (`preprocess`, `execute`, `postprocess`, `end_to_end`) each with
`min/mean/p50/p90/p95/p99/max/stdev` in ms, `throughput_samples_per_s`, `gpu_memory_mb`,
`cpu_memory_mb`, an optional `server_side` block (mean queue/compute times from the server's
statistics, served backends only), `environment` reference, and raw sample arrays stored alongside the JSON. Methodology
(warmup, CUDA event timing for device phases, synchronization points, clock/power caveats) is in
`docs/benchmarking/`. No number is ever produced without a measurement behind it.

## 7. Triton integration

The repository generator reads the built engine's real I/O tensors (names, dtypes, shapes, profile
ranges) through the TensorRT API and emits `config.pbtxt` (`platform: "tensorrt_plan"`,
`max_batch_size`, `input`, `output`, `instance_group`, optional `dynamic_batching`). The generator
consumes an `EngineInfo` value object, so it is unit-tested on CPU with synthetic `EngineInfo` while
the *extraction* of `EngineInfo` is the GPU-tested part. `serve/stop/status` drive the official
`nvcr.io/nvidia/tritonserver` image via the Docker CLI (no shell string interpolation; argv lists
only), and readiness is verified through the Triton health/model-ready endpoints. Clients wrap
`tritonclient` HTTP and gRPC. Served benchmarks (`TritonTarget`) measure through those clients and
read the model's cumulative Triton statistics around the timed section, which become
`server_side` on the measurement; `reporting.serving_report` pairs served and direct measurements
to derive serving overhead.

## 8. Calibration

`CalibrationDataset` is an abstract iterable of raw samples with a stable `identity()` (name + content
fingerprint). `Preprocessor` turns samples into named-input batches. Implementations: directory of
images, `.npy/.npz` tensors, and a seeded synthetic generator that is explicitly labelled
non-representative and rejected for production INT8 unless `calibration.allow_synthetic: true`. The
TensorRT calibrator (`IInt8EntropyCalibrator2` / `MinMax`, selected by config) reads batches, manages
device buffers, and reads/writes the calibration cache. Metadata recorded next to the cache: dataset
identity, sample count, batch size, preprocessing config, method, TensorRT and CUDA versions, model
hash. A cache is only reused when all of these match.

## 9. Security posture

Model loading is treated as code execution and gated (3.5). Paths from config are resolved and
constrained to the project/run roots where they are outputs. Subprocess calls use argv lists, never
`shell=True`. Docker containers run without `--privileged`, with read-only mounts for the model
repository. Artifacts from other runs are hash-verified before reuse.
