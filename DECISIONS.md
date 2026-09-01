# Decisions

Format: numbered, dated (real date the decision was made), with rationale and consequences.

## D-001 - Repository baseline (2026-09-19)

The working directory was a clone of a repository that previously held an unrelated project; the
owner renamed the GitHub repository to `trtship` and cleared its contents (remote `main` = `c88b3f9`).
Local `master` was fast-forwarded to `origin/main` and renamed `main`; `origin` was repointed at
`git@github.com:namesarnav/trtship.git`. No history was rewritten and nothing has been pushed.
The old `setup.py` (empty) is removed in favor of `pyproject.toml`. Commits use the repo-local
identity chosen by the owner.

## D-002 - Package manager: uv (2026-09-19)

Spec allows uv or Poetry. uv chosen: the target quickstart is `uv sync`, it resolves faster, and it
handles the PyTorch CPU/CUDA index split via `tool.uv.sources`. uv was not installed on the dev
machine and was installed with `pip install --user uv`. Build backend: `hatchling`, `src/` layout.

## D-003 - Python version (2026-09-19)

`requires-python >= 3.11`. Dev machine runs 3.12.9. CI matrix: 3.11 and 3.12.

## D-004 - Dependency tiers (2026-09-19)

- Core: `pydantic`, `typer`, `rich`, `pyyaml`, `numpy`, `onnx`, `onnxruntime`, `torch`.
- Extra `trt`: `tensorrt` (+ `cuda-python` for device memory). Not installable on machines without the
  NVIDIA index / suitable platform, so it is never a hard dependency.
- Extra `triton`: `tritonclient[all]`.
- Dependency group `dev` (not an extra): `pytest`, `pytest-cov`, `ruff`, `mypy`, `pre-commit`, type stubs.
Torch is core because model loading/export/validation are core, CPU-testable functionality.

Amended 2026-09-19: uv rejects a `tool.uv.sources` entry conditioned on an extra when the package is
also a base dependency, so a `cpu` extra that swaps the torch index is not possible while torch
stays in the base dependencies. Instead the lock uses PyPI torch (CUDA on Linux, so a plain
`uv sync` works on GPU machines) and machines without a GPU run `make install-cpu`, which installs
the CPU wheel first and then the project, so no CUDA packages are pulled in.

## D-005 - Optional native stacks behind capability probes (2026-09-19)

TensorRT, CUDA, Docker, and tritonclient are probed once in `utils/env.py` and imported lazily. Missing
capability raises `EnvironmentUnavailableError` (exit 3) at use, and `trtship doctor` reports it. The
CLI itself must always start. No silent GPU->CPU fallback anywhere.

## D-006 - Input specs are declared, not inferred (2026-09-19)

`nn.Module.forward` carries no shape/dtype information, so input specs come from config (name, dtype,
shape with symbolic dims). A real forward pass verifies them and derives output specs. Rejected
alternative: guessing from the first example input, which cannot express dynamic dims.

## D-007 - Safe model loading by default (2026-09-19)

`torch.load(weights_only=True)`; TorchScript and full-pickle loads require `model.trust_source: true`
because they execute arbitrary code.

## D-008 - `config.pbtxt` generated from an `EngineInfo` value object (2026-09-19)

Separates GPU-only extraction of engine metadata from CPU-testable text generation, while keeping the
config derived from the real engine rather than hardcoded.

## D-009 - Docker driven via CLI argv, not the Docker SDK (2026-09-19)

Fewer dependencies, identical behavior to what documentation tells users to run, and argv lists avoid
shell-injection. Trade-off: parsing CLI output; mitigated by using `--format json` where available.

## D-010 - No AI attribution; brief kept out of the repo (2026-09-19)

Per the project brief, no source, docs, metadata, or commit message mentions an AI tool. The brief
(`project.md`) is a local working document and is excluded through `.git/info/exclude`, not committed.

## D-011 - Path semantics in configuration (2026-09-19)

Input paths (weights, datasets) resolve against the config file's directory; output paths (run root,
cache, model repository, timing cache) resolve against the current working directory. Resolving
outputs against the config directory would put `runs/` inside `configs/examples/`. Both are made
absolute during validation, so a stored config snapshot is unambiguous. Field defaults are
validated (`validate_default=True`) so default output paths are made absolute too; without this the
defaults silently stayed relative.

## D-012 - CLI error boundary uses Typer's public API only (2026-09-19)

Typer 0.27 vendors a private copy of click (`typer._click`), so subclassing its group class would
depend on unstable internals. Commands are wrapped by `cli.guard.handle_errors`, which maps
`TrtshipError` to its exit code and any other exception to exit 70 using only `typer.Exit`,
`typer.Abort` and `typer.BadParameter`.

## D-013 - Tooling configuration (2026-09-19)

- Dev interpreter: uv-managed CPython 3.12. The pyenv 3.12.9 on the dev machine was built without
  `_sqlite3`, which crashes mypy's default cache and coverage.py. mypy also has `sqlite_cache =
  false` so it works on such interpreters.
- mypy runs on the running interpreter (3.12 in CI); numpy 2.x stubs use 3.12-only syntax, so a
  `python_version = "3.11"` pin fails. Python 3.11 compatibility is covered by the test matrix.
- Ruff `TC` (typing-only-import) rules are not enabled: they would move ~25 imports for negligible
  benefit and are a footgun with pydantic's runtime annotation evaluation.
- `ruff format` in this version reformats Markdown code fences; Markdown is excluded.

## D-014 - License: MIT (2026-09-19)

The owner's earlier README for this repository stated MIT; that is used as the default. Copyright
holder is the repository owner per the git identity. Change `LICENSE` and `pyproject.toml` if a
different license is wanted.

## D-015 - `require_gpu` checks the driver stack only (2026-09-19)

`require_gpu` requires a working NVIDIA driver/device (`nvidia-smi`). It does not require a CUDA
build of torch, because a healthy GPU machine with CPU-only torch is still able to run TensorRT.
Stages that move tensors with torch additionally call `require(TORCH_CUDA, ...)`. Whether TensorRT
stages use torch or cuda-python for device memory is decided in Phase 7.

## D-016 - TorchScript supported despite upstream deprecation (2026-09-19)

torch 2.14 emits a `FutureWarning` for `torch.jit.script`/`torch.jit.load` recommending
`torch.export`. The brief asks for TorchScript "where appropriate", so loading is supported, gated
behind `trust_source`, and documented as deprecated upstream. The warning is filtered in the test
suite only; users still see it. Revisit if a future torch removes TorchScript loading.

## D-017 - Shared tensor spec and signature inference (2026-09-19)

`DType` and `TensorSpec` live in `trtship.specs` (no torch/numpy imports) and are used by config,
model inspection, export, validation, and Triton generation, instead of separate config and runtime
types. `InputSpec` remains as a config-facing alias. `value_range` on a spec bounds generated data
so integer inputs (token ids) are safe to fabricate.

Output specs are derived by running the model at two symbol-size assignments (profile `opt`, then
profile `min` or a distinct probe size) and attributing changing dimensions to the symbol that
changed identically. This needs no static analysis of `forward` and works for any tensor-returning
model. Limits: dims determined by several symbols are named synthetically rather than expressed as
formulas, and a profile that pins a symbol (min=opt=max) makes its effect unobservable.

## D-018 - Activation memory is reported as an upper bound (2026-09-19)

Peak activation memory depends on the runtime's buffer reuse and scheduling, so it cannot be
measured honestly from PyTorch module hooks. The inspection report therefore sums every leaf
module's output size during one forward pass and labels it an upper bound, alongside the symbol
sizes it was taken at. It is `null` with an explanatory note when unmeasurable (e.g. scripted
modules, which reject forward hooks). Real memory numbers come from the benchmark stage (Phase 10),
which measures the actual engine on the actual device.

## D-019 - Two ONNX exporters, structural verification only (2026-09-19)

torch 2.14 makes the `torch.export`-based exporter the default and deprecates the TorchScript-tracing
exporter. Both work here and both are supported (`export.dynamo`). The tracing exporter stays the
default because it needs no extra dependency (the newer one needs `onnxscript`, an optional
`dynamo` extra) and its per-tensor `dynamic_axes` semantics are simple; revisit the default when
upstream removes it.

Export verifies the graph against the inferred signature, but that is a check of *declared* shapes.
A probe showed `int(x.shape[0])` inside `forward` bakes the traced batch size into the graph while
the declared output dim stays dynamic, and onnxruntime returned batch 4 for a batch-2 input. Only
running at several shapes exposes this, so the ONNX validation stage (Phase 5) must execute the
graph at min/opt/max and compare with PyTorch; `trtship_fixtures.models:baked_batch` is the
regression model for it. The documentation and error hints say so explicitly rather than implying
export success proves dynamic-shape correctness.

Export never overwrites its destination (atomic hard-link publish), which is the immutability rule
applied at the lowest level; cache reuse across identical inputs is the artifact store's job.

## D-020 - ONNX validation design (2026-09-19)

- **Multiple shape points.** The graph is run at the profile's min/opt/max (or the probe sizes),
  because D-019 showed a traced-in constant passes every declared-shape check. Verified: the
  `baked_batch` model passes at `opt` and fails at `min` (ORT cannot reshape) and `max` (wrong output
  shape).
- **ORT on the CPU provider with graph optimizations disabled**, so the exported artifact itself is
  what is validated and no other backend can be substituted silently. ORT's own error logging is
  silenced (fatal only) because failures are surfaced as `ValidationFailedError`.
- **A report is always produced.** Comparison failures are recorded in the report;
  `raise_for_failure()` converts them to exit code 6 afterwards. The CLI writes `-o` before raising
  so a failing run keeps its evidence.
- **Metric definitions are explicit** (see docs/pipeline/onnx-validation.md). "Top-k agreement" means
  the reference's top-1 class lies in the candidate's top-k, which is the decision-quality reading of
  agreement. Per-sample minimum cosine is used for pass/fail because a global cosine hides a single
  wrong sample.
- **Comparison code is shared** in `trtship.validation` so Phase 9 applies identical metrics to
  TensorRT engines.

## D-021 - Own conservative ONNX passes instead of a third-party optimizer (2026-09-19)

Considered `onnxoptimizer`, `onnx-simplifier`, and ONNX Runtime's offline optimizer. ONNX Runtime's
higher levels emit vendor-domain fused operators that TensorRT cannot parse, and the two packages
add native dependencies whose compatibility with the very new onnx used here is not something this
project can guarantee. Six small passes are implemented directly on the protobuf instead: they are
transparent, dependency-free, testable against ONNX Runtime before/after, and safe with subgraphs
(an outer-scope name read inside an `If` branch is renamed or kept correctly, which is tested).
TensorRT performs the heavy optimization. Every optimized file is interface-checked against the
model signature before it is published, and the CLI validates it numerically by default.
Trade-off: the passes do little on already-clean exports; that is accepted and documented rather
than overstated.

## D-022 - Orchestrator design (2026-09-19)

- **Scratch, then publish.** Stages never write into the run's artifact directories directly. They
  write to a scratch directory and `publish` moves files in without overwriting and registers them,
  so a crash cannot leave a half-written artifact and a retry cannot collide with one. Conflicting
  content is versioned (`name.2.ext`) rather than rejected, because a legitimate retry (for example
  a report with a new timestamp) must be possible.
- **Order from data, not lists.** `requires`/`uses`/`produces` artifact types define the graph, so
  adding the engine stages needs no change to the orchestrator.
- **Preflight over partial failure.** Every selected stage's required capabilities are checked before
  any stage runs; a CPU-only machine gets one error listing everything, plus the `--until` that
  would work, instead of doing four stages and then failing. Nothing falls back from GPU to CPU.
- **Cache keys** hash input artifact hashes, the config slice, the weights hash, the seed, and tool
  versions (plus GPU/driver for GPU stages). Measurement stages are `cacheable = False`.
- **A tampered artifact is rebuilt, not trusted.** Every use re-hashes; a mismatch makes the
  producing stage rerun and publish a new version, leaving the damaged file as evidence.
- **Resuming requires an identical config** except for the `artifacts` section (output locations),
  because results are only meaningful for the configuration that produced them.
- Real runs of the example config exposed two bugs unit tests had missed (stage order in the
  summary, and `changed` counting shape metadata as a change); both are fixed and tested.

## D-023 - `model.python_path` (2026-09-19)

Users' model code normally lives in their own repository. Rather than require `PYTHONPATH`, a config
may list directories (relative to the config file) that are added to `sys.path` before the factory is
imported. It is excluded from cache keys (machine-specific; the weights hash identifies content) and
is covered by the existing trust statement: `kind: module` runs your code by design.

## D-024 - TensorRT builder design and how it is (not) verified (2026-09-19)

- **Fake TensorRT for translation logic only.** `tests/fakes/fake_tensorrt.py` records the builder
  calls trtship makes so tests can assert workspace bytes, flags, profile shapes, network flags per
  TensorRT version, timing-cache handling, and error reporting. It is documented as a fake; it says
  nothing about whether real TensorRT accepts the calls. The real check is `tests/gpu`
  (`@pytest.mark.tensorrt`), which cannot run on the dev machine (driver mismatch, and TensorRT is
  not installed). Status files and docs therefore call the builder *unverified on hardware*.
- **The `trt` module is a parameter** of `build_engine`, so the same code path runs against the fake
  and the real module; there is no test-only branch in production code.
- **GPU buffers use torch CUDA tensors** (settles D-015). It avoids a `cuda-python` dependency and
  gives correct stream/pointer handling for free, at the cost of requiring a CUDA torch build for
  execution (engine validation and benchmarks, later phases). Building only needs the driver.
- **INT8 refusal is explicit** until the calibrate stage exists: the `build` stage raises an
  `EngineBuildError` saying so, instead of building an uncalibrated INT8 engine.
- **Preflight before run creation.** An impossible request (no GPU) exits 3 without leaving an empty
  run directory. Tests that exercise only the CPU stages now say so (`--until optimize`).
- **Engine cache keys** include the TensorRT version, GPU name/compute capability, and driver (via the
  stage's required capabilities), so plans are never reused across incompatible environments.
- **Open question, unverified:** whether current TensorRT releases support this machine's Pascal
  GPU (sm_61). NVIDIA's support-matrix page is a JavaScript explorer whose static HTML holds no
  hardware table, so this could not be settled from documentation here.

## D-025 - Calibration design (2026-09-19)

- **Calibrate by building.** TensorRT computes scales during an INT8 build, so the `calibrate`
  stage runs that build with a data-feeding calibrator, keeps the cache, and discards the engine;
  `build` rebuilds from the cache with a cache-only calibrator. The cost is one extra build; the
  benefit is that final engines never depend on calibration-time state and the cache is a plain,
  hashable artifact.
- **Reuse requires proof.** `calibration.cache_path` is a way to adopt an existing cache, and only
  after integrity and a field-by-field compatibility check (dataset fingerprint, sample count,
  batch size, method, preprocessing, input, seed, model and ONNX hashes, TensorRT version).
  Mismatches are an error that lists them; a stale cache is never used quietly. The earlier meaning
  of the field ("where to keep the cache") is dropped: shared reuse is the artifact cache's job.
- **Dataset identity is content.** Fingerprints hash file contents, not paths or mtimes, so moving a
  dataset keeps its identity and editing one file changes it. The calibrate stage's cache key
  includes the fingerprint.
- **Synthetic data is opt-in and labelled.** Refused by the config unless `allow_synthetic`, marked
  `representative: false` in metadata, and warned about in the stage result.
- **Device copies are injectable** (`DeviceBuffers`): torch CUDA tensors in production, a host
  stand-in in tests, so the calibrator protocol is tested without a GPU.
- **Bugs found by the tests along the way:** `NumpyDataset.__len__` crashed for any array
  (`array or []`), and the first draft of the build stage never passed the calibrator to
  `build_engine` (a reformat had defeated an edit); both fixed.

## D-026 - Engine validation design (2026-09-19)

- **Gate on PyTorch, report ONNX.** TensorRT is judged against PyTorch with the precision's
  tolerance; the ONNX Runtime comparison is context, since ONNX-vs-PyTorch has its own gate. This
  keeps one unambiguous pass/fail per engine.
- **The executor is a protocol at the validation boundary** (`EngineExecutor`), so tolerance gating,
  shape-point coverage, and reporting are tested with executors that reproduce, perturb, or break
  PyTorch, independent of TensorRT. The TensorRT-specific binding logic is tested separately
  against the fake context, and both are marked unverified on hardware until `tests/gpu` runs.
- **Bound executions own their buffers** (a bug caught while writing this: the first draft kept
  input buffers on the executor, so a second `bind()` would have freed the first one's). The
  execution *context* still holds tensor addresses, so one bound execution per executor at a time.
- **Layering:** `trtship.validation` (metrics) is below `trtship.onnx`, which is below
  `trtship.validation.engine`; the engine validator is therefore not re-exported from the package
  `__init__` (doing so created an import cycle).

## D-027 - Benchmark design (2026-09-19)

- **Only measurements, always labelled.** Every measurement carries its backend and device; GPU
  memory is `null` unless measured; unsupported combinations are listed as `skipped` with the reason
  (directly executed TensorRT cannot run concurrent requests) instead of being dropped or emulated.
  The repository contains no TensorRT or Triton benchmark numbers because none were measured.
- **Targets own the work, the runner owns scheduling and statistics.** `BenchmarkTarget` is a small
  protocol (ONNX Runtime, TensorRT direct, later Triton), so the loop, percentile maths, warmup and
  cold-call handling, concurrency accounting, and noise notes are tested once with injected clocks
  and exact expected values.
- **The cold call is recorded separately** (`first_call_ms`) and excluded from the statistics, in
  addition to `warmup_iters` discarded iterations.
- **Errors keep their category.** A structured trtship error raised during measurement (for example
  an engine runtime error) propagates with its own exit code and hint; anything else becomes a
  `BenchmarkError`. (Found by a test: the cold call previously escaped unwrapped.)
- **Never cached.** The benchmark stage is `cacheable = False`: it measures this machine now.
- **Comparison never overstates.** `compare` matches measurements on backend/precision/batch/
  concurrency, warns on different GPUs, tool versions, or weights, and lists unmatched measurements.

## D-028 - Triton model repository design (2026-09-19)

- **The config comes from the engine, not the user.** `config.pbtxt` is rendered from the engine's
  own tensor list and optimization profile (`EngineInfo`), so names, dtypes and batch limits cannot
  drift from the plan. `triton.max_batch_size` can only lower the engine's limit; asking for more
  is a configuration error naming both numbers.
- **Packaging needs no GPU.** `EngineInfo` is stored in the engine artifact's metadata by the build
  and next to the plan by `trtship build` (`MODEL.plan.json`), so the `package` stage and command
  run anywhere. The alternative, deserializing the plan at package time, would have made a text
  file depend on TensorRT and a GPU.
- **Triton's batch convention is enforced, not guessed.** With batching on, every tensor needs a
  dynamic leading axis (it is dropped from `dims`); otherwise packaging fails and points to
  `max_batch_size: 0`. Silent guesses would produce a repository Triton rejects at load time, on a
  machine trtship may not have.
- **Validation gates packaging.** A failed engine validation report blocks the stage (exit code of
  the validation failure); a missing or partial report only warns, because `package` must remain
  usable for engines built and checked elsewhere.
- **Repositories are immutable, like every artifact.** They are assembled beside the destination and
  moved into place, never overwrite an existing directory, and leave nothing behind on failure.
  The stage publishes the repository into the run as an artifact (never cached: it is a copy of the
  plan plus a text file); `trtship package -o DIR` writes a standalone copy.
- **Verified against Triton's schema, not a server.** Tests parse the output with
  `tritonclient`'s `model_config_pb2`, which validates syntax and field values. No Triton server
  has loaded the result. `tritonclient` is a dev dependency for this reason.

## D-029 - Triton server lifecycle design (2026-09-19)

- **Docker CLI, not a Docker SDK.** One less dependency, the same commands the user would run, and
  every call is an argv list. The runner and the HTTP transport are injected, so orchestration is
  tested without Docker; `status` and `stop` were additionally run against the real daemon.
- **Local by default.** Triton has no authentication, so ports are published on `127.0.0.1` unless
  `triton.bind_address` says otherwise. Container and address values are validated (a container name
  must be a plain Docker name), and the repository is mounted read-only.
- **No default image.** A plan loads only in the TensorRT
  version that built it, and guessing the Triton tag would fail at load time with a confusing error.
- **Never leave a half-started server, never touch someone else's container.** A failed `serve`
  captures the last container log lines into the error and removes the container it created;
  a pre-existing container of the same name makes `serve` refuse before doing anything.
- **Unknown is not "not ready".** If the server cannot be reached, `status` reports `ready: null`
  with a note. Only an actual non-200 answer is "not ready".
- **Stdlib HTTP.** Readiness and status use `urllib` (plain http only) so `serve` works without the
  optional `tritonclient`; the clients of Phase 14 are a separate layer.
- **Unverified pieces are named.** The `--gpus all` flag and the `/v2/repository/index` endpoint are
  used as their projects document them; neither has been observed against a real Triton here.

## D-030 - Triton client design (2026-09-19)

- **One wrapper over both protocols.** `tritonclient.http` and `.grpc` expose the same operations, so
  a single `TritonClient` is parameterized by the module and normalizes the differences (gRPC's
  JSON form carries shapes as strings). Everything downstream (validation, later benchmarking)
  is written once against it.
- **Only client-library and socket errors are translated.** `InferenceServerException` and
  `OSError` become `TritonError`; anything else is a bug in trtship and is allowed to surface
  instead of being disguised as a server failure.
- **Binary tensor data.** HTTP requests ask for binary outputs, so float tensors are not rounded
  through JSON and large batches are not slow for the wrong reason.
- **Validation reuses the engine-validation machinery.** A served model is just another executor
  (`TritonExecutor`), so shape points, tolerances, metrics and the report schema are shared; the
  report gained a `backend` field (`tensorrt`, `triton-http`, `triton-grpc`) so a result can never
  be mistaken for a direct-TensorRT one. The default keeps existing reports valid.
- **A protocol stub, labelled as one.** `tests/fakes/fake_triton.py` implements the KServe v2
  endpoints over real HTTP and gRPC sockets so the real `tritonclient` code paths run in CI without a
  GPU. It serves ordinary Python functions; no claim about Triton's behavior rests on it.

## D-031 - Triton benchmarking design (2026-09-19)

- **The server's own statistics, not inference.** The client sees one opaque request time. The
  queue and compute breakdown comes from Triton's cumulative statistics read before and after the
  timed section, so warmup and the cold call are excluded and the figures are the server's clock.
  The client + network share is derived by subtraction and labelled as derived.
- **A target capability, not a special case.** `ServerSideObserver` is an optional protocol on a
  benchmark target (`timed_started` / `timed_finished`); `measure()` calls it around exactly the
  timed section. Other targets are unaffected and report no server-side times.
- **One thread per worker for its whole life.** `tritonclient`'s HTTP client runs on gevent and
  cannot be used from a thread other than the one that created it. `measure()` therefore warms up
  and times each worker on one thread, with a barrier between the phases whose action starts the
  server-side snapshot and the clock. Each thread opens its own client. Found by running concurrent
  workers against the protocol stub; the earlier two-batch design would have failed the same way
  against a real server.
- **Overhead only where the work is identical.** Direct engines run one request at a time, so
  served rows are paired with them only at concurrency 1. Other rows are reported as not compared.
- **Mismatched request counts are surfaced.** If the server counts a different number of requests
  than were sent, a note says the means describe a different population instead of silently
  reporting them.

## D-032 - Example models (2026-09-19)

- **ResNet-50 uses torchvision, as an optional extra** (`trtship[examples]`), with random weights
  from `build()`. Committing or downloading weights would break "no weights in the repo" and make
  the example depend on the network; real weights load through `kind: checkpoint`. The extra is
  separate because torchvision must match the installed torch build.
- **The BERT-style model is written in plain PyTorch**, not taken from `transformers`. It avoids a
  heavy dependency and a network download, exports through the default exporter, and still has the
  properties the pipeline must handle: several integer inputs and two dynamic axes. It is labelled
  BERT-*style*: it is not BERT and loads no BERT weights.
- **Calibration data is never shipped.** The ResNet config names a directory the user fills in, and
  a test asserts it is absent, so the example cannot silently calibrate on synthetic data.
- **INT8 is not enabled for the transformer example**; see docs/examples.md.
- **Tests exercise the CPU stages only** and export (not validate) ResNet-50 to keep the suite fast.

## D-033 - Docker images and Compose files (2026-09-19)

- **Two targets in one Dockerfile.** `dev` is CPU-only and reproduces `make check`, so it can be
  built and verified anywhere; `runtime` is `FROM` a configurable NGC PyTorch image that already has
  a CUDA torch and TensorRT. trtship does not install torch or TensorRT itself in that image: pip
  would replace the vendor build, and the TensorRT version is a deployment decision, not ours.
- **The runtime image and Triton share a release tag by documented convention, not by default.**
  The Triton compose file has no default image (`${TRITON_IMAGE:?}`), matching `triton.image`
  (D-029): a plan loads only in the TensorRT that built it.
- **Serving and tooling are separate Compose files.** One file with a required variable would make
  even `docker compose run dev` fail when it is unset.
- **Compose mirrors `trtship serve`** (command, read-only mount, loopback ports), and a contract test
  compares the two so they cannot drift.
- **`trtship serve` is not run inside the runtime container**; it drives the Docker CLI, which the
  image deliberately lacks (no Docker socket is mounted into containers).

