# Implementation Status

Last updated: 2026-09-19

## Current phase

Phase 14 - Triton clients (not started). Phases 12 and 13 are complete.

## Current task

Implement `triton/client.py`: HTTP and gRPC clients over `tritonclient` (server/model metadata,
readiness, inference with numpy tensors, structured errors), tested on CPU against a fake transport
and, where possible, a local stub server. Then Phase 15 (benchmark Triton vs direct TensorRT with a
latency decomposition) and a deployment-validation step comparing Triton outputs with PyTorch.

## Completed phases

- **Phase 0** - Requirements and architecture: `ARCHITECTURE.md`, `DECISIONS.md`,
  `PROJECT_PLAN.md`, this file. (2026-09-19)
- **Phase 1** - Project foundation. (2026-09-19)
  - Package (`src/trtship`, uv + hatchling), Ruff, mypy strict, pytest/coverage, Makefile,
    pre-commit config, README, CHANGELOG, LICENSE.
  - `errors` (exit-code model), `utils` (hashing, atomic fs, subprocess, git, env detection),
    `logging` (Rich + JSON lines with run/stage context), `config` (models, loader, overrides,
    snapshots), `artifacts` (records, run directory, manifest), CLI (`version`, `doctor`,
    `config validate`).
  - Verified: 156 unit tests, 94% line+branch coverage, `ruff check`, `ruff format --check`,
    `mypy --strict` (37 files) all pass on the dev machine.
  - Not done from the Phase 1 list: nothing outstanding. CI workflow is Phase 24; the JSON Schema
    export of the config is Phase 19.

- **Phase 2** - Model abstraction. (2026-09-19)
  - `trtship.specs` (`TensorSpec`, `DType`), `trtship.models` (loaders for module/checkpoint/
    TorchScript, `LoadedModel.run`, weight hashing, deterministic inputs, output normalization,
    signature inference), fixture models in `tests/fixtures/trtship_fixtures`, `docs/pipeline/models.md`.
  - Verified on CPU: 249 tests total (93 new), ruff, mypy --strict.
  - Model kinds exercised: MLP, two-input integer token model with dynamic sequence length and
    tuple outputs, dict outputs. Nothing assumes image classification. Not yet exercised: real
    ResNet/BERT weights (Phase 22).

- **Phase 3** - Model inspection. (2026-09-19)
  - `models/inspection.py` (`inspect_model`, `ModelReport`), `reporting/model_report.py` (Rich
    rendering + plain-text export), `trtship inspect` (`--json`, `-o/--force`, `--depth`, `--top`,
    `--set`), `docs/pipeline/inspection.md`.
  - Verified on CPU: 292 tests total (43 new), ruff, mypy --strict. Parameter counts and the
    activation bound are asserted against hand-computed values.
  - Not verified: models with very large parameter counts (hashing and inspection are streaming and
    linear, but this has not been timed on a ResNet-50/BERT-size model).

- **Phase 4** - ONNX export. (2026-09-19)
  - `export/onnx_export.py` (`export_onnx`, `check_onnx_signature`, `ExportResult`,
    `ExportMetadata`), `trtship export`, `utils.fs.publish_new`, `models.symbol_ranges`, optional
    `dynamo` extra (onnxscript), `docs/pipeline/export.md`.
  - Verified on CPU: 330 tests total (both exporters run here; the `torch.export` tests need
    `onnxscript`, installed in the dev venv, and skip with a reason without it), ruff, mypy --strict.
  - Known limitation (documented, D-019): verification is structural; a dimension frozen inside the
    graph body is not detected until Phase 5 runs the graph at several shapes.
  - Not verified: models over 2 GiB (unsupported, fails with a clear error), custom op domains.

- **Phase 5** - ONNX validation. (2026-09-19)
  - `validation/metrics.py` (shared comparison metrics), `onnx/graph.py` (checker, strict shape
    inference, statistics), `onnx/runtime.py` (CPU-only ORT), `onnx/validate.py` (multi-shape
    PyTorch-vs-ONNX comparison, `OnnxValidationReport`), `reporting/validation_report.py`,
    `trtship validate onnx`, `docs/pipeline/onnx-validation.md`.
  - Verified on CPU: 386 tests total, ruff, mypy --strict. Metrics are asserted against
    hand-computed values. The `baked_batch` model is caught exactly as predicted (passes at the
    traced size, fails at min and max).
  - Bug found and fixed by tests: `onnx.checker` raises `InferenceError` (not `ValidationError`) for
    type-inconsistent graphs; that would have escaped as exit 70 instead of 6.
  - Not verified: agreement on real trained models (only fixtures so far; ResNet/BERT in Phase 22);
    large-shape performance of the `max` point.

- **Phase 6** - ONNX optimization. (2026-09-19)
  - `onnx/optimize.py` (six passes, `optimize_onnx`, `OptimizeResult`), `trtship optimize`,
    `OptimizePass` config type, `docs/pipeline/optimization.md`.
  - Verified on CPU: 418 tests total, ruff, mypy --strict. Passes are checked by running ONNX
    Runtime before and after on hand-built graphs, including an `Identity` whose output is read
    inside an `If` subgraph; real exports are checked with the Phase 5 validation.
  - Known limits (documented): passes are conservative and do little on clean exports; only
    top-level nodes are rewritten.

- **Phases 16-19 (CPU part)** - Reproducibility, artifact management, orchestration, configuration
  tooling. (2026-09-19)
  - `artifacts/store.py` (immutable store, shared cache), `utils/seed.py`, `pipeline/`
    (orchestrator, stages: inspect/export/validate/optimize), `reporting/run_report.py`,
    commands `run`, `report`, `init`, `config schema`, `configs/schemas/trtship.schema.json`,
    `examples/custom_model` + `configs/examples/custom_model.yaml`, `model.python_path`,
    docs (quickstart, configuration, pipeline overview).
  - Verified on CPU: 525 tests total including an end-to-end run of the example config through the
    real CLI (run, report, resume, cache reuse, independent validation), ruff, mypy --strict.
  - Still to do in these phases: engine stages plug into the same orchestrator as Phases 7-15 land
    (build/calibrate/validate_engine/benchmark/package/serve); `benchmark compare`, `package`,
    `package` (Phase 12) and `serve`/`stop`/`status` (Phase 13) CLI commands; benchmark sections in `trtship report` (Phase 11).

- **Phase 7** - TensorRT engine builder: **implemented, NOT verified on hardware** (2026-09-19).
  - `tensorrt/` (`loader`, `profiles`, `engine_info`, `build`), `trtship build`, `build` pipeline
    stage, `tests/fakes/fake_tensorrt.py`, `tests/gpu/test_tensorrt_gpu.py`,
    `docs/pipeline/tensorrt.md`.
  - Verified on CPU with the fake: 582 tests total (3 GPU tests skipped with reason), ruff,
    mypy --strict. This proves trtship's translation logic and error reporting, **not** that real
    TensorRT accepts what is sent. **BLOCKED BY ENVIRONMENT**: `tests/gpu` must pass on a machine with
    a working NVIDIA driver, TensorRT, and a supported GPU before this phase is called verified.

- **Phase 8** - INT8 calibration: data path **implemented and verified on CPU**; TensorRT
  calibrator **implemented, NOT verified on hardware** (2026-09-19).
  - `calibration/` (`dataset`, `preprocess`, `sampling`, `cache`, `calibrator`, `run`),
    `trtship calibrate`, `calibrate` stage, INT8 consumption in the `build` stage,
    `tests/gpu/test_calibration_gpu.py`, `docs/calibration/int8.md`.
  - Verified: 631 tests passing, 4 skipped (GPU) at this point, ruff, mypy --strict. Datasets,
    preprocessing math, deterministic sampling, cache integrity/compatibility, and the calibrator
    protocol against the fake are all tested. **BLOCKED BY ENVIRONMENT:** real INT8 calibration.

- **Phase 9** - Engine execution and numerical validation: logic **verified on CPU** with
  stand-in executors; TensorRT executor **implemented, NOT verified on hardware** (2026-09-19).
  - `tensorrt/executor.py` (`TensorRTExecutor`, `BoundExecution`, `TorchDeviceMemory`),
    `validation/engine.py`, `validate_engine` stage, `trtship validate engine`,
    `reporting/engine_report.py`, `tests/gpu/test_engine_validation_gpu.py`,
    `docs/pipeline/engine-validation.md`.
  - Verified: 657 tests passing, 6 skipped (GPU), ruff, mypy --strict. Bugs caught by tests along the
    way are recorded in D-026. **BLOCKED BY ENVIRONMENT:** real TensorRT execution and accuracy.

- **Phases 10-11** - Benchmarking and benchmark reports: ONNX Runtime CPU backend **real and
  verified on CPU**; TensorRT backend **implemented, NOT verified on hardware** (2026-09-19).
  - `benchmark/` (`stats`, `schema`, `runner`, `targets`, `run`), `reporting/benchmark_report.py`
    (rendering, Markdown, comparison), `benchmark` stage, `trtship benchmark onnx|engine|compare`,
    `docs/benchmarking/methodology.md`.
  - Verified: 694 tests passing, 6 skipped (GPU), ruff, mypy --strict. Statistics and scheduling are
    asserted exactly with injected clocks; the ONNX Runtime backend was run for real (e.g. the example
    CNN at ~45 us per batch-1 request on this CPU) with structural assertions only.
  - **BLOCKED BY ENVIRONMENT:** TensorRT (GPU) timing and memory. No GPU numbers exist.

- **Phase 12** - Triton model repository: generation **verified on CPU** against Triton's protobuf
  schema; **not loaded by a Triton server** (2026-09-19).
  - `triton/{config,repository}.py`, `package` stage, `trtship package`, `MODEL.plan.json` written by
    `trtship build`, `docs/triton/model-repository.md`, D-028.
  - Verified: 722 tests passing, 6 skipped (GPU), ruff, mypy --strict. `config.pbtxt` files for
    dynamic, static, multi-tensor, all-dtype and batching cases are parsed by `tritonclient`'s
    `ModelConfig`. **BLOCKED BY ENVIRONMENT:** loading the repository in a Triton server.

- **Phase 13** - Triton server lifecycle: orchestration **verified on CPU** with stand-ins;
  `status`/`stop` also exercised against real Docker; **no real Triton container started**
  (2026-09-19).
  - `triton/server.py`, `serve`/`stop`/`status` commands, `triton.bind_address`,
    `docs/triton/server.md`, D-029.
  - Verified: 752 tests passing, 6 skipped (GPU), ruff, mypy --strict. **BLOCKED BY ENVIRONMENT:**
    starting Triton needs a GPU, the NVIDIA container runtime, and the Triton image.

## Remaining tasks

Phases 14-15 (Triton clients, benchmarking), 20-26 (remaining CLI, testing suite completion, examples for ResNet/BERT, Docker, CI,
documentation consolidation, final audit) per `PROJECT_PLAN.md`.

## Tests

- Passing: see `make check` (752 passed, 6 skipped after Phase 13)
- Failing: none
- Skipped: the tests in `tests/gpu` (TensorRT build, INT8 calibration), each reporting
  `no usable NVIDIA GPU: Failed to initialize NVML: Driver/library version mismatch`. Skips are not
  passes: the TensorRT builder is unverified on hardware.
- Coverage: 94% (`make test-cov`, measured after Phase 1); the uncovered lines are probe branches for states this machine
  cannot produce naturally and a few error paths.

## Known bugs

None known.

## Blocked tasks

**BLOCKED BY ENVIRONMENT** (execution/verification only; implementation proceeds against the real
APIs, and nothing is marked working until it has run on real hardware):

- Phase 7 TensorRT engine build, Phase 8 calibrator execution, Phase 9 TensorRT leg,
  Phase 10 TensorRT benchmarking, Phase 12 loading a repository in Triton, Phase 13 server lifecycle,
  Phase 15 Triton benchmarking.

## Environment limitations (dev machine, observed 2026-09-19)

- **GPU unusable right now.** NVIDIA GeForce GTX 1050 Ti Mobile (Pascal, compute capability 6.1).
  `nvidia-smi` fails with `Failed to initialize NVML: Driver/library version mismatch`: the loaded
  kernel module is 575.57.08 while the userspace NVML library is 580.178.04. Normally fixed by a
  reboot or reloading the nvidia kernel modules; not attempted from this session (needs root and
  would disrupt the display session). `trtship doctor` reports this exactly.
- Even once the driver is fixed, this GPU is Pascal. Recent TensorRT releases have narrowed the
  supported architectures; whether a current TensorRT wheel supports sm_61 must be checked against
  its release notes before relying on this machine for GPU tests. Not yet verified.
- CUDA toolkit `nvcc` 12.2 is present. TensorRT and tritonclient are not installed. NVIDIA's
  support-matrix page could not be read (JavaScript explorer), so whether current TensorRT
  supports this Pascal GPU remains unverified.
- Docker 29.2.1 is present; the NVIDIA container runtime is not configured (`docker info` lists
  only `runc`), so GPU containers and Triton-on-GPU cannot run here.
- The system pyenv Python 3.12.9 lacks `_sqlite3` (breaks mypy's default cache and coverage.py).
  The dev venv uses uv-managed CPython 3.12.14 instead (see D-013).
- Torch is the CPU wheel (2.14.0+cpu) in the dev venv; 4 CPU cores, 14 GiB RAM, ~29 GiB free disk.
- Network access to PyPI and the PyTorch wheel index works.

## Repository state

- Branch `main`, tracking `origin` (`git@github.com:namesarnav/trtship.git`), which was
  fast-forwarded to `c88b3f9` (the owner had already cleared the old contents). Local commits are
  **not pushed**; pushing is left to the owner.
- `project.md` (the brief) is intentionally untracked and excluded via `.git/info/exclude`.

## Next recommended action

Phase 14: Triton HTTP/gRPC clients. If a GPU machine becomes available
first, run `pytest -m tensorrt -v` to verify Phases 7-10.
