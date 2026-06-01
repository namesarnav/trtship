# Implementation Status

Last updated: 2026-09-19

## Current phase

Phase 7 - TensorRT engine builder (not started).

## Current task

Implement `tensorrt/`: engine building from an ONNX file (FP32/FP16/INT8 flags, workspace,
optimization profiles for dynamic shapes, min/opt/max, input/output bindings, timing cache), engine
inspection into an `EngineInfo` value object, TensorRT-version differences, and clear
unsupported-operation reporting. Runs only where TensorRT and a supported GPU exist; everything else
(config translation, profile construction from `TensorRTConfig`, `EngineInfo`, error reporting from
parser messages) must be unit-testable with a fake `tensorrt` module. **BLOCKED BY ENVIRONMENT** for
real execution (see Environment limitations).

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
    `serve`, `stop`, `status` CLI commands; benchmark sections in `trtship report` (Phase 11).

## Remaining tasks

Phases 7-15 (TensorRT, calibration, engine validation, benchmarking, Triton), 11 (benchmark
reports), 20-26 (remaining CLI, testing suite completion, examples for ResNet/BERT, Docker, CI,
documentation consolidation, final audit) per `PROJECT_PLAN.md`.

## Tests

- Passing: see `make check` (525 after the orchestration work)
- Failing: none
- Skipped: none yet (no GPU-marked tests exist; the marker/skip machinery is in
  `tests/conftest.py` and skips with an explicit reason when no GPU/TensorRT is usable)
- Coverage: 94% (`make test-cov`, measured after Phase 1); the uncovered lines are probe branches for states this machine
  cannot produce naturally and a few error paths.

## Known bugs

None known.

## Blocked tasks

**BLOCKED BY ENVIRONMENT** (execution/verification only; implementation proceeds against the real
APIs, and nothing is marked working until it has run on real hardware):

- Phase 7 TensorRT engine build, Phase 8 calibrator execution, Phase 9 TensorRT leg,
  Phase 10 TensorRT benchmarking, Phase 12 engine-metadata extraction, Phase 13 server lifecycle,
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
- CUDA toolkit `nvcc` 12.2 is present. TensorRT and tritonclient are not installed.
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

Phase 7: TensorRT engine builder (`tensorrt/`), built against the real TensorRT API and unit-tested
with a fake module; real-hardware verification is blocked by environment.
