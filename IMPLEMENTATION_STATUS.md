# Implementation Status

Last updated: 2026-09-19

## Current phase

Phase 3 - Model inspection (not started).

## Current task

Implement model inspection (architecture summary, parameter counts, trainable parameters, memory
footprint, input/output shapes and dtypes, device) with machine-readable and human-readable reports.

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

## Remaining tasks

Phases 3-26 per `PROJECT_PLAN.md`.

## Tests

- Passing: 249 (unit), `make check` green
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

Phase 2: model abstraction (`models/`), starting with the fixture models the rest of the test suite
will build on.
