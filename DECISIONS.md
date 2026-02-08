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
- Extra `dev`: `pytest`, `pytest-cov`, `ruff`, `mypy`, `pre-commit`, type stubs.
Torch is core because model loading/export/validation are core, CPU-testable functionality. On the
dev machine (no usable GPU) the CPU torch wheel is used to avoid multi-GB CUDA wheels; GPU installs
use the CUDA index documented in the README.

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
