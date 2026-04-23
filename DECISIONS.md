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
