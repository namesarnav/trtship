# Pipeline overview

```
inspect -> export -> validate -> optimize -> calibrate -> build -> validate_engine -> benchmark
                                                                           \-> package
```

Every stage is also a command that works on explicit file paths (`inspect`, `export`,
`validate onnx`, `optimize`, `calibrate`, `build`, `validate engine`, `benchmark ...`, `package`).
`calibrate` only runs when `int8` is in `tensorrt.precisions`. Serving is not a stage: `trtship
serve`, `status` and `stop` manage the Triton container, and `validate triton` and `benchmark
triton` measure it once it is running. The engine stages need a GPU and TensorRT; see the
[capability preflight](#capability-preflight).

## Stages and ordering

A stage declares the artifact types it **requires**, optionally **uses**, and **produces**, and the
orchestrator orders stages from those declarations, not from a fixed list. A stage runs after every
stage that produces something it needs. `--from X`, `--only X` and `--until X` select a slice of that
order; `--dry-run` prints the plan.

## Run directories

```
runs/2026-09-19_ab12cd/
  config.yaml        resolved configuration snapshot
  manifest.json      status, per-stage records, artifact records
  environment.json   tool versions, GPU/driver info, git commit
  artifacts/  validation/  benchmarks/  reports/  logs/
```

`manifest.json` records, per stage, its status (`pending`, `running`, `succeeded`, `failed`,
`skipped`, `cached`), timings, cache key, produced artifact ids, metrics, and any error. Every
artifact record has its type, path (relative to the run), sha256, size, timestamp, producing stage,
parent artifacts, and metadata. `logs/trtship.jsonl` holds the structured log with `run_id` and
`stage` on every line.

## Artifacts are immutable

- Stages write to a scratch directory and **publish** into the run only on success, moving files
  into place without overwriting. A crashed stage leaves nothing half-written.
- Publishing content that differs from an existing file of the same name writes `name.2.ext`,
  `name.3.ext`, and so on; identical content is reused. Nothing is ever overwritten.
- Registered artifacts are re-hashed whenever they are used. A modified or missing artifact is
  detected, and the stage that produced it is rebuilt as a new version (the damaged file is left as
  evidence and `trtship report` flags it).

## Caching and resume

A stage's cache key hashes its inputs' artifact hashes, the config slice it depends on, the model's
weights hash, the seed, and the tool versions (plus GPU and driver for GPU stages). Within a run, a
stage with an unchanged key and intact outputs is skipped (`up to date`). Across runs, the same key
is served from the shared cache (`artifacts.cache_dir`, disable with `artifacts.reuse_cache: false`)
after re-verifying hashes, and reported as `cached`. Stages that measure the environment
(benchmarks) are never cached.

To continue a run, pass `--run RUN_DIR`. The configuration must match the run's snapshot in every
section except `artifacts`; otherwise the command refuses and says which sections differ.

## Failures

A failing stage is recorded in the manifest with its error (message, hint, details, exit code), the
run is marked `failed`, later stages do not run, and the command exits with the error's exit code.
Evidence produced before the failure is kept: a failed validation still publishes its report; an
optimized model that fails validation is never registered. After fixing the cause, re-run with the
same `--run`; completed stages are skipped.

## Capability preflight

Before any stage runs, the orchestrator checks that every selected stage's required capabilities
(GPU, TensorRT, Docker, ...) are available. If not, it fails immediately, listing every problem and,
where possible, the `--until` value that would run only what this machine supports. There is no
silent fallback from GPU work to CPU.

## Exit codes

| code | meaning |
|---|---|
| 0 | success |
| 1 | an external command (for example `docker`) failed |
| 2 | invalid configuration or usage (including a config that differs from the run's snapshot) |
| 3 | a required capability is unavailable |
| 4 | model loading or inspection failed |
| 5 | ONNX export or optimization failed |
| 6 | a validation check failed |
| 7 | TensorRT engine build, load or execution failed |
| 8 | INT8 calibration failed or its data was invalid |
| 9 | a benchmark could not be run |
| 10 | Triton repository generation, server lifecycle or client failure |
| 11 | artifact conflict, missing or corrupt artifact, an existing run id, or a filesystem error |
| 70 | unexpected error (a bug) |
