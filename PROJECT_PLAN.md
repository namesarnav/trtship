# Project Plan

Goal: a complete, production-quality `trtship` implementation per the phase list below. A phase is
done only when it is implemented, tested to the extent the environment allows, documented, and
reflected in `IMPLEMENTATION_STATUS.md`.

Legend for verification level: **CPU** = fully verified on the dev machine; **GPU** = requires an
NVIDIA environment, implemented against real APIs, verified only where noted in the status file.

| Phase | Name | Deliverable | Verification |
|---|---|---|---|
| 0 | Requirements & architecture | `ARCHITECTURE.md`, `DECISIONS.md`, plan, status | review |
| 1 | Project foundation | package, config, logging, errors, CLI (`version`, `doctor`, `config validate`), env detection, run IDs, metadata | CPU |
| 2 | Model abstraction | `ModelSource`, `IOSpec`, nn.Module/TorchScript/checkpoint loaders | CPU |
| 3 | Model inspection | params, memory, I/O shapes; JSON + human report | CPU |
| 4 | ONNX export | static/dynamic export, opset, names, metadata, clean failures | CPU |
| 5 | ONNX validation | checker, shape inference, ORT run, PyTorch-vs-ONNX metrics | CPU |
| 6 | ONNX optimization | non-destructive optimize, stats, hashes | CPU |
| 7 | TensorRT builder | FP32/FP16/INT8, profiles, workspace, unsupported-op report | GPU |
| 8 | INT8 calibration | dataset abstraction, preprocessing, cache, metadata | CPU (data/cache) + GPU (calibrator) |
| 9 | Numerical validation | PyTorch/ONNX/TRT metrics, per-precision tolerances | CPU + GPU |
| 10 | Benchmark engine | latency percentiles, throughput, memory, phases | CPU (stats, ORT) + GPU (TRT) |
| 11 | Benchmark reports | human + JSON + `benchmark compare` | CPU |
| 12 | Triton repository | `config.pbtxt` from `EngineInfo`, repository layout | CPU + GPU (extraction) |
| 13 | Triton server | `serve/stop/status`, health/readiness checks | GPU + Docker |
| 14 | Triton clients | HTTP/gRPC, metadata, readiness, infer | CPU (mocked transport only for unit tests) + GPU |
| 15 | Triton benchmarking | direct-TRT vs Triton decomposition | GPU |
| 16 | Reproducibility | environment.json, manifest, seeds | CPU |
| 17 | Artifact management | records, store, immutability, cache reuse | CPU |
| 18 | Pipeline orchestration | DAG, `--from/--only/--until`, resume | CPU |
| 19 | Configuration | full YAML schema, overrides, JSON Schema export | CPU |
| 20 | CLI | all commands, Rich output, exit codes | CPU |
| 21 | Testing | unit/integration/e2e/GPU/contract suites | CPU + GPU |
| 22 | Examples | ResNet-50, BERT-style, small custom model | CPU (custom) + GPU |
| 23 | Docker | dev/runtime images, compose, NVIDIA toolkit docs | Docker |
| 24 | CI/CD | lint, format, mypy, tests, coverage; separate GPU workflow | CI |
| 25 | Documentation | README + docs tree, every documented command verified | CPU |
| 26 | Final audit | code, security, perf, reproducibility, docs, full test run | all |

## Status

Phases 0-26 are implemented and the CPU-verifiable work is audited (see `IMPLEMENTATION_STATUS.md`).
The GPU-marked phases (7-10, 12-15) remain unverified on hardware; the status file lists the
commands that close that gap.

## Ordering notes

- Phases 1-6 and 16-19 are deliberately built first: they are fully verifiable here and everything
  else depends on the artifact store, stage protocol, and config.
- Phase 18 (orchestrator) is built against the `Stage` protocol early (with the CPU stages) and gains
  GPU stages as 7-15 land.
- Docs are written alongside each phase, not deferred to Phase 25; Phase 25 is consolidation and
  command verification.

## Context management

Checkpoint at roughly 150k-200k tokens of working context: finish the atomic task, run tests and
lint, update the four state documents, then re-read them before continuing.
