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
- Shared `TensorSpec`/`DType` (`trtship.specs`) with `value_range` for generated data.

#### Notes
- No TensorRT, calibration, benchmark, or Triton functionality exists yet.
