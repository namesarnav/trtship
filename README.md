# trtship

A reproducible deployment pipeline that takes a trained PyTorch model to a benchmarked, calibrated,
Triton-servable TensorRT engine.

```
PyTorch -> validate -> ONNX -> optimize -> TensorRT (FP32/FP16/INT8) -> numerical validation
        -> benchmark -> Triton model repository -> Triton server -> deployment validation -> report
```

Every stage is modular, independently runnable, and records what it produced (content hashes, tool
versions, configuration) in a self-describing run directory.

## Status

trtship is under active development. This table is the source of truth for what works today;
`IMPLEMENTATION_STATUS.md` has the detail, including what has and has not been verified on real
GPU hardware.

| Area | State |
|---|---|
| Package, configuration, structured errors, logging | implemented, tested |
| Environment detection (`trtship doctor`) | implemented, tested |
| Run directories, run IDs, manifest, environment/config snapshots | implemented, tested |
| Model loading (module / checkpoint / TorchScript), signature inference | implemented, tested |
| Model inspection reports (`trtship inspect`) | implemented, tested |
| ONNX export (`trtship export`) | implemented, tested |
| ONNX validation (`trtship validate onnx`) | implemented, tested |
| ONNX optimization (`trtship optimize`) | implemented, tested |
| TensorRT engine builder (`trtship build`, `build` stage) | implemented against the real API; unit-tested with a fake; **not yet verified on hardware** |
| INT8 calibration (`trtship calibrate`, `calibrate` stage) | data path and cache implemented and tested on CPU; TensorRT calibrator **not yet verified on hardware** |
| Engine executor and numerical validation (`validate engine`, `validate_engine` stage) | implemented; logic tested with stand-in executors and a fake TensorRT; **not yet verified on hardware** |
| Benchmarking (`trtship benchmark onnx/engine/compare`, `benchmark` stage), benchmark reports | ONNX Runtime CPU backend real and runnable; TensorRT backend implemented, tested with a fake, **not yet verified on hardware** |
| Triton model repository (`trtship package`, `package` stage) | implemented; `config.pbtxt` validated against Triton's protobuf schema, **not yet loaded by a Triton server** |
| Triton server management, clients | planned |
| Pipeline orchestration (`trtship run`, `report`, `init`), artifact store, run caching and resume | implemented, tested (CPU stages) |
| Remaining stages in `trtship run` (package, serve) | planned |

Nothing labelled *planned* exists yet; commands for it are not registered. TensorRT- and
Triton-dependent stages need an NVIDIA environment. They are implemented against the real APIs, and
are only claimed to work once they have run on real hardware (see Limitations).

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:namesarnav/trtship.git
cd trtship
uv sync                       # GPU machines: PyTorch from PyPI (CUDA build on Linux)
uv sync --extra trt           # add the TensorRT Python bindings (NVIDIA GPU environment)
uv sync --extra triton        # add the Triton HTTP/gRPC clients
uv sync --extra dynamo        # add onnxscript for the torch.export-based ONNX exporter
```

Machines without a GPU can avoid the multi-GB CUDA wheels with the CPU-only PyTorch build:

```bash
make install-cpu
```

## Quickstart

```bash
trtship version
trtship doctor                          # what can this machine run?
trtship doctor --require tensorrt       # exit code 3 unless TensorRT is usable
trtship init                            # write a starter config
trtship config validate my-config.yaml  # validate a config; --set key=value to override
trtship run configs/examples/custom_model.yaml --until optimize   # the CPU stages; a full run needs a GPU
trtship report                          # summarize the latest run
trtship build cfg.yaml model.onnx -o engines/   # TensorRT engines (needs a GPU; see docs/pipeline/tensorrt.md)
trtship package cfg.yaml engines/m.fp16.plan -o model_repository   # Triton repository (no GPU needed)
trtship inspect my-config.yaml          # parameters, memory estimate, I/O signature, module tree
trtship export my-config.yaml -o model.onnx   # ONNX export, verified against the model signature
trtship validate onnx my-config.yaml model.onnx   # graph checks + PyTorch-vs-ONNX at min/opt/max shapes
trtship optimize my-config.yaml model.onnx -o model.opt.onnx   # safe graph cleanup, then validated
```

`trtship doctor` reports Python, PyTorch (and whether it can use CUDA), the NVIDIA driver and GPUs,
the CUDA toolkit, TensorRT, ONNX, ONNX Runtime, Docker (and its NVIDIA runtime), and the Triton
client, and summarizes which parts of the pipeline are ready on the current machine. Failures are
reported with their real cause, for example a driver/library mismatch, rather than as a bare
"no GPU".

### Configuration

A trtship config is a strictly validated YAML file (unknown keys are rejected, and all errors are
reported at once):

```yaml
model:
  name: tiny
  kind: module                      # module | checkpoint | torchscript
  factory: my_package.models:build  # callable returning an nn.Module
  inputs:
    - name: x
      dtype: float32
      shape: [batch, 16]            # string dims are dynamic, integer dims are static
tensorrt:
  precisions: [fp32, fp16]
  profiles:                         # required for dynamic inputs
    - inputs:
        x: {min: [1, 16], opt: [4, 16], max: [8, 16]}
```

- Input paths (weights, datasets) resolve relative to the config file; output paths (`runs/`, cache,
  model repository) resolve relative to the working directory.
- Precedence: defaults < YAML < `TRTSHIP__SECTION__KEY` environment variables < `--set key=value`.
- INT8 requires a `calibration` section. Synthetic calibration data is refused unless
  `allow_synthetic: true` is set explicitly, because it produces unreliable INT8 scales.
- `model.trust_source` is validated today. The model loader that enforces it (weights-only loading
  by default; `true` required for formats that execute code on load) is planned.

### Run directories

```
runs/2026-09-19_ab12cd/
  config.yaml        resolved configuration snapshot
  manifest.json      run id, per-stage status, artifact records, config hash
  environment.json   tool versions, GPU/driver info, git commit
  artifacts/  validation/  benchmarks/  reports/  logs/
```

### Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 2 | invalid configuration or usage |
| 3 | a required capability (GPU, TensorRT, Docker, ...) is unavailable |
| 4-11 | model, export, validation, engine build, calibration, benchmark, Triton, artifact failures |
| 70 | unexpected error (a bug); a traceback is printed |

## Documentation

- [Quickstart](docs/getting-started/quickstart.md), [configuration reference](docs/configuration.md)
- Pipeline: [overview](docs/pipeline/overview.md), [models](docs/pipeline/models.md),
  [inspection](docs/pipeline/inspection.md), [export](docs/pipeline/export.md),
  [ONNX validation](docs/pipeline/onnx-validation.md), [optimization](docs/pipeline/optimization.md),
  [TensorRT](docs/pipeline/tensorrt.md), [INT8 calibration](docs/calibration/int8.md),
  [engine validation](docs/pipeline/engine-validation.md),
  [benchmarking](docs/benchmarking/methodology.md),
  [Triton model repository](docs/triton/model-repository.md)

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the stage contracts, artifact model, error model,
GPU/CPU boundary, and reproducibility strategy, and [DECISIONS.md](DECISIONS.md) for the reasoning
behind the main choices.

## Development

```bash
make install-cpu     # or: uv sync
make check           # ruff lint, format check, mypy --strict, CPU tests
make test-cov        # with coverage
```

Tests that need a GPU, TensorRT, Docker, or Triton carry markers (`gpu`, `tensorrt`, `docker`,
`triton`) and are skipped with an explicit reason when the hardware is absent. A skipped GPU test is
not a passed GPU test.

## Limitations

- The TensorRT, INT8 calibration, and Triton stages have not yet been executed on real hardware.
  They will be marked as verified only after they have been.
- A TensorRT plan can only be loaded by the TensorRT version that built it. The Triton image used for
  serving must ship the same TensorRT version; there is intentionally no default image.

## License

MIT. See [LICENSE](LICENSE).
