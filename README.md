# trtship

A reproducible deployment pipeline that takes a trained PyTorch model to a benchmarked, calibrated,
Triton-servable TensorRT engine.

```
PyTorch -> validate -> ONNX -> optimize -> TensorRT (FP32/FP16/INT8) -> numerical validation
        -> benchmark -> Triton model repository -> Triton server -> deployment validation -> report
```

**Contents:** [What it is](#1-what-trtship-is) · [Why](#2-why-it-exists) ·
[Architecture](#3-architecture) · [Installation](#4-installation) · [Quickstart](#5-quickstart) ·
[Pipeline](#6-the-complete-pipeline) · [Configuration](#7-configuration) ·
[Precision](#8-precision-modes) · [Calibration](#9-calibration) · [Benchmarking](#10-benchmarking) ·
[Triton](#11-triton) · [Troubleshooting](#12-troubleshooting) · [Development](#13-development) ·
[Testing](#14-testing) · [Decisions](#15-architecture-decisions) · [Limitations](#16-limitations)

## 1. What trtship is

trtship is a command-line tool and Python library. Given a PyTorch model and a YAML file describing
it, it exports and validates ONNX, builds TensorRT engines at the precisions you ask for (FP32, FP16,
INT8 with calibration), checks the engines numerically against PyTorch, benchmarks them, packages
them as a Triton Inference Server model repository, starts Triton, and validates and benchmarks the
served model. Every stage is modular and independently runnable, and each one records what it
produced (content hashes, tool versions, configuration) in a self-describing run directory.

## 2. Why it exists

Getting a model onto TensorRT and Triton is a chain of steps that each fail in their own way: an
export that silently changes behaviour, an INT8 engine calibrated on the wrong data, a plan that
loads only on the TensorRT version that built it, a benchmark that measured the wrong thing. trtship
makes each step explicit, checks its output before the next step consumes it, and refuses to guess:

- **Nothing is fabricated.** Calibration never invents data, benchmarks never extrapolate, and a
  capability that is missing (no GPU, no TensorRT, no Docker runtime) stops the stage with exit code 3
  and a reason instead of falling back to something that only looks like it worked.
- **Numbers are checked.** Each artifact is compared with the PyTorch reference at the profile's
  min, opt and max shapes, against tolerances you can read and change.
- **Runs are reproducible.** Seeds, tool versions, GPU and driver, the configuration and content
  hashes are recorded, and stage caching keys include all of them.

## 3. Architecture

Stages exchange files through an artifact store, and a stage that finds its inputs unchanged is
skipped. See [ARCHITECTURE.md](ARCHITECTURE.md) for the stage contracts, artifact model, error
model, GPU/CPU boundary and reproducibility strategy.

| Stage | Produces | Needs a GPU |
|---|---|---|
| `inspect` | parameter counts, memory estimate, I/O signature | no |
| `export` | ONNX model, checked against the signature | no |
| `validate` | PyTorch vs ONNX Runtime report | no |
| `optimize` | cleaned-up ONNX, validated again | no |
| `calibrate` | INT8 calibration cache | yes |
| `build` | TensorRT plans and their descriptions | yes |
| `validate_engine` | PyTorch vs engine report, per precision | yes |
| `benchmark` | latency and throughput reports | ONNX baseline no, TensorRT yes |
| `package` | Triton model repository | no |

`serve`, `stop`, `status`, `validate triton` and `benchmark triton` are separate commands that act on
a running server (Docker, the NVIDIA runtime and a GPU).

## 4. Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:namesarnav/trtship.git
cd trtship
uv sync                       # GPU machines: PyTorch from PyPI (CUDA build on Linux)
uv sync --extra trt           # add the TensorRT Python bindings (NVIDIA GPU environment)
uv sync --extra triton        # add the Triton HTTP/gRPC clients
uv sync --extra dynamo        # add onnxscript for the torch.export-based ONNX exporter
uv sync --extra examples      # add torchvision for the ResNet-50 example
```

Machines without a GPU can avoid the multi-GB CUDA wheels with the CPU-only PyTorch build:

```bash
make install-cpu
```

Docker images (a CPU development image and a GPU runtime image) are described in
[docs/docker.md](docs/docker.md).

### Compatibility

- **TensorRT.** 8.6 or newer is required; the builder is written against the TensorRT 10 Python API.
  A plan loads only in the TensorRT version, on the GPU architecture, that built it, and stage caches
  are keyed on both, so an engine is never reused across them. The Triton image used for serving
  must ship the same TensorRT version (there is intentionally no default image).
- **CUDA and driver.** trtship does not pin a CUDA version. The CUDA build of PyTorch, the TensorRT
  wheel and the driver must agree; the driver must be at least as new as the CUDA runtime it serves.
  `trtship doctor` prints all of them and detects a driver/library mismatch.
- **GPUs.** Which GPU generations a TensorRT release supports comes from NVIDIA's support matrix,
  not from trtship. Check it for your GPU before relying on a machine.
- **Python.** 3.11 and 3.12 are tested.

### Supported model types

`model.kind` selects how the model is loaded:

| kind | source | notes |
|---|---|---|
| `module` | a callable `package.module:function` returning an `nn.Module` | the default; the example models use it |
| `checkpoint` | a `state_dict` file plus that factory | loaded weights-only unless `trust_source: true` |
| `torchscript` | a TorchScript archive | can carry executable content, so requires `trust_source: true` |

Any model whose graph exports to ONNX operators TensorRT's parser supports works. Operators it does
not support are named in the error (`details.unsupported_operators`). Examples: a small CNN,
ResNet-50 and a BERT-style encoder (see [docs/examples.md](docs/examples.md)).

## 5. Quickstart

```bash
trtship version
trtship doctor                          # what can this machine run?
trtship doctor --require tensorrt       # exit code 3 unless TensorRT is usable
trtship init                            # write a starter config
trtship config validate configs/examples/custom_model.yaml
trtship run configs/examples/custom_model.yaml --until optimize   # the CPU stages
trtship report                          # summarize the latest run
```

That runs `inspect`, `export`, `validate` and `optimize` on any machine. A full run adds the GPU
stages and needs an NVIDIA environment:

```bash
trtship run configs/examples/resnet50.yaml     # needs data/calibration for INT8; see docs/examples.md
trtship report
trtship serve configs/examples/resnet50.yaml -r runs/<run-id>/artifacts/model_repository   # needs triton.image
```

`trtship doctor` reports Python, PyTorch (and whether it can use CUDA), the NVIDIA driver and GPUs,
the CUDA toolkit, TensorRT, ONNX, ONNX Runtime, Docker (and its NVIDIA runtime), and the Triton
client, and summarizes which parts of the pipeline are ready on the current machine. Failures are
reported with their real cause, for example a driver/library mismatch, rather than as a bare
"no GPU". A longer walk-through is in [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md).

## 6. The complete pipeline

```bash
trtship inspect configs/examples/custom_model.yaml                       # parameters, memory, I/O
trtship export configs/examples/custom_model.yaml -o model.onnx          # ONNX, checked against the signature
trtship validate onnx configs/examples/custom_model.yaml model.onnx      # PyTorch vs ONNX at min/opt/max
trtship optimize configs/examples/custom_model.yaml model.onnx -o model.opt.onnx
trtship calibrate configs/examples/custom_model.yaml model.opt.onnx -o build/calibration   # INT8 only, GPU
trtship build configs/examples/custom_model.yaml model.opt.onnx -o engines   # GPU
trtship validate engine configs/examples/custom_model.yaml engines/model.fp16.plan --onnx model.onnx --precision fp16   # GPU
trtship benchmark engine configs/examples/custom_model.yaml --engine fp16:engines/model.fp16.plan   # GPU
trtship package configs/examples/custom_model.yaml engines/model.fp16.plan -o model_repository
trtship serve configs/examples/custom_model.yaml --repository model_repository       # GPU, Docker
trtship validate triton configs/examples/custom_model.yaml --onnx model.opt.onnx --precision fp16 --repository model_repository
trtship benchmark triton configs/examples/custom_model.yaml --repository model_repository
trtship stop configs/examples/custom_model.yaml
```

`trtship run` performs the first nine steps in order, resuming and caching as it goes, and leaves the
repository at `runs/<run-id>/artifacts/model_repository`; `serve`, `validate triton`, `benchmark
triton` and `stop` are run separately against it (pass it with `--repository`). Each
command is documented under [Documentation](#documentation). After a served model is up, inference
through the client library looks like this:

```python
from trtship.config import load_config
from trtship.triton import TritonClient

config = load_config("configs/examples/custom_model.yaml")
with TritonClient.from_settings(config.triton, "grpc") as client:
    outputs = client.infer(config.model.name, {"x": batch})   # {output name: numpy array}
```

## 7. Configuration

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
- `model.trust_source` controls code execution while loading: checkpoints are loaded weights-only by
  default, and TorchScript archives (which can carry executable content) require `true`.

### Dynamic shapes

An input dimension written as a string (`batch`, `seq`) is dynamic; an integer is fixed. Every
dynamic input needs an optimization profile with `min`, `opt` and `max` shapes, and a dynamic input
with no profile is a configuration error raised before TensorRT runs. The ONNX export marks the
dynamic axes, validation and engine comparison run at the min, opt and max shapes, benchmarks use
`opt` with the batch axis replaced (and refuse a batch outside the profile's range), and the Triton
`config.pbtxt` is generated from the built engine's real tensors rather than from the config. See
[docs/pipeline/tensorrt.md](docs/pipeline/tensorrt.md#optimization-profiles).

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

## 8. Precision modes

| precision | how it is built | what to check |
|---|---|---|
| `fp32` | no reduced-precision flags | the reference engine; validated against PyTorch at tight tolerances |
| `fp16` | TensorRT `FP16` flag | a warning is recorded when the GPU reports no fast FP16 |
| `int8` | TensorRT `INT8` from a calibration cache, with FP16 fallback for layers lacking an INT8 kernel (`tensorrt.int8_fp16_fallback`) | requires a `calibration` section; accuracy must be confirmed by `validate engine` |

Select them with `tensorrt.precisions`. Each has its own tolerance under `validation.tolerances`, and
the defaults are starting points, not guarantees: an INT8 engine passes only if it stays within the
tolerance you set for your task. Whether reduced precision is faster is measured, not assumed; see
[Benchmarking](#10-benchmarking).

## 9. Calibration

INT8 needs representative data to pick activation scales. trtship reads it from your dataset
(`calibration.dataset`: `images` or `numpy`, read from `calibration.path`), applies the configured
preprocessing, samples deterministically from the seed, and lets TensorRT compute the scales while
it builds. The `calibrate` stage keeps the resulting cache; `build` then builds the final INT8 engine
from the cache alone, so engines never depend on calibration-time state. The cache's metadata records the
dataset fingerprint, preprocessing, method, seed, model and ONNX hashes and TensorRT version. Reusing
an existing cache fails and lists every mismatch, and inside a pipeline run changed data triggers
recalibration and a rebuild of the INT8 engine.

Synthetic calibration data is **refused** unless `allow_synthetic: true` is set explicitly, because
it produces unreliable scales. Calibration data is never shipped with the repository. Details:
[docs/calibration/int8.md](docs/calibration/int8.md).

## 10. Benchmarking

```bash
trtship benchmark onnx configs/examples/custom_model.yaml model.onnx          # ONNX Runtime CPU baseline
trtship benchmark engine configs/examples/custom_model.yaml --engine fp16:engines/model.fp16.plan
trtship benchmark compare runs/<run-a> runs/<run-b>                            # measured deltas
trtship benchmark overhead <direct-report> <served-report>                    # serving cost
```

**Methodology.** For each (batch size, concurrency) pair, one cold call is recorded separately, then
`warmup_iters` iterations are discarded, then `iters` iterations per worker are timed. Each request
is split into preprocess, execute (GPU time from CUDA events for TensorRT), postprocess and
end-to-end wall-clock phases. Reported per phase: min, mean, stdev, p50, p90, p95, p99 and max, plus
throughput, peak CPU RSS, GPU memory and the raw samples. Inputs are seeded and deterministic.

**Reading the numbers.** Results depend on the machine, clocks, power state and other load, so
compare runs from the same machine only; `benchmark compare` warns when GPUs, tool versions or
weights differ. A measurement with fewer than 100 timed requests, or a coefficient of variation above
0.25, carries a note. A directly executed engine runs one request at a time, so concurrency above 1
is listed as *not measured* rather than faked and is measured through Triton instead. The ONNX Runtime
baseline runs on the CPU: comparing it with a GPU engine shows the benefit of moving to the GPU, not a
like-for-like framework comparison. Full method: [docs/benchmarking/methodology.md](docs/benchmarking/methodology.md).

## 11. Triton

```bash
trtship package configs/examples/custom_model.yaml engines/model.fp16.plan -o model_repository
trtship serve configs/examples/custom_model.yaml --repository model_repository
trtship status configs/examples/custom_model.yaml
trtship validate triton configs/examples/custom_model.yaml --onnx model.opt.onnx --precision fp16 --repository model_repository
trtship benchmark triton configs/examples/custom_model.yaml --repository model_repository
trtship stop configs/examples/custom_model.yaml
```

`package` writes a model repository whose `config.pbtxt` is generated from the engine and validated
against Triton's own protobuf schema. `serve` runs the Triton container you name in `triton.image`
(there is no default: use the release whose TensorRT matches the one that built the plan), waits
until the server and model are ready, and publishes ports on `127.0.0.1` by default. **Triton has no
authentication**; binding to another interface is a deliberate `triton.bind_address` decision.
`validate triton` runs the engine-validation comparison through the served model, and `benchmark
triton` measures it, including the server's own queue and compute statistics, so serving overhead can
be separated from engine time. Compose files are in [docs/docker.md](docs/docker.md).

## 12. Troubleshooting

Failures carry an exit code, the cause and a hint; `--log-level DEBUG` adds detail.

| Symptom | Cause and fix |
|---|---|
| exit 3, a required capability is unavailable | The stage needs hardware this machine lacks or cannot use. `trtship doctor` shows which part is missing and why. |
| `doctor` reports a driver/library mismatch | The loaded NVIDIA kernel module and user-space libraries differ, typically after a driver update without a reboot. Reboot or reinstall matching packages. |
| exit 2, "Extra inputs are not permitted" | A config key is misspelled or in the wrong section. Unknown keys are rejected; all errors are listed together. |
| exit 5, ONNX export failure | Try `export.dynamo: true` (needs `uv sync --extra dynamo`), simplify data-dependent control flow, or check the input signature. |
| exit 6, validation failed | The outputs differ by more than the tolerance. The report names the shape, output and metric; for INT8, recalibrate on more representative data before loosening a tolerance. |
| exit 7, unsupported operators | The ONNX parser rejected an op. The error lists them; rewrite the layer or export a different operator set. |
| exit 8, calibration | No data, too little data, or synthetic data without `allow_synthetic`. |
| an engine loads nowhere else | A plan is bound to its TensorRT version and GPU architecture. Rebuild on the target, and serve with the matching Triton image. |
| exit 10, Triton | Connection refused, model not ready or an inference error. `trtship status` shows readiness; `docker logs <container>` shows why a model failed to load. |
| exit 11, artifact or filesystem | A missing or unwritable path, or a run directory that is not one. |
| exit 70 | A bug. A traceback is printed; please report it with `trtship doctor --json`. |

## 13. Development

```bash
make install-cpu     # or: uv sync
make check           # ruff lint, format check, mypy --strict, CPU tests
make test-cov        # with coverage
make help            # all targets
```

Source is under `src/trtship`, tests under `tests/`. Code is formatted and linted with Ruff, type
checked with `mypy --strict`, and every commit should pass `make check`. Configuration models are
frozen pydantic classes; failures are `TrtshipError` subclasses with an exit code and a hint.

## 14. Testing

The suite has unit, contract, integration and end-to-end tests. TensorRT and Triton are exercised on
machines without them through clearly labelled stand-ins (`tests/fakes`): a fake TensorRT module and
a KServe v2 stub server driven by the real `tritonclient`. They prove trtship's logic and its use of
those libraries, not the behaviour of the real ones. Contract tests check that every command and
config path in the documentation exists.

Tests that need a GPU, TensorRT, Docker, or Triton carry markers (`gpu`, `tensorrt`, `docker`,
`triton`) and are skipped with an explicit reason when the hardware is absent. A skipped GPU test is
not a passed GPU test:

```bash
make test-cpu        # everything that needs no hardware
make test-gpu        # the hardware tests; fails if any were skipped
```

### Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and every pull request: lint, format check,
`mypy --strict` and a check that the configuration schema is current; the CPU tests with coverage
(floor 90%) on Python 3.11 and 3.12; and a build of the development image with `make check` inside
it. Coverage counts only CPU tests.

`.github/workflows/gpu.yml` runs the `gpu`, `tensorrt` and `triton` tests nightly and on demand. It
needs a self-hosted runner labelled `gpu` (with an NVIDIA driver, TensorRT, and Docker with the
NVIDIA runtime) and a Triton image, given as the `triton_image` input or the `TRITON_IMAGE`
repository variable. `make test-gpu` fails when any of those tests were skipped, so a runner
without the hardware cannot report success. No such runner has been available, so this workflow has
never run.

## 15. Architecture decisions

[DECISIONS.md](DECISIONS.md) records each significant choice with its reasoning and the alternatives
considered, for example how runs are cached, why there is no default Triton image, why concurrency is
not simulated for direct engines, and how CI separates hardware from CPU checks.
[ARCHITECTURE.md](ARCHITECTURE.md) describes the design, and [PROJECT_PLAN.md](PROJECT_PLAN.md) the
phases.

## 16. Limitations

- The TensorRT, INT8 calibration, Triton and Docker GPU paths have not been executed on real
  hardware: the development machine has no working NVIDIA driver, TensorRT, Triton image or NVIDIA
  container runtime. They are implemented against the real APIs and tested with stand-ins, and are
  claimed to work only once they have run on hardware. The table below is the current state.
- A TensorRT plan can only be loaded by the TensorRT version that built it. The Triton image used for
  serving must ship the same TensorRT version; there is intentionally no default image.
- Directly executed engines are measured at concurrency 1 only; concurrent load is measured through
  Triton.
- One calibrated input per model, and dynamic non-batch dimensions must be fixed by the profile's `opt`.
- Triton is started with the Docker CLI (single node, one container); Kubernetes and multi-model
  deployments are out of scope.
- Benchmark numbers are specific to the machine that produced them.

### Verification status

This table is the source of truth for what works today; `IMPLEMENTATION_STATUS.md` has the detail,
including what has and has not been verified on real GPU hardware.

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
| Triton server management (`serve`, `stop`, `status`) | implemented; orchestration tested with stand-ins, `status`/`stop` exercised against real Docker, **no real Triton container started** |
| Triton clients (HTTP/gRPC), `validate triton` | implemented; tested against a protocol stub with the real `tritonclient`, **not yet run against a real Triton server** |
| Triton benchmarking (`trtship benchmark triton`, `benchmark overhead`) | implemented; tested against a protocol stub with the real `tritonclient`, **not yet run against a real Triton server** |
| Pipeline orchestration (`trtship run`, `report`, `init`), artifact store, run caching and resume | implemented, tested (CPU stages) |
| Docker: CPU development image, GPU runtime image, Triton Compose file | `dev` image built and passes `make check`; runtime image and Compose GPU passthrough **not verified** (no NVIDIA runtime here) |

TensorRT- and Triton-dependent stages need an NVIDIA environment. They are implemented against the
real APIs, and are only claimed to work once they have run on real hardware (see Limitations).

## Documentation

- [Quickstart](docs/getting-started/quickstart.md), [configuration reference](docs/configuration.md),
  [example models](docs/examples.md), [Docker](docs/docker.md)
- Pipeline: [overview](docs/pipeline/overview.md), [models](docs/pipeline/models.md),
  [inspection](docs/pipeline/inspection.md), [export](docs/pipeline/export.md),
  [ONNX validation](docs/pipeline/onnx-validation.md), [optimization](docs/pipeline/optimization.md),
  [TensorRT](docs/pipeline/tensorrt.md), [INT8 calibration](docs/calibration/int8.md),
  [engine validation](docs/pipeline/engine-validation.md),
  [benchmarking](docs/benchmarking/methodology.md),
  [Triton model repository](docs/triton/model-repository.md), [running Triton](docs/triton/server.md),
  [clients and deployment validation](docs/triton/clients.md),
  [benchmarking a served model](docs/triton/benchmarking.md)

## License

MIT. See [LICENSE](LICENSE).
