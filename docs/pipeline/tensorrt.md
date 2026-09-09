# TensorRT engine building

```bash
trtship build my-config.yaml build/model.onnx -o build/engines            # every configured precision
trtship build my-config.yaml build/model.onnx -o build/engines --precision fp16
trtship run my-config.yaml                                                # ... inspect -> ... -> build
```

> **Verification status.** The builder is written against the TensorRT 10 Python API (with
> compatibility for 8.6+) and its translation logic is unit-tested with a fake TensorRT module
> (`tests/fakes/fake_tensorrt.py`). **It has not yet run against real TensorRT on a GPU.** The
> acceptance tests that will do that are in `tests/gpu` (`pytest -m tensorrt`); they skip, with the
> reason, on machines without a usable GPU, and a skip is not a pass. Treat the builder as
> unverified until they have passed on your hardware.

## Requirements

- An NVIDIA GPU with a working driver (`nvidia-smi` succeeds), the TensorRT Python package
  (`uv sync --extra trt`), and a CUDA build of PyTorch (trtship uses torch tensors for GPU memory).
- TensorRT 8.6 or newer. Older versions are rejected with a clear error.
- Nothing falls back to the CPU. Without these, `build` fails with exit code 3 and says what is
  missing (`trtship doctor` shows the same information); in a pipeline run this is detected before
  anything runs (see [overview](overview.md)).

Which GPU generations a given TensorRT release supports is defined by NVIDIA's release notes and
support matrix, not by trtship. Check yours before relying on a machine.

## What a build does

1. Parse the ONNX file with TensorRT's ONNX parser. Parse failures list every parser error and the
   **unsupported operators** by name (`details.unsupported_operators`), with a hint.
2. Check that the parsed network's inputs match the model signature.
3. Configure the builder: workspace (`tensorrt.workspace_mb`), builder optimization level,
   precision flags, optimization profiles, timing cache.
4. Build the plan, deserialize it, and describe it (`EngineInfo`: I/O tensors, dtypes, shapes with
   `-1` for dynamic dimensions, and the min/opt/max range of every profile). Triton configuration is
   generated from this description, never from hardcoded values.
5. Write the plan next to its destination and publish it without overwriting.

The build result records the TensorRT version, flags, workspace, profiles, calibrator, timing-cache
use, build time, warnings, and the captured WARNING/ERROR lines of the builder log.

## Precisions

| precision | flags | notes |
|---|---|---|
| `fp32` | none | |
| `fp16` | `FP16` | a warning is recorded if the GPU reports no fast FP16 |
| `int8` | `INT8` (+ `FP16` if `int8_fp16_fallback`, default true) | needs a calibrator or a model with Q/DQ nodes |

INT8 without either is refused before TensorRT is touched. The `calibrate` stage produces the
calibration cache (see [INT8 calibration](../calibration/int8.md)); `build` then builds the INT8
engine from that cache.

## Optimization profiles

`tensorrt.profiles` supplies `min/opt/max` per dynamic input (see
[configuration](../configuration.md)); static inputs get their fixed shape automatically. A dynamic
input with no profile is a configuration error raised before TensorRT runs. TensorRT rejects an
invalid profile (for example `min > max`), which is reported with the shapes involved.

## Timing cache

Set `tensorrt.timing_cache_path` to keep the builder's timing cache across builds; it is loaded when
present and rewritten (atomically) after each build. It speeds up repeated builds and is not part of
the engine's identity.

## Compatibility

A plan can only be loaded by the same TensorRT version, on the same GPU architecture, that built it.
The `build` stage's cache key includes the TensorRT version, GPU name, compute capability, and driver,
so an engine is never reused across them. The Triton image you serve with must ship the TensorRT
version that built the plan.

Version differences handled: TensorRT 10 networks are always explicit-batch (the flag is only passed
for older versions); engines are described with the tensor API (8.5+) or the older bindings API; the
builder optimization level is only set where the builder supports it.

## Failure reporting

Failures are `EngineBuildError` (exit code 7) with structured details: parser errors and unsupported
operators, the builder log for a failed build, or the offending profile. The `build` stage records
them in the run manifest.

## Testing

```bash
pytest tests/unit/test_tensorrt.py tests/integration/test_build_stage.py   # fake TensorRT, any machine
pytest -m tensorrt -v                                                     # real TensorRT, GPU machine
```
