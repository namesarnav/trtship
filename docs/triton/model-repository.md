# Triton model repository

`trtship package` and the `package` pipeline stage turn a built TensorRT engine into a Triton model
repository. Packaging reads the engine's metadata, not the engine, so it needs no GPU and no
TensorRT installation.

```text
<repository>/
  <model name>/
    config.pbtxt
    <version>/model.plan
```

The model name is `model.name` from the configuration; the version directory is
`triton.model_version` (default 1).

## What goes into `config.pbtxt`

Nothing about the model is written by hand. Tensor names, data types and shapes come from the
engine (`EngineInfo`, recorded by the build), and the batch limit comes from the optimization
profile the engine was built with.

| Field | Source |
|---|---|
| `platform: "tensorrt_plan"`, `default_model_filename: "model.plan"` | fixed |
| `max_batch_size` | the largest batch every input's profile allows, or `triton.max_batch_size` if set (it may only lower it) |
| `input` / `output` | the engine's tensors, in order, with Triton data types |
| `dims` | the shape without its leading axis when batching is on; the full shape (`-1` for dynamic axes) when `max_batch_size` is 0 |
| `instance_group` | `triton.instance_count` instances of kind `KIND_GPU` on `triton.instance_gpus` |
| `dynamic_batching` | only if `triton.dynamic_batching` is set (`preferred_batch_sizes`, `max_queue_delay_us`) |

Supported data types: float32, float16, int64, int32, int8, uint8 and bool.

Triton's batching rule is that the leading axis of every tensor is the batch axis. If an output does
not have a dynamic leading axis, packaging fails and says so; set `triton.max_batch_size: 0` to
serve such an engine without Triton batching.

## Configuration

```yaml
triton:
  repository_dir: model_repository   # default target of `trtship package`
  model_version: 1
  precision: fp16                    # required when the run built several precisions
  max_batch_size: null               # null: take the engine's limit; 0: disable batching
  instance_count: 1
  instance_gpus: [0]
  dynamic_batching:
    preferred_batch_sizes: [4, 8]
    max_queue_delay_us: 100
```

`triton.precision` must be one of `tensorrt.precisions`. `dynamic_batching` needs a batching engine,
and its preferred sizes may not exceed the batch limit. Both are reported as configuration errors
with the limits involved.

## In a pipeline run

The `package` stage runs after `build` (and after `validate_engine` if that stage is selected). It
publishes the repository as the `triton_repository` artifact of the run (`artifacts/model_repository`)
and records the engine it copied as a parent.

- If the run contains an engine validation report and the report failed, the stage refuses to
  package and exits with the validation-failure code. Fix the accuracy problem first.
- If there is no engine validation report, or it did not cover the chosen precision, the stage
  packages the engine and records a warning.
- With several built precisions and no `triton.precision`, the stage fails and lists the choices.

## From the command line

```bash
trtship build config.yaml model.onnx -o engines      # also writes engines/<model>.<precision>.plan.json
trtship package config.yaml engines/m.fp16.plan -o model_repository
```

`build` writes the engine description next to each plan (`MODEL.plan.json`); `package` reads it. For
an engine built elsewhere, pass `--engine-info FILE`, where the file is either an `EngineInfo` or a
build result containing one. `package` refuses to write into an existing directory: repositories are
immutable, so choose a new path.

## Verification status

The generated `config.pbtxt` is parsed by Triton's own protobuf schema (`tritonclient`) in the test
suite, which proves it is syntactically valid and carries the intended values. It has **not** been
loaded by a Triton server, because no GPU or Triton container is available in the development
environment. A plan also loads only in the TensorRT version that built it, so use the Triton
release that ships the same TensorRT.
