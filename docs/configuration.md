# Configuration reference

A trtship config is a YAML file, validated strictly: unknown keys are errors and all problems are
reported together. `trtship config validate FILE` checks a file; `trtship config schema` prints the
JSON Schema (also kept in `configs/schemas/trtship.schema.json`, which a test keeps in sync with the
models) for editor validation.

## Paths and overrides

- Input paths (`model.path`, `model.python_path`, `calibration.path`) are relative to the **config
  file**. Output paths (`artifacts.root`, `artifacts.cache_dir`, `tensorrt.timing_cache_path`) are
  relative to the **working directory**. Both are made absolute during validation.
- Precedence, lowest to highest: defaults < YAML < `TRTSHIP__SECTION__KEY=value` environment
  variables < `--set section.key=value` on the command line. Values are parsed as YAML, so
  `--set optimize.passes=[eliminate_identity]` works.

## Top level

| key | default | meaning |
|---|---|---|
| `schema_version` | `1` | config format version |
| `seed` | `0` | seeds example inputs and other trtship-controlled randomness |

## `model` (required)

| key | default | meaning |
|---|---|---|
| `name` | required | letters, digits, `_`, `.`, `-`; used in file and repository names |
| `kind` | required | `module`, `checkpoint`, or `torchscript` |
| `factory` | | `package.module:callable` returning an `nn.Module` (`module`, `checkpoint`) |
| `factory_kwargs` | `{}` | keyword arguments for the factory |
| `python_path` | `[]` | directories added to `sys.path` before importing the factory |
| `path` | | weights (`checkpoint`) or archive (`torchscript`) |
| `trust_source` | `false` | allow formats that execute code when loaded |
| `inputs` | required | list of `{name, dtype, shape, value_range}`; a string dimension is dynamic |
| `output_names` | model's own | names for outputs (positional for tuples/tensors) |

`kind: module` needs `factory` and no `path`; `checkpoint` needs both; `torchscript` needs `path` and
`trust_source: true` and no `factory`. See [models](pipeline/models.md).

## `export`

| key | default | meaning |
|---|---|---|
| `opset` | `17` | ONNX opset, 11 to 23 |
| `constant_folding` | `true` | let the exporter fold constants |
| `dynamo` | `false` | use the `torch.export`-based exporter (needs `uv sync --extra dynamo`) |

## `optimize`

| key | default | meaning |
|---|---|---|
| `enabled` | `true` | run the stage in a pipeline (the `optimize` command always runs) |
| `passes` | all | subset of `extract_constants`, `eliminate_identity`, `deduplicate_initializers`, `eliminate_dead_nodes`, `eliminate_unused_initializers`, `infer_shapes` |

## `tensorrt`

| key | default | meaning |
|---|---|---|
| `precisions` | `[fp32]` | any of `fp32`, `fp16`, `int8` (`int8` requires `calibration`) |
| `workspace_mb` | `4096` | builder workspace limit |
| `optimization_level` | `3` | TensorRT builder optimization level, 0 to 5 |
| `profiles` | `[]` | optimization profiles: `- inputs: {name: {min: [...], opt: [...], max: [...]}}` |
| `timing_cache_path` | | where to keep the builder timing cache |
| `device_index` | `0` | GPU to build on |

A dynamic input needs a profile entry. Profile ranks must match the declared input, static
dimensions must not vary, and `min <= opt <= max` per axis. The **first** profile also chooses the
shapes used to trace the export (`opt`) and to validate it (`min`, `opt`, `max`). `config validate`
warns about a dynamic input with no profile.

## `calibration` (required when `precisions` includes `int8`)

| key | default | meaning |
|---|---|---|
| `dataset` | required | `images`, `numpy`, or `synthetic` |
| `path`, `glob` | | data location and file pattern (not for `synthetic`) |
| `input_name` | the only input | which model input the data feeds |
| `num_samples` | `512` | samples to calibrate with |
| `batch_size` | `8` | calibration batch size |
| `method` | `entropy2` | `entropy2` or `minmax` |
| `preprocessing` | see below | `resize`, `center_crop`, `rescale` (1/255), `mean`/`std` (given together), `channel_order` |
| `seed` | top-level seed | selects samples deterministically |
| `cache_path` | | where to keep the calibration cache |
| `allow_synthetic` | `false` | `synthetic` data is rejected unless this is `true`, because it gives unreliable INT8 scales |

## `validation`

| key | default | meaning |
|---|---|---|
| `num_samples` | `8` | random inputs compared at each shape point |
| `seed` | top-level seed | seed for those inputs |
| `onnx_tolerance` | `atol 1e-4, rtol 1e-3, cosine_min 0.99999` | PyTorch vs ONNX Runtime |
| `tolerances` | per precision | `fp32`, `fp16`, `int8` tolerances for engine validation; unspecified precisions keep their defaults |

A tolerance is `{atol, rtol, cosine_min, top1_agreement_min, topk, topk_agreement_min}`; the last
three are optional and apply to `[samples, classes]` outputs. The defaults are starting points, not
guarantees. See [ONNX validation](pipeline/onnx-validation.md).

## `benchmark`, `triton`, `artifacts`

`benchmark` (`warmup_iters` 50, `iters` 500, `batch_sizes` `[1]`, `concurrency` `[1]`,
`precisions`, `seed`) and `triton` (`repository_dir`, `model_version`, `precision`,
`max_batch_size`, `instance_count`, `instance_gpus`, `dynamic_batching`, `image`, `container_name`,
ports, `startup_timeout_s`) are validated now and used by the stages that are not implemented yet.
There is deliberately no default Triton `image`: a TensorRT plan loads only in the TensorRT version
that built it, so choose the Triton release that ships yours.

`artifacts` holds `root` (default `runs`), `cache_dir` (default `.trtship-cache`) and `reuse_cache`
(default `true`).
