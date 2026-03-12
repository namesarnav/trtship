# Models

trtship accepts three kinds of PyTorch model, selected by `model.kind`.

| kind | what it loads | required fields | code execution |
|---|---|---|---|
| `module` | calls `factory(**factory_kwargs)`, which must return an `nn.Module` | `factory` | runs the factory (your Python code) |
| `checkpoint` | builds the module from `factory`, then loads a state dict from `path` | `factory`, `path` | none by default (`weights_only=True`) |
| `torchscript` | `torch.jit.load(path)` | `path`, `trust_source: true` | archives can carry executable content |

A `factory` is written `package.module:callable` and must be importable in the environment running
trtship (installed, or on `PYTHONPATH`).

## Trust model

- `kind: module` runs code from your environment by design. Only run configs you trust.
- Checkpoints load with `torch.load(weights_only=True)`. A file that needs full unpickling is
  refused with an error that explains how to opt in. `model.trust_source: true` enables it, and
  should only be used for files you trust, since unpickling can execute arbitrary code.
- TorchScript archives always require `trust_source: true`.
- TorchScript is deprecated upstream in favour of `torch.export`. It still works and is supported
  here, but new models should prefer `module`/`checkpoint`.

Checkpoint files may hold a bare state dict or one wrapped under `state_dict`, `model_state_dict`, or
`model`. A `module.` key prefix left by `DataParallel`/DDP is stripped. Keys must match the model
exactly: missing or unexpected keys, and shape mismatches, fail with the offending keys listed.

## Inputs

Inputs are declared, not inferred, because an `nn.Module` does not say what it accepts:

```yaml
model:
  name: bert-tiny
  kind: module
  factory: my_package.models:build
  inputs:
    - {name: input_ids,      dtype: int64, shape: [batch, seq], value_range: [0, 30522]}
    - {name: attention_mask, dtype: int64, shape: [batch, seq], value_range: [0, 2]}
  output_names: [logits, hidden]
```

- String dimensions are symbolic (dynamic); integer dimensions are static. The same symbol used in
  several inputs means the same size.
- Inputs are passed to `forward` positionally in the order listed, so that order must match
  `forward`'s parameters.
- `value_range: [lo, hi)` bounds generated example data (uniform floats, or integers). It matters
  for integer inputs: token ids outside the vocabulary crash an embedding lookup. Without it,
  floats are standard normal and integers are drawn from `[0, 2)`.
- Supported dtypes: `float32`, `float16`, `int64`, `int32`, `int8`, `uint8`, `bool`.

Example data is deterministic: it depends only on the seed, the input name, and the resolved shape,
so adding or reordering inputs never changes another input's values.

## Outputs and signature inference

`forward` may return a tensor, a tuple/list of tensors, or a dict of tensors (entries that are
`None` are dropped). Output names are `output_names` if given, otherwise `output`, `output_<i>`, or
the dict keys.

Output shapes are derived by running the model at two different assignments of the symbolic
dimensions and attributing every output dimension that changed to the symbol whose size changed the
same way. The sizes come from `tensorrt.profiles[0]` (`opt`, and `min` as the second point). Symbols
the profile does not constrain use distinct probe sizes (2, 3, 5, ...), which may be invalid for
your model; declare a profile to choose valid ones. An output dimension that no single symbol
explains, such as `batch * seq`, is kept dynamic under a synthetic name (`<output>_dim<axis>`).

Rejected, with an explanation: scalar outputs, outputs whose rank or dtype changes with input size,
non-tensor outputs, and unsupported dtypes such as `float64`.

## Identity

Each loaded model carries `weights_sha256`, a hash of parameter names, dtypes, shapes, and raw
bytes. It identifies the weights independent of the file format they came from: the same weights
loaded from a checkpoint and from a TorchScript archive hash identically.

## Devices

Models load on CPU by default. Requesting `cuda` without a usable GPU raises
`EnvironmentUnavailableError` (exit code 3); it never falls back to CPU.
