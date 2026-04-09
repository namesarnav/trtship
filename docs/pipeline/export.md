# ONNX export

```bash
trtship export my-config.yaml -o build/model.onnx
trtship export my-config.yaml -o build/model.onnx --set export.opset=18 --json
```

Export traces the model at the `opt` shapes of `tensorrt.profiles[0]` (or small probe sizes when
there is no profile), writes the ONNX file, verifies it against the model's signature, embeds
traceability metadata, and publishes it atomically.

## Configuration

```yaml
export:
  opset: 17               # 11..23
  constant_folding: true
  dynamo: false           # false: TorchScript-tracing exporter; true: torch.export-based exporter
```

| exporter | needs | dynamic shapes are declared with |
|---|---|---|
| `dynamo: false` (default) | nothing extra | per-tensor `dynamic_axes` (names from the model signature) |
| `dynamo: true` | `uv sync --extra dynamo` (`onnxscript`) | `torch.export.Dim` ranges from `tensorrt.profiles[0]` |

Torch has deprecated the TorchScript-based exporter ("the feature will be removed") and made the
`torch.export` exporter its default. trtship keeps the tracing exporter as its default because it
needs no extra dependency and its dynamic-axis semantics are well understood, and supports both so
you can switch (or fall back) without changing anything else. See `DECISIONS.md` D-019.

## Names and dynamic axes

Input names come from `model.inputs`; output names from `model.output_names` or the model's own
outputs (see [models](models.md)). A symbolic dimension (a string in `shape`) becomes a dynamic axis
carrying that name, for inputs and, through signature inference, outputs. With `dynamo: true`, a
symbol that the profile pins to one size (`min == opt == max`) is exported as a constant.

## Verification

After export the graph is compared with the inferred `ModelSignature`: input and output names and
order, dtypes, ranks, static dimensions, and that every symbolic dimension is dynamic (input
dimensions must also keep their symbol name). A mismatch is an `ExportError` (exit code 5) that lists
each difference.

**This is a structural check of what the graph declares.** It cannot see a dimension that tracing
turned into a constant inside the graph body while the declared shapes still say "dynamic". For
example `n = int(x.shape[0])` in `forward` bakes the traced batch size into the graph: the exported
model *declares* a dynamic batch but returns wrong shapes at any other batch size. Catching that
requires running the graph at more than one shape and comparing with PyTorch, which is what ONNX
validation does. Do not treat a successful export as proof of dynamic-shape correctness.

## Failures

Exporter exceptions become `ExportError` with the exporter, opset, trace sizes, and, for an
unsupported operator, the operator name (for example `aten::fft_rfft`) and a hint (raise
`export.opset`, switch `export.dynamo`, or replace the operator). Nothing is left behind on failure:
the temporary file is removed and the destination is never created.

## Outputs

- The destination must not exist. Export never overwrites; it refuses with exit code 11. The file is
  written next to the destination and moved into place atomically, so a crash cannot leave a partial
  model at the destination.
- The result records path, sha256, size, opset, IR version, producer, the traced sizes, the dynamic
  axes, constant-folding, and any warnings the exporter raised (for example a `TracerWarning` about
  data-dependent control flow).
- The ONNX file's `metadata_props` carry `trtship.model_name`, `trtship.weights_sha256`,
  `trtship.exporter`, and `trtship.torch_version`.
- Export is not promised to be byte-for-byte reproducible across machines or torch versions; the
  recorded sha256 identifies the artifact that was actually produced.

## Limits

- Models over 2 GiB fail with a clear error: external weight data is not supported yet.
- Only single-file, default-domain-opset models are produced; custom operator domains are not
  handled.
