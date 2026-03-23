# Model inspection

`trtship inspect CONFIG` loads the model and reports what it is, without exporting anything.

```bash
trtship inspect my-config.yaml
trtship inspect my-config.yaml --json            # machine-readable, pure JSON on stdout
trtship inspect my-config.yaml -o model_report.json   # also write the JSON (refuses to overwrite; --force)
trtship inspect my-config.yaml --depth 5 --top 20
```

## What is reported

| section | contents |
|---|---|
| Model | name, kind, root module type, module count, device, torch version, weights hash, source path |
| Parameters | total / trainable / frozen counts, number of parameter tensors, counts by dtype, buffer tensors and elements, the largest tensors (name, shape, dtype, elements, bytes) |
| Memory estimate | exact weight bytes (parameters + buffers) and an activation upper bound (below) |
| Signature | every input and output with dtype and shape; symbolic dimensions are shown by name |
| Architecture | the module tree with per-module parameter counts, to `--depth` levels (deeper modules are counted but summarized as "+N nested modules") |

Parameter counts count shared (tied) parameters once. A count for a module includes all of its
descendants.

## The memory estimate

- **Weights** are exact: element count times element size for every parameter and buffer.
- **Activations** are an *upper bound*, not a measurement of peak memory. During a real forward pass
  at the profile's `opt` shape, every leaf module's output size is summed. A runtime that reuses
  buffers (TensorRT does) will use less. The report states the symbol sizes it was measured at.
- If activations cannot be measured, the field is `null` and a note says why, and the rest of the
  report is unaffected. Scripted (TorchScript) modules are one such case: PyTorch does not support
  forward hooks on them.

## Report format

The JSON report is versioned (`schema_version`) and validated by `trtship.models.inspection.ModelReport`.
`generated_at` is the real wall-clock time (UTC). The signature block is the same
`ModelSignature` the exporter uses, so dynamic-axis information is identical everywhere.
