# ONNX validation

```bash
trtship validate onnx my-config.yaml build/model.onnx
trtship validate onnx my-config.yaml build/model.onnx --json -o reports/onnx_validation.json
```

The command exits `0` when every check passes and `6` when the graph is rejected or the outputs
deviate beyond tolerance. With `-o` the JSON report is written first, so a failed run still leaves
its evidence behind (an existing report file is not overwritten without `--force`).

## What is checked

1. **Graph analysis.** `onnx.checker.check_model(full_check=True)` and strict shape inference. The
   report includes opset, IR version, producer, node count, an operator histogram, initializer
   count and size, and warnings for operator domains that are not standard ONNX (TensorRT is
   unlikely to support them).
2. **Interface.** ONNX Runtime must report exactly the model's input and output names, in order. If
   it does not, no numeric comparison is attempted.
3. **Numerical agreement with PyTorch** at several *shape points*, on deterministic inputs.

## Shape points

The graph is executed at more than one shape because a successful export does not prove the graph
handles other sizes (see [export](export.md)). Points come from the first optimization profile in
`tensorrt.profiles`: its `min`, `opt` and `max` shapes. Points with identical sizes are merged
(`min=opt`). Without a profile, the two probe sizes used for signature inference are used, and a
model with no symbolic dimensions has a single `static` point. At each point `validation.num_samples`
different random inputs (seeded from `validation.seed`, else the top-level `seed`) are compared.

A model that bakes the traced batch size into the graph passes at `opt` and fails at `min` and `max`,
either because ONNX Runtime cannot run it or because an output has the wrong shape. That is exactly
what this stage is for; `trtship_fixtures.models:baked_batch` is the regression model.

## How the runtime is used

ONNX Runtime runs on the **CPU provider only** (an explicit provider list, so a run can never
silently move to another backend) with **graph optimizations disabled**, so the exported graph itself
is what executes. The report records the onnxruntime version, providers, and that setting. The
reference is the PyTorch model on the device it was loaded on (CPU by default).

## Metrics

Computed in float64 per output tensor; the worst result across samples is reported per output.

| metric | definition |
|---|---|
| max / mean absolute error | `abs(candidate - reference)` |
| max / mean relative error | `abs(candidate - reference) / max(abs(reference), 1e-6)` |
| violations | elements with `abs(candidate - reference) > atol + rtol * abs(reference)` |
| cosine similarity | over the whole tensor, plus the minimum over samples (first axis) for rank >= 2; two zero vectors are identical, one zero vector is orthogonal |
| top-1 agreement | fraction of samples whose argmax matches (rank-2 `[samples, classes]` float outputs only) |
| top-k agreement | fraction of samples whose *reference* top-1 class is within the candidate's top-k (same outputs) |
| KL divergence | mean `KL(softmax(reference) || softmax(candidate))` (same outputs) |
| distribution stats | mean/std/min/max of reference and candidate |

Integer and boolean outputs must match exactly. Non-finite values in the candidate that the
reference does not have fail the comparison and are counted. A comparison passes only if every
applicable criterion holds.

## Tolerances

`validation.onnx_tolerance` (`atol`, `rtol`, `cosine_min`, and optionally `top1_agreement_min`,
`topk`, `topk_agreement_min`). The defaults (`atol=1e-4`, `rtol=1e-3`, `cosine_min=0.99999`) are
starting points for float32, chosen to sit well above float32 rounding differences between PyTorch
and ONNX Runtime (measured around 1e-7 on the test models) and well below a real error such as a
corrupted weight. Loosen them only when the deviation is understood.

## Limits

- Agreement is measured on the sampled inputs. It shows that the exported graph reproduces PyTorch,
  not that a model is accurate on a task; random inputs also do not exercise data-dependent branches
  the way real data can.
- Top-k, top-1 and KL apply only to rank-2 floating-point outputs.
- Very large shapes at the `max` point can be slow on the CPU; reduce `validation.num_samples` or
  the profile's `max`.
