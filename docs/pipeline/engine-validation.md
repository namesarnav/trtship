# Engine validation

```bash
trtship validate engine my-config.yaml build/model.fp16.plan --onnx build/model.onnx --precision fp16
trtship run my-config.yaml           # the validate_engine stage validates every engine built
```

> **Verification status.** The validation logic is tested on the CPU with stand-in executors that
> reproduce PyTorch, corrupt it, or fail on purpose. The `TensorRTExecutor` binding logic is tested
> against a labelled fake TensorRT. **Neither has run real TensorRT on a GPU**; the acceptance test
> is `tests/gpu/test_engine_validation_gpu.py` (skipped, with a reason, without a GPU).

## What is compared

For each engine (one per precision) the model runs at the optimization profile's **min, opt and max**
shapes on deterministic inputs (`validation.num_samples` per shape, seeded from `validation.seed`)
through three paths:

1. **PyTorch**: the reference.
2. **ONNX Runtime** on the CPU: the exported graph.
3. **TensorRT**: the engine.

TensorRT is compared with PyTorch against the tolerance configured for its precision
(`validation.tolerances.fp32/fp16/int8`), and that comparison decides pass or fail. The comparison
with ONNX Runtime is reported next to it for context and does not gate the result (ONNX vs PyTorch
was already validated by the ONNX stage).

The metrics are those of [ONNX validation](onnx-validation.md): absolute and relative error,
tolerance violations, whole-tensor and per-sample cosine similarity, distribution statistics, and for
`[samples, classes]` outputs top-1/top-k agreement and KL divergence. Integer and boolean outputs
must match exactly.

## Tolerances

Defaults (starting points, not guarantees): fp32 `atol 1e-4, rtol 1e-3, cosine >= 0.99999, top-1
1.0`; fp16 `atol 5e-2, rtol 5e-2, cosine >= 0.999, top-1 >= 0.99`; int8 `atol 0.5, rtol 0.1,
cosine >= 0.98, top-1 >= 0.95`. Quantized precisions are judged mainly on direction (cosine) and
decision agreement because element-wise error is dominated by quantization noise. Tune them per
model with `validation.tolerances`; unspecified precisions keep their defaults.

## Behavior

- A comparison failure is recorded in the report, which is published as evidence before the stage
  fails with exit code 6. A shape or profile the engine cannot run (for example the profile does not
  cover `max`) fails that shape point with the reason and the profile ranges.
- A missing/unloadable engine raises `EngineRuntimeError` (exit code 7): a plan loads only with the
  TensorRT version and GPU architecture that built it.
- Engines are always released, including when validation raises.
- Agreement is measured on sampled inputs; it does not measure task accuracy.

## The executor

`TensorRTExecutor` deserializes a plan, sets input shapes and tensor addresses, runs
`execute_async_v3`, and returns numpy outputs. GPU memory is managed with torch CUDA tensors
(`TorchDeviceMemory`); inputs are cast to the engine's binding dtype. Only optimization profile 0 is
used, and outputs whose size is only known after execution are rejected. `bind()` uploads inputs once
and returns an execution that can be repeated without further copies (used by the benchmarks); use
one bound execution at a time per executor, since the execution context holds the addresses.
