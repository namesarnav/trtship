# INT8 calibration

```bash
trtship calibrate my-config.yaml model.onnx -o build/calibration     # writes a cache directory
trtship run my-config.yaml                                            # calibrate, then build INT8
```

> **Verification status.** The data path (datasets, preprocessing, sampling, the cache and its
> metadata) is implemented and tested on the CPU. The TensorRT calibrator is written against the
> real API and tested with a labelled fake TensorRT; **it has not run against real TensorRT on a
> GPU**. `tests/gpu` holds the acceptance tests, which skip (with a reason) without a GPU. Calibration
> needs an NVIDIA GPU, TensorRT, and a CUDA build of PyTorch.

## Nothing is invented

The scales come from TensorRT running your data through the network; trtship never fabricates or
estimates them. Synthetic (random) data is refused unless `calibration.allow_synthetic: true`, is
flagged `representative: false` in the cache metadata and in the run summary, and produces a
warning: INT8 accuracy with scales from random data is unreliable.

## Configuration

See [configuration](../configuration.md#calibration-required-when-precisions-includes-int8). In
short: `dataset` (`images`, `numpy`, `synthetic`), `path`, `num_samples`, `batch_size`, `method`
(`entropy2` or `minmax`), `preprocessing`, `seed`, and `input_name` when the model has several inputs.

- **Images**: decoded (RGB), optionally resized (bilinear) and center-cropped, rescaled (default
  1/255), optionally normalized with `mean`/`std`, reordered to CHW, and converted to the input's
  dtype. **Use exactly the preprocessing your deployed model sees**; scales computed on differently
  preprocessed data are wrong.
- **Numpy**: a `.npy` array or `.npz` (key `data`) with samples on the first axis, or a directory of
  one `.npy` per sample. Shape and dtype must already match the input.
- **Selection is deterministic**: `num_samples` samples are chosen with a seeded generator (all of
  them, in order, if the dataset is not larger). A trailing partial batch is dropped, since
  calibrators need a fixed batch size; fewer samples than one batch is an error.
- **Multi-input models**: calibrate the input named by `input_name`; other inputs are fed
  deterministic generated values within their `value_range`.
- **Dynamic shapes**: batches use the first profile's `opt` shape for non-batch dimensions
  (`sample_shape`), and that profile is set as the calibration profile. The calibrated input's leading
  axis is the calibration batch size.

## The cache

A calibration cache is a directory holding `calibration.cache` (TensorRT's opaque scales) and
`metadata.json`, published as a `calibration_cache` artifact. The metadata records the dataset
identity (kind, name, **content fingerprint**, item count), requested and used sample counts, batch
size and count, seed, method, preprocessing, the calibrated input, TensorRT and CUDA versions, GPU,
the model weights hash, the source ONNX hash, the cache's own hash, and whether the data was
representative.

Reuse is never silent. `calibration.cache_path` adopts an existing cache only after verifying its
integrity and that dataset fingerprint, sample count, batch size, method, preprocessing, input, seed,
model and ONNX hashes, and TensorRT version all match; otherwise the run fails and lists every
mismatch. Within a pipeline, the stage's cache key includes the dataset fingerprint, so changed data
triggers recalibration and a rebuild of the INT8 engine.

## How it fits the pipeline

TensorRT calibrates while it builds an engine. The `calibrate` stage therefore runs an INT8 build with
the data-feeding calibrator, keeps the cache, and discards that engine. The `build` stage then builds
the final INT8 engine from the cache with a calibrator that serves only the cache (and no data), so
final engines never depend on calibration-time state. `calibrate` is skipped when `int8` is not in
`tensorrt.precisions`. INT8 engines are built with FP16 enabled for layers lacking an INT8 kernel
unless `tensorrt.int8_fp16_fallback: false`.

## Limits

- One calibrated input per model; dynamic non-batch dimensions must be fixed by the profile's `opt`.
- Whether an INT8 engine is accurate enough is not decided here: that is what numerical validation
  against PyTorch measures (see [engine validation](../pipeline/engine-validation.md)).
