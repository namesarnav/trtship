# Example models

Three examples ship with trtship. Each is a config in `configs/examples/` plus a model factory in
`examples/`. All three run the CPU stages (`inspect`, `export`, `validate`, `optimize`) on any
machine; the TensorRT, Triton and benchmark stages need the GPU environment described in
[the TensorRT guide](pipeline/tensorrt.md).

| Example | Config | Shows | Extra install |
|---|---|---|---|
| `custom_model` | `configs/examples/custom_model.yaml` | The smallest end-to-end run: a 5.5k-parameter CNN, one dynamic batch axis. Used by the CI end-to-end tests. | none |
| `resnet50` | `configs/examples/resnet50.yaml` | Image classification through FP32, FP16 and INT8, with an image calibration dataset. | `pip install "trtship[examples]"` (torchvision) |
| `bert_style` | `configs/examples/bert_style.yaml` | Three integer inputs, dynamic batch **and** sequence length, one optimization profile covering all inputs. | none (plain PyTorch) |

```bash
trtship config validate configs/examples/bert_style.yaml
trtship run configs/examples/bert_style.yaml --until optimize      # CPU only
trtship run configs/examples/bert_style.yaml                       # needs a GPU and TensorRT
```

## ResNet-50

`examples/resnet50/model.py` builds torchvision's `resnet50` with deterministic random weights and
**does not download anything**. That is enough to exercise export, engine building, benchmarking and
serving, where the architecture determines the cost. It says nothing about accuracy: with random
weights the FP16 and INT8 outputs agree with FP32 only in the numerical-error sense, and the INT8
scales are meaningless. To deploy a trained model, load your weights instead:

```yaml
model:
  kind: checkpoint
  path: weights/resnet50.pth        # a state_dict saved with torch.save
  factory: resnet50.model:build
```

INT8 needs representative images. The config points `calibration.path` at `data/calibration/`
(relative to the config file), which is **not shipped**: put a few hundred JPEG or PNG files there.
The preprocessing block (resize 256, center-crop 224, ImageNet mean and std) must match what you
train and serve with. Synthetic calibration data is refused unless `allow_synthetic: true` is set
explicitly; see [INT8 calibration](calibration/int8.md).

The profile allows batch 1 to 32 at 224x224 (`opt` 8). The Triton `max_batch_size` is derived from
the engine, so it is 32.

## BERT-style classifier

`examples/bert_style/model.py` is a 2-layer transformer encoder written in plain PyTorch, so the
example needs no `transformers` dependency and exports through the standard TorchScript exporter
(opset 17). Its interface is what makes transformer deployment different from image models:

- `input_ids`, `attention_mask` and `token_type_ids` are all `int64` with shape `[batch, seq]`. Give
  `input_ids` a `value_range` inside the vocabulary; random ids outside it would index past the
  embedding table.
- Both axes are dynamic, and TensorRT needs a `min`/`opt`/`max` range for **every** input. The
  profile in the config gives all three inputs the same range: batch 1 to 16, sequence 8 to 384.
- The attention mask is applied as an additive bias, and a test checks that masked positions do not
  influence the output.
- Only FP32 and FP16 are enabled. INT8 on a transformer needs text calibration data and per-layer
  care (softmax and layer norm are usually kept in higher precision), which a generic example
  cannot honestly provide.

## Verification status

The CPU stages of both new examples run in the test suite (`tests/e2e/test_examples_e2e.py`; the
ResNet-50 test needs torchvision and is skipped without it, and only exports, because validating a
25M-parameter model at three shapes takes about 40 seconds). **Not verified on hardware:** building
engines, INT8 calibration, benchmarking and serving either model; this development machine has no
usable GPU or TensorRT.
