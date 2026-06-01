# Quickstart

This walks through the CPU pipeline on the example model shipped in the repository. It needs no GPU.
The TensorRT, calibration, benchmark, and Triton stages are described in the README status table;
they are not part of `trtship run` yet.

## 1. Install

```bash
git clone git@github.com:namesarnav/trtship.git
cd trtship
make install-cpu        # or `uv sync` on a machine with a GPU
source .venv/bin/activate
```

## 2. Check the machine

```bash
trtship doctor
```

`doctor` lists what is installed and usable, with the real reason for anything that is not (for
example a driver/library mismatch), and summarizes which parts of the pipeline can run here.

## 3. Run the example

```bash
trtship config validate configs/examples/custom_model.yaml
trtship run configs/examples/custom_model.yaml
```

The example is a small convolutional classifier defined in `examples/custom_model/`. The run creates
`runs/<date>_<id>/` and executes:

| stage | what it does |
|---|---|
| `inspect` | parameter counts, memory estimate, input/output signature |
| `export` | PyTorch to ONNX with dynamic batch |
| `validate` | ONNX checker, then PyTorch vs ONNX Runtime at the profile's min/opt/max shapes |
| `optimize` | conservative graph cleanup, validated again |

Progress is printed per stage, followed by a summary. See what a run produced at any time:

```bash
trtship report                    # the latest run
trtship report runs/<run-id>      # a specific run
trtship report --json             # machine-readable
```

## 4. Resume and reuse

```bash
trtship run configs/examples/custom_model.yaml --run runs/<run-id>   # everything is up to date
trtship run configs/examples/custom_model.yaml --from validate --run runs/<run-id>
trtship run configs/examples/custom_model.yaml --dry-run             # show the plan only
```

A second run of the same configuration in a new run directory reuses results from the shared cache
(`.trtship-cache/`), after re-verifying their hashes. If a stage fails, the message tells you how to
resume; finished stages are not repeated.

## 5. Use your own model

```bash
trtship init trtship.yaml --name my-model --factory my_package.models:build
$EDITOR trtship.yaml      # declare your inputs, and profiles for dynamic dimensions
trtship config validate trtship.yaml
trtship inspect trtship.yaml           # look before you export
trtship run trtship.yaml
```

See [configuration](../configuration.md) for every option, and [models](../pipeline/models.md) for
how models are loaded and what is (and is not) trusted.
