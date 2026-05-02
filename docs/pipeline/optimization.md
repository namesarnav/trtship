# ONNX optimization

```bash
trtship optimize my-config.yaml model.onnx -o model.optimized.onnx
trtship optimize my-config.yaml model.onnx -o out.onnx --pass extract_constants --pass eliminate_dead_nodes
trtship optimize my-config.yaml model.onnx -o out.onnx --no-validate --json
```

Optimization writes a **new** file. The source is never modified, the destination must not exist
(exit code 11 otherwise), and the file is written next to its destination and moved into place
atomically. Unless `--no-validate` is given, the optimized model is then validated against PyTorch
at the profile's min/opt/max shapes (see [ONNX validation](onnx-validation.md)); a failing
comparison exits with code 6 and the file is left in place for inspection but should not be used.

## Passes

The passes are deliberately conservative: each provably preserves behavior, and each stays correct
when the graph contains control-flow subgraphs (`If`/`Loop`/`Scan`), whose bodies can read tensors of
the enclosing graph by name. Heavier optimization (operator fusion, layout changes, precision) is
left to TensorRT, which does it with knowledge of the target GPU.

Passes always run in this order, whatever order they are listed in:

| pass | effect |
|---|---|
| `extract_constants` | `Constant` nodes holding a tensor become initializers (constants that are graph outputs, or use `value_float`-style attributes, are left alone; skipped for IR < 4) |
| `eliminate_identity` | removes `Identity` nodes and rewires their readers, including readers inside subgraphs; an `Identity` that produces a graph output is kept |
| `deduplicate_initializers` | merges initializers with identical dtype, shape and contents (initializers that are graph inputs or outputs are not merged) |
| `eliminate_dead_nodes` | removes nodes that do not contribute to any graph output; a node read only inside a subgraph is live |
| `eliminate_unused_initializers` | removes initializers nothing reads; a read from inside a subgraph counts |
| `infer_shapes` | fills in `value_info` for intermediate tensors |

Select passes with `optimize.passes` in the config or `--pass` (repeatable) on the command line;
the default is all of them. `optimize.enabled: false` disables the stage in a full pipeline run;
invoking `trtship optimize` directly always runs.

## Safety checks

- Before the optimized file is published it must pass the ONNX checker and its **interface must
  equal the model signature** (names, order, dtypes, ranks, static and dynamic dimensions). A pass
  that changes the interface fails with `ExportError` and leaves nothing behind.
- The result carries the source's hash and the pass list in the file's `metadata_props`
  (`trtship.optimized_from_sha256`, `trtship.optimize_passes`); metadata from the export is kept.

## What is recorded

The result records the source and optimized paths, hashes and sizes, the changes made by each pass,
and a graph summary before and after (node count, operator histogram, initializer count and bytes).
Running the optimizer on its own output changes nothing.

## Limits

- The passes are simple by design. On many exported models they change little (a plain MLP has
  nothing to optimize). The value is a normalized, smaller graph and a recorded, checked
  transformation, not a large speedup.
- Nodes inside subgraphs are not rewritten or removed; only top-level nodes are.
- Models over 2 GiB are not supported.
