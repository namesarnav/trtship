# Triton clients and deployment validation

`trtship.triton.TritonClient` wraps the official `tritonclient` HTTP and gRPC clients behind one
interface. `tritonclient` is optional: `uv sync --extra triton`.

```python
from trtship.config import load_config
from trtship.triton import TritonClient

config = load_config("config.yaml")
with TritonClient.from_settings(config.triton, "grpc") as client:
    client.is_ready()
    client.model_ready("m")
    client.model_metadata("m").inputs          # names, Triton datatypes, shapes (-1 = dynamic)
    outputs = client.infer("m", {"x": array})  # {name: numpy array}
```

- The HTTP and gRPC clients return the same values; gRPC's string-encoded shapes are converted.
- Requests use binary tensor data over HTTP, so large arrays are not routed through JSON.
- Every failure (connection refused, unknown model, an inference error, a timeout) becomes a
  `TritonError` (exit code 10) that names the protocol, URL and the server's own message.
- `wait_until_ready(client, model, timeout_s)` polls until the server and the model are ready.

## `trtship validate triton`

```bash
trtship serve config.yaml
trtship validate triton config.yaml --onnx model.onnx --precision fp16 --protocol grpc
```

The command runs the same comparison as `validate engine`: deterministic inputs at the profile's
min/opt/max shapes go through PyTorch (the reference), ONNX Runtime, and, here, the model served by
Triton. The Triton result is gated by the tolerance configured for `--precision` and exits 6 on
failure. The report is the engine validation report with `backend` set to `triton-http` or
`triton-grpc`.

`--repository` (default `triton.repository_dir`) must be the repository that is being served: the
plan inside it is hashed into the report, so the report says which engine file was validated. The
command cannot check that the running container really mounted that repository.

## Verification status

The client wrapper runs against the real `tritonclient` library over real HTTP and gRPC sockets, but
talking to a **stub** that implements the KServe v2 protocol (`tests/fakes/fake_triton.py`). That
proves trtship uses the client libraries correctly and handles their results and errors. It does not
prove anything about Triton itself: request scheduling, dynamic batching, TensorRT model loading, or
the accuracy of a served engine. A run against a real Triton server needs a GPU and is listed in the
status file as blocked by the environment.
