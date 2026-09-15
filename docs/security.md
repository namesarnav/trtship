# Security model

trtship loads models, parses graph files, runs external programs and can start a network service.
This page states what it treats as trusted, what it guards, and what it leaves to you.

## Trust boundaries

| Input | Treated as | Consequence |
|---|---|---|
| The YAML config | trusted, like a Makefile | `model.factory` and `model.python_path` import and run Python you name. Do not run a config you have not read. |
| A `module` factory | code you wrote | Runs with your privileges when the model loads. |
| A `checkpoint` file | untrusted by default | Loaded with `torch.load(weights_only=True)`, which cannot execute code. `model.trust_source: true` is required for anything else. |
| A TorchScript archive | code | Can carry executable content, so it always requires `model.trust_source: true`; a warning is logged when it is loaded. |
| An ONNX file given on the command line | untrusted input to a parser | It is parsed by `onnx`, ONNX Runtime and TensorRT, which have had memory-safety bugs. Keep those libraries current and do not feed them files you do not trust. |
| A run directory or artifact from another run | hash-verified | Artifacts are checked against their recorded SHA-256 before reuse, and record paths that resolve outside the run directory are rejected. |
| A TensorRT plan | trusted | Plans are deserialized by TensorRT; load only plans you built. |

## Guarded

- **Shell injection.** External commands (`docker`, `nvidia-smi`, `git`) are run as argument lists,
  never through a shell.
- **Option injection.** `triton.image` must be an image reference and cannot begin with `-`, so it
  cannot become a `docker run` flag. `triton.container_name` and `triton.bind_address` are validated.
  A repository path containing `:` is refused because it would change the meaning of the volume spec.
- **Path traversal.** Artifact names and run IDs cannot contain path separators or `..`; artifact
  records are resolved and must stay inside the run directory. Model names, which name directories in
  the Triton repository, are restricted to letters, digits, `_`, `.` and `-`.
- **Environment leakage.** Only tool versions and device information are written to
  `environment.json`; process environment variables are not recorded, so secrets in them are not
  copied into run directories or reports.
- **Weights, engines and calibration data** are never committed, packaged into images (see
  `.dockerignore`) or shipped with the examples.

## The Triton server

- Triton has **no authentication**. Ports are published on `127.0.0.1` by default; setting
  `triton.bind_address` to `0.0.0.0` (or `BIND_ADDRESS` in the Compose file) exposes model
  inference, and Triton's management endpoints, to that network. Put an authenticating proxy or a
  firewall in front of anything reachable by others.
- The model repository is mounted read-only and the server runs with `--model-control-mode=none`, so
  clients cannot load or unload models. The container is not privileged.
- trtship's own HTTP client accepts only plain `http://` URLs built from the configured address.
- Docker itself is a privileged interface: anyone who can run `docker run` on the host can gain
  root. `trtship serve` needs that access and does not mount the Docker socket into any container.

## Not covered

trtship does not sandbox model code, scan ONNX files, sign or encrypt artifacts, or authenticate
Triton clients. Dependency integrity is whatever `uv.lock` provides.
