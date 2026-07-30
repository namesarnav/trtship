# Running Triton

`trtship serve`, `trtship status` and `trtship stop` manage one Triton Inference Server container
through the Docker CLI.

```bash
trtship package config.yaml engines/m.fp16.plan -o model_repository
trtship serve config.yaml            # start, then wait until the server and every model are ready
trtship status config.yaml           # container state, readiness, per-model state, endpoints
trtship stop config.yaml             # stop and remove the container (a no-op if there is none)
```

All three accept `--set section.key=value` and `--json`. `serve` also takes `--repository DIR`
(default `triton.repository_dir`).

## Requirements

- Docker with the NVIDIA container runtime (`trtship doctor` reports both).
- `triton.image`. There is deliberately no default: a TensorRT plan loads only in the TensorRT version
  that built it, so pick the `nvcr.io/nvidia/tritonserver:<xx.yy>-py3` release whose TensorRT version
  matches `trtship doctor` / the engine's recorded `tensorrt_version`.

## What `serve` runs

```text
docker run --detach --name <container_name> --gpus all
  --publish <bind_address>:<http_port>:8000 --publish <bind_address>:<grpc_port>:8001
  --publish <bind_address>:<metrics_port>:8002
  --volume <repository>:/models:ro
  <image> tritonserver --model-repository=/models --strict-model-config=true
                       --model-control-mode=none
```

- Ports are published on `triton.bind_address`, which defaults to `127.0.0.1`. Set it to `0.0.0.0`
  deliberately if the server must be reachable from other machines; Triton has no authentication.
- The repository is mounted read-only.
- `--model-control-mode=none` loads exactly the models present at startup and forbids remote
  load/unload; `--strict-model-config=true` makes Triton use the generated `config.pbtxt` as is.
- Every command is an argument list. Nothing is interpolated into a shell.

## Readiness

After starting the container, `serve` polls until `GET /v2/health/ready` and
`GET /v2/models/<name>/ready` return 200 for every model in the repository, or until
`triton.startup_timeout_s` passes. While waiting it also checks the container is still running.

If the container exits or the timeout passes, `serve` captures the container's last 60 log lines
into the error (Triton reports why a model failed to load there), removes the container, and exits
with the Triton error code (10). A failed `serve` therefore never leaves a half-started server
holding the ports. An existing container with the same name is never touched: `serve` refuses to
start and asks for `trtship stop`.

## Status

`status` reports Docker's container state and, for a running container, `/v2/health/ready` and the
per-model states from `POST /v2/repository/index`. If the server does not answer, the fields are
reported as unknown (not "not ready") with a note explaining why.

## Verification status

The orchestration (argument construction, readiness polling, failure cleanup, status parsing) is
tested with a scripted Docker runner and fake HTTP responses; the HTTP transport is tested against a
local stub server; `status` and `stop` were also exercised against a real Docker daemon with a
stand-in container. A **real Triton container has not been started**: the development machine has no
usable GPU, no NVIDIA container runtime, and no Triton image. In particular the model-index endpoint
and the Docker GPU flag are used as documented by their projects but unverified here.
