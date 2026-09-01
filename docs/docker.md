# Docker

Three Docker uses, none of which need a GPU except the last two:

| What | File | Needs a GPU |
|---|---|---|
| Development image: lint, types and the CPU test suite | `docker/Dockerfile` (`dev`) | no |
| Runtime image: run the pipeline against your models | `docker/Dockerfile` (`runtime`) | yes |
| Triton Inference Server on a model repository | `docker/compose.triton.yml` | yes |

The build context is the repository root, and `.dockerignore` keeps weights, engines, run
directories and caches out of images.

## Development image

```bash
docker build -f docker/Dockerfile --target dev -t trtship-dev .
docker run --rm trtship-dev                       # make check: ruff, mypy, the CPU tests
docker compose -f docker/compose.yml run --rm dev # the same, with the source mounted
```

It is `python:3.12-slim` with the CPU-only PyTorch wheel (a plain install would pull gigabytes of
CUDA packages), the `triton` and `dynamo` extras and the dev tools. The tests marked `gpu`,
`tensorrt`, `docker` and `triton` are deselected by `make check` and are never counted as passed.

## Runtime image

```bash
docker build -f docker/Dockerfile --target runtime -t trtship .
docker run --rm --gpus all -v "$PWD:/workspace" trtship doctor
docker run --rm --gpus all -v "$PWD:/workspace" trtship run configs/examples/custom_model.yaml
```

The image starts from `RUNTIME_BASE` (default `nvcr.io/nvidia/pytorch:24.12-py3`), which already
contains a CUDA build of PyTorch and TensorRT; trtship is installed on top and runs as a non-root
user in `/workspace`. Override the base with `--build-arg RUNTIME_BASE=...`, or `RUNTIME_BASE=...`
when using Compose (`docker compose -f docker/compose.yml --profile gpu run --rm trtship doctor`).

**Choose the base deliberately.** A TensorRT plan loads only in the TensorRT version that built it.
Engines built in `nvcr.io/nvidia/pytorch:<xx.yy>-py3` are meant to be served by
`nvcr.io/nvidia/tritonserver:<xx.yy>-py3` with the same tag, so use the same `<xx.yy>` for both.
`trtship doctor` inside the container shows the TensorRT version, and every engine records it.

`trtship serve` starts Triton with the Docker CLI, which the runtime image does not contain. Run
`serve`, `status` and `stop` from the host, or use the Compose file below.

## Triton with Compose

```bash
trtship package configs/examples/custom_model.yaml model.fp16.plan -o model_repository
TRITON_IMAGE=nvcr.io/nvidia/tritonserver:<xx.yy>-py3 \
  docker compose -f docker/compose.triton.yml up
```

`TRITON_IMAGE` has no default and Compose refuses to start without it. Other settings:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_REPOSITORY` | `../model_repository` (relative to `docker/`) | repository to serve, mounted read-only |
| `BIND_ADDRESS` | `127.0.0.1` | interface for the published ports |
| `HTTP_PORT`, `GRPC_PORT`, `METRICS_PORT` | 8000, 8001, 8002 | host ports |

The command line is identical to the one `trtship serve` runs (`--strict-model-config=true
--model-control-mode=none`, repository mounted read-only), and a test keeps the two in step. Triton
has no authentication: ports are bound to loopback, and exposing them with
`BIND_ADDRESS=0.0.0.0` should be a deliberate decision. With the container up, `trtship validate
triton` and `trtship benchmark triton` work against it as they do with `trtship serve`.

## NVIDIA Container Toolkit

Containers see GPUs only through the NVIDIA Container Toolkit. On the host you need:

1. A recent NVIDIA driver. The driver must be at least as new as the CUDA version in the image
   (each NGC release notes its minimum driver).
2. Docker Engine, and the NVIDIA Container Toolkit installed from NVIDIA's package repository
   (see NVIDIA's installation guide for your distribution).
3. Docker configured to use it, then restarted:

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

4. A check that a container sees the GPU:

   ```bash
   docker run --rm --gpus all nvcr.io/nvidia/pytorch:24.12-py3 nvidia-smi
   ```

`trtship doctor` reports `docker` and `docker_nvidia_runtime` separately, and `trtship serve` refuses
to start (exit code 3) when the second is missing.

## Verification status

- **Verified** (2026-09-19): the `dev` image builds and passes `make check` (953 tests); both
  Compose files parse and their variables interpolate (`docker compose config`); the Dockerfile
  passes `docker buildx build --check`; the `runtime` steps (install, non-root user, entrypoint,
  `doctor`) work on a stand-in base image; `tests/contract/test_docker_files.py` keeps the files
  consistent.
- **Not verified, blocked by environment:** building the `runtime` image on the real NGC base
  (about 20 GB, not pulled), running anything with `--gpus`, and starting Triton from
  `docker/compose.triton.yml`. The container health check assumes `curl` exists in the Triton
  image, which is unconfirmed here. This machine has no working NVIDIA driver or container runtime.
