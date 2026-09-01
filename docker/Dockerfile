# syntax=docker/dockerfile:1.7
#
# Two targets:
#   dev      CPU-only image that runs everything CI runs on CPU (lint, types, tests). Needs no GPU.
#              docker build -f docker/Dockerfile --target dev -t trtship-dev .
#              docker run --rm trtship-dev
#   runtime  trtship on top of an image that already ships a CUDA build of PyTorch and TensorRT.
#              docker build -f docker/Dockerfile --target runtime -t trtship .
#              docker run --rm --gpus all -v "$PWD:/workspace" trtship run configs/my-model.yaml
#            Needs the NVIDIA Container Toolkit on the host (docs/docker.md).
#
# The build context is the repository root.

ARG PYTHON_IMAGE=python:3.12-slim
# The TensorRT build must match the Triton release that will serve the engines, so choose the base
# deliberately: NGC PyTorch and Triton releases with the same <xx.yy> tag carry matching TensorRT.
ARG RUNTIME_BASE=nvcr.io/nvidia/pytorch:24.12-py3

# --------------------------------------------------------------------------- dev
FROM ${PYTHON_IMAGE} AS dev

ARG UV_VERSION=0.12.17
ARG TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends git make \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "uv==${UV_VERSION}" \
    && uv venv /opt/venv

WORKDIR /workspace

# CPU-only torch first, from its own index: a plain install would pull gigabytes of CUDA wheels.
RUN uv pip install --index-url "${TORCH_CPU_INDEX}" --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match "torch>=2.3" torchvision

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv pip install -e ".[triton,dynamo]" --group dev

COPY . .
# The Makefile looks for its tools in $VENV (default .venv); here they live in /opt/venv.
ENV VENV=/opt/venv
CMD ["make", "check"]

# --------------------------------------------------------------------------- runtime
FROM ${RUNTIME_BASE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/trtship
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# torch and tensorrt come from the base image; pip sees them as already satisfied.
RUN pip install ".[triton]" \
    && useradd --create-home --uid 1000 trtship \
    && mkdir -p /workspace \
    && chown trtship /workspace

USER trtship
WORKDIR /workspace
ENTRYPOINT ["trtship"]
CMD ["doctor"]
