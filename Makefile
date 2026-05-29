VENV ?= .venv
BIN := $(VENV)/bin
CPU_TORCH_INDEX := https://download.pytorch.org/whl/cpu
CPU_MARKERS := not gpu and not tensorrt and not docker and not triton

.PHONY: help install install-cpu lint format format-check typecheck test test-cpu test-cov schema check clean

help:  ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:  ## Install with uv (GPU machines: torch comes from PyPI with CUDA)
	uv sync

install-cpu:  ## Dev install with the CPU-only torch wheel (no CUDA packages)
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(BIN)/python --index-url $(CPU_TORCH_INDEX) \
		--extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match "torch>=2.3"
	uv pip install --python $(BIN)/python -e . --group dev

lint:  ## Ruff lint
	$(BIN)/ruff check .

format:  ## Apply Ruff formatting
	$(BIN)/ruff format .

format-check:  ## Verify formatting without changing files
	$(BIN)/ruff format --check .

typecheck:  ## mypy (strict)
	$(BIN)/mypy

test:  ## Full test suite (GPU/TensorRT/Docker tests skip with a reason when unavailable)
	$(BIN)/python -m pytest

test-cpu:  ## Only tests that need no GPU, TensorRT, Docker, or Triton
	$(BIN)/python -m pytest -m "$(CPU_MARKERS)"

test-cov:  ## CPU tests with coverage
	$(BIN)/python -m pytest -m "$(CPU_MARKERS)" --cov --cov-report=term-missing

schema:  ## Regenerate configs/schemas/trtship.schema.json
	$(BIN)/python scripts/export_schema.py

check: lint format-check typecheck test-cpu  ## Everything CI runs on CPU

clean:  ## Remove caches and build output
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -name __pycache__ -type d -not -path './.venv/*' -prune -exec rm -rf {} +
