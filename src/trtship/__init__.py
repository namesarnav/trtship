"""trtship: reproducible PyTorch -> ONNX -> TensorRT -> Triton deployment pipeline."""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("trtship")
except _metadata.PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = ["__version__"]
