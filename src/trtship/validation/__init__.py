"""Numerical comparison of model outputs across backends.

The metrics here are a low-level layer used by ``trtship.onnx``. Engine validation
(``trtship.validation.engine``) builds on ``trtship.onnx`` and is therefore imported from its own
module, not re-exported here.
"""

from trtship.validation.metrics import (
    TensorComparison,
    TensorStats,
    compare_outputs,
    compare_tensors,
    worst_comparison,
)

__all__ = [
    "TensorComparison",
    "TensorStats",
    "compare_outputs",
    "compare_tensors",
    "worst_comparison",
]
