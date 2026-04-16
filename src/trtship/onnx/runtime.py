"""ONNX Runtime execution on the CPU.

The provider list is explicit (CPU only) so a run can never silently move to another backend, and
graph optimizations are off by default so the *exported* graph is what is being executed.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from trtship.errors import ValidationFailedError

_PROVIDERS = ["CPUExecutionProvider"]


class OrtSession:
    def __init__(self, model_path: Path, *, optimize: bool = False) -> None:
        options = ort.SessionOptions()
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if optimize
            else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        options.log_severity_level = 4  # fatal only: failures are reported as ValidationFailedError
        try:
            self._session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=_PROVIDERS
            )
        except Exception as exc:
            raise ValidationFailedError(
                f"ONNX Runtime cannot load {model_path.name}: {type(exc).__name__}: {exc}",
                hint="The model may use operators or opset versions this onnxruntime build lacks.",
                details={"stage": "onnxruntime_load"},
            ) from exc
        self.optimize = optimize
        self.version: str = ort.__version__

    @property
    def providers(self) -> list[str]:
        return list(self._session.get_providers())

    @property
    def input_names(self) -> list[str]:
        return [i.name for i in self._session.get_inputs()]

    @property
    def output_names(self) -> list[str]:
        return [o.name for o in self._session.get_outputs()]

    def run(
        self, inputs: Mapping[str, npt.NDArray[np.generic]]
    ) -> dict[str, npt.NDArray[np.generic]]:
        missing = [name for name in self.input_names if name not in inputs]
        if missing:
            raise ValidationFailedError(f"missing ONNX Runtime inputs: {missing}")
        try:
            results = self._session.run(self.output_names, dict(inputs))
        except Exception as exc:
            shapes = {name: list(array.shape) for name, array in inputs.items()}
            raise ValidationFailedError(
                f"ONNX Runtime failed: {type(exc).__name__}: {exc}",
                details={"input_shapes": shapes, "stage": "onnxruntime_run"},
            ) from exc
        return dict(zip(self.output_names, results, strict=True))
