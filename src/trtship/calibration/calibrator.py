"""TensorRT INT8 calibrators.

Two calibrators are built dynamically from the TensorRT module handed in (so the code runs against
a fake in unit tests): one that feeds batches and collects the cache TensorRT computes, and one that
only serves an existing cache. Device memory is provided by a :class:`DeviceBuffers` so the copy to
the GPU is swappable; the real one uses torch CUDA tensors.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from trtship.errors import CalibrationError, EnvironmentUnavailableError


class DeviceBuffers(Protocol):
    def upload(self, array: npt.NDArray[Any]) -> tuple[object, int]:
        """Copy ``array`` to the device; return ``(keepalive, device pointer)``. The keepalive
        object must stay referenced while TensorRT reads the pointer."""


class TorchDeviceBuffers:
    """Device buffers backed by torch CUDA tensors."""

    def __init__(self) -> None:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            raise EnvironmentUnavailableError(
                "INT8 calibration needs a CUDA build of PyTorch to stage batches on the GPU",
                hint="Install a CUDA build of PyTorch (a plain `uv sync` on a GPU machine).",
            )
        self._torch = torch

    def upload(self, array: npt.NDArray[Any]) -> tuple[object, int]:
        tensor = self._torch.from_numpy(np.ascontiguousarray(array)).cuda()
        return tensor, int(tensor.data_ptr())


def _base(trt: Any, method: str) -> Any:
    if method == "entropy2":
        return trt.IInt8EntropyCalibrator2
    if method == "minmax":
        return trt.IInt8MinMaxCalibrator
    raise CalibrationError(f"unknown calibration method {method!r}")


def make_calibrator(
    trt: Any,
    method: str,
    *,
    batches: Iterator[npt.NDArray[Any]],
    batch_size: int,
    input_name: str,
    other_inputs: Mapping[str, npt.NDArray[Any]],
    device: DeviceBuffers,
) -> Any:
    """A calibrator that feeds ``batches`` for ``input_name`` and constant data for other inputs.

    After TensorRT finishes, ``calibrator.cache_bytes`` holds the computed cache (``None`` if
    TensorRT never wrote one) and ``calibrator.batches_served`` the number of batches consumed.
    """

    class Calibrator(_base(trt, method)):  # type: ignore[misc]
        def __init__(self) -> None:
            _base(trt, method).__init__(self)
            self._batches = batches
            self._keepalive: list[object] = []
            self.cache_bytes: bytes | None = None
            self.batches_served = 0

        def get_batch_size(self) -> int:
            return batch_size

        def get_batch(self, names: Sequence[str]) -> list[int] | None:
            array = next(self._batches, None)
            if array is None:
                return None
            self._keepalive = []
            pointers = []
            for name in names:
                data = array if name == input_name else other_inputs.get(name)
                if data is None:
                    raise CalibrationError(
                        f"TensorRT asked for calibration data for input {name!r}, which has none"
                    )
                keepalive, pointer = device.upload(data)
                self._keepalive.append(keepalive)
                pointers.append(pointer)
            self.batches_served += 1
            return pointers

        def read_calibration_cache(self) -> bytes | None:
            return None  # calibrate from the data; reuse is decided by trtship, with metadata

        def write_calibration_cache(self, cache: Any) -> None:
            self.cache_bytes = bytes(cache)

    return Calibrator()


def make_cache_calibrator(trt: Any, method: str, cache: bytes) -> Any:
    """A calibrator that only serves ``cache`` (no data), for building with known scales."""

    class CacheCalibrator(_base(trt, method)):  # type: ignore[misc]
        def __init__(self) -> None:
            _base(trt, method).__init__(self)

        def get_batch_size(self) -> int:
            return 1

        def get_batch(self, names: Sequence[str]) -> list[int] | None:
            return None  # no calibration data: TensorRT must use the cache

        def read_calibration_cache(self) -> bytes:
            return cache

        def write_calibration_cache(self, cache: Any) -> None:
            return None

    return CacheCalibrator()
