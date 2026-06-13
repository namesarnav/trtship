"""Turning raw calibration samples into model-input tensors.

Preprocessing must match what the deployed model will see at inference time: scales calibrated on
differently-preprocessed data are wrong. The configuration records every step, and the cache
metadata stores it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from trtship.config import PreprocessingConfig
from trtship.errors import CalibrationError
from trtship.specs import DType, TensorSpec

_RESAMPLING_BILINEAR = 2  # PIL.Image.Resampling.BILINEAR, avoiding an import at module load


def sample_shape(spec: TensorSpec, fixed_dims: dict[str, int] | None = None) -> tuple[int, ...]:
    """One sample's shape: the input shape without its leading (batch) axis.

    Symbolic dimensions other than the batch axis need a size from ``fixed_dims``.
    """
    dims: list[int] = []
    for position, dim in enumerate(spec.shape[1:], start=1):
        if isinstance(dim, int):
            dims.append(dim)
        elif fixed_dims is not None and dim in fixed_dims:
            dims.append(fixed_dims[dim])
        else:
            raise CalibrationError(
                f"input {spec.name!r} axis {position} is dynamic ({dim!r}); calibration needs a "
                "fixed sample shape",
                hint="Calibration batches use the `opt` shape of the first optimization profile; "
                "give that dimension a size there.",
            )
    return tuple(dims)


def _image_to_array(
    image: npt.NDArray[np.generic], config: PreprocessingConfig
) -> npt.NDArray[np.float32]:
    from PIL import Image  # noqa: PLC0415

    pil = Image.fromarray(np.ascontiguousarray(image))
    if config.resize is not None:
        height, width = config.resize
        pil = pil.resize((width, height), _RESAMPLING_BILINEAR)
    if config.center_crop is not None:
        crop_h, crop_w = config.center_crop
        width, height = pil.size
        if crop_h > height or crop_w > width:
            raise CalibrationError(
                f"center_crop {config.center_crop} is larger than the image {[height, width]}",
                hint="Set preprocessing.resize to at least the crop size.",
            )
        top, left = (height - crop_h) // 2, (width - crop_w) // 2
        pil = pil.crop((left, top, left + crop_w, top + crop_h))
    return np.asarray(pil, dtype=np.float32)


class Preprocessor:
    """Converts raw samples to arrays of the model input's dtype and per-sample shape."""

    def __init__(self, config: PreprocessingConfig, spec: TensorSpec, shape: Sequence[int]) -> None:
        self.config = config
        self.spec = spec
        self.shape = tuple(shape)

    def __call__(self, raw: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
        array = self._image(raw) if self._is_image(raw) else self._tensor(raw)
        if tuple(array.shape) != self.shape:
            raise CalibrationError(
                f"preprocessed sample has shape {list(array.shape)} but input "
                f"{self.spec.name!r} expects {list(self.shape)} per sample",
                hint="Adjust preprocessing.resize/center_crop, or check the input shape.",
            )
        return array.astype(_numpy_dtype(self.spec.dtype), copy=False)

    def _is_image(self, raw: npt.NDArray[np.generic]) -> bool:
        return raw.dtype == np.uint8 and raw.ndim == 3 and raw.shape[-1] in (1, 3, 4)

    def _image(self, raw: npt.NDArray[np.generic]) -> npt.NDArray[np.float32]:
        config = self.config
        array = _image_to_array(raw[..., :3] if raw.shape[-1] == 4 else raw, config)
        if config.channel_order == "bgr":
            array = array[..., ::-1]
        array = array * np.float32(config.rescale)
        if config.mean is not None and config.std is not None:
            channels = array.shape[-1]
            if len(config.mean) not in (1, channels):
                raise CalibrationError(
                    f"preprocessing.mean has {len(config.mean)} values for {channels} channels"
                )
            mean = np.asarray(config.mean, dtype=np.float32)
            std = np.asarray(config.std, dtype=np.float32)
            array = ((array - mean) / std).astype(np.float32)
        chw = np.transpose(array, (2, 0, 1))  # HWC -> CHW
        channels_expected = self.shape[0] if len(self.shape) == 3 else None
        if channels_expected == 1:
            chw = chw.mean(axis=0, keepdims=True)  # RGB -> single channel
        return np.ascontiguousarray(chw, dtype=np.float32)

    def _tensor(self, raw: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
        return raw  # already in the model's layout; shape and dtype are checked by __call__


def _numpy_dtype(dtype: DType) -> type[np.generic]:
    return {
        DType.FLOAT32: np.float32,
        DType.FLOAT16: np.float16,
        DType.INT64: np.int64,
        DType.INT32: np.int32,
        DType.INT8: np.int8,
        DType.UINT8: np.uint8,
        DType.BOOL: np.bool_,
    }[dtype]
