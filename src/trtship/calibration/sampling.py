"""Deterministic selection and batching of calibration samples."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from trtship.calibration.dataset import CalibrationDataset
from trtship.calibration.preprocess import Preprocessor
from trtship.errors import CalibrationError


@dataclass(frozen=True)
class Selection:
    indices: tuple[int, ...]  # sample indices, in the order they are batched
    batch_size: int

    @property
    def batches(self) -> int:
        return len(self.indices) // self.batch_size

    @property
    def sample_count(self) -> int:
        """Samples actually used: a trailing partial batch is dropped (calibrators need a fixed
        batch size)."""
        return self.batches * self.batch_size


def select_samples(dataset_size: int, requested: int, batch_size: int, seed: int) -> Selection:
    """Pick ``requested`` samples reproducibly (all, in order, if the dataset is not larger).

    The choice depends only on ``seed`` and the sizes, never on time or environment.
    """
    if dataset_size < 1:
        raise CalibrationError("the calibration dataset is empty")
    if requested >= dataset_size:
        chosen = list(range(dataset_size))
    else:
        rng = np.random.default_rng(seed)
        chosen = sorted(int(i) for i in rng.choice(dataset_size, size=requested, replace=False))
    selection = Selection(indices=tuple(chosen), batch_size=batch_size)
    if selection.batches < 1:
        raise CalibrationError(
            f"{len(chosen)} calibration sample(s) cannot fill one batch of {batch_size}",
            hint="Lower calibration.batch_size or provide more samples.",
        )
    return selection


def iter_batches(
    dataset: CalibrationDataset, selection: Selection, preprocess: Preprocessor
) -> Iterator[npt.NDArray[np.generic]]:
    """Yield ``[batch, *sample_shape]`` arrays for the selected samples."""
    for start in range(0, selection.sample_count, selection.batch_size):
        chunk = selection.indices[start : start + selection.batch_size]
        yield np.stack([preprocess(dataset.load(i)) for i in chunk])
