"""Calibration datasets.

A dataset yields raw samples by index and knows its own identity: a fingerprint of its content, so a
calibration cache can be tied to the exact data it was computed from. Nothing here fabricates data
except :class:`SyntheticDataset`, which is explicitly labelled non-representative.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from trtship.config import CalibrationConfig, CalibrationDatasetKind
from trtship.errors import CalibrationError
from trtship.utils.hashing import sha256_file, sha256_json

Sample = npt.NDArray[np.generic]
IMAGE_EXTENSIONS: Final = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


class DatasetIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    name: str
    fingerprint: str  # sha256 over the content (or, for synthetic data, its generating parameters)
    items: int
    representative: bool = True


class CalibrationDataset(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def load(self, index: int) -> Sample:
        """Raw sample ``index``: an image (HWC uint8) or an array in the model's input layout."""

    @abstractmethod
    def identity(self) -> DatasetIdentity: ...


def _fingerprint(entries: Sequence[tuple[str, str]]) -> str:
    """Hash of ``(relative name, content sha256)`` pairs, independent of file locations."""
    return sha256_json(sorted(entries))


class ImageFolderDataset(CalibrationDataset):
    """Images found (recursively, in sorted order) under a directory."""

    def __init__(self, root: Path, pattern: str = "*") -> None:
        if not root.is_dir():
            raise CalibrationError(
                f"calibration image directory not found: {root}",
                hint="calibration.path must be a directory of images.",
            )
        self.root = root
        self.files = sorted(
            p for p in root.rglob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.files:
            raise CalibrationError(
                f"no images matching {pattern!r} ({', '.join(IMAGE_EXTENSIONS)}) under {root}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def load(self, index: int) -> Sample:
        from PIL import Image, UnidentifiedImageError  # noqa: PLC0415 - optional-heavy import

        path = self.files[index]
        try:
            with Image.open(path) as image:
                image.load()
                return np.asarray(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as exc:
            raise CalibrationError(f"cannot read calibration image {path}: {exc}") from exc

    def identity(self) -> DatasetIdentity:
        entries = [(p.relative_to(self.root).as_posix(), sha256_file(p)) for p in self.files]
        return DatasetIdentity(
            kind="images",
            name=self.root.name,
            fingerprint=_fingerprint(entries),
            items=len(self.files),
        )


class NumpyDataset(CalibrationDataset):
    """Samples along the first axis of a ``.npy`` file, an ``.npz`` (key ``data``), or one ``.npy``
    file per sample in a directory."""

    def __init__(self, path: Path, pattern: str = "*") -> None:
        if not path.exists():
            raise CalibrationError(f"calibration data not found: {path}")
        self.path = path
        self._files: list[Path] | None = None
        self._array: npt.NDArray[np.generic] | None = None
        if path.is_dir():
            self._files = sorted(p for p in path.rglob(pattern) if p.suffix == ".npy")
            if not self._files:
                raise CalibrationError(f"no .npy files matching {pattern!r} under {path}")
        else:
            self._array = self._read_array(path)
            if self._array.ndim < 2:
                raise CalibrationError(
                    f"{path.name} must hold samples along its first axis, got shape "
                    f"{list(self._array.shape)}"
                )

    @staticmethod
    def _read_array(path: Path) -> npt.NDArray[np.generic]:
        try:
            if path.suffix == ".npz":
                with np.load(path) as archive:
                    if "data" not in archive:
                        raise CalibrationError(
                            f"{path.name} has no 'data' array; keys: {archive.files}"
                        )
                    return np.asarray(archive["data"])
            return np.asarray(np.load(path, allow_pickle=False))
        except CalibrationError:
            raise
        except (OSError, ValueError) as exc:
            raise CalibrationError(f"cannot read calibration data {path}: {exc}") from exc

    def __len__(self) -> int:
        if self._files is not None:
            return len(self._files)
        return 0 if self._array is None else len(self._array)

    def load(self, index: int) -> Sample:
        if self._files is not None:
            return self._read_array(self._files[index])
        assert self._array is not None
        return np.asarray(self._array[index])

    def identity(self) -> DatasetIdentity:
        if self._files is not None:
            entries = [(p.relative_to(self.path).as_posix(), sha256_file(p)) for p in self._files]
            fingerprint = _fingerprint(entries)
        else:
            fingerprint = _fingerprint([(self.path.name, sha256_file(self.path))])
        return DatasetIdentity(
            kind="numpy", name=self.path.name, fingerprint=fingerprint, items=len(self)
        )


class SyntheticDataset(CalibrationDataset):
    """Seeded random data of a fixed sample shape. **Not representative** of real inputs: INT8
    scales derived from it are unreliable. It exists for pipeline smoke tests and requires an
    explicit opt-in in the configuration."""

    def __init__(self, sample_shape: Sequence[int], count: int, seed: int) -> None:
        self.sample_shape = tuple(sample_shape)
        self.count = count
        self.seed = seed

    def __len__(self) -> int:
        return self.count

    def load(self, index: int) -> Sample:
        rng = np.random.default_rng([self.seed, index])
        return rng.standard_normal(self.sample_shape).astype(np.float32)

    def identity(self) -> DatasetIdentity:
        digest = hashlib.sha256(
            f"synthetic|{self.sample_shape}|{self.count}|{self.seed}".encode()
        ).hexdigest()
        return DatasetIdentity(
            kind="synthetic",
            name=f"synthetic(seed={self.seed})",
            fingerprint=digest,
            items=self.count,
            representative=False,
        )


def open_dataset(
    config: CalibrationConfig, sample_shape: Sequence[int], seed: int
) -> CalibrationDataset:
    """Create the dataset described by ``config``. ``sample_shape`` is one sample's shape (the
    model input without its batch dimension)."""
    kind = config.dataset
    if kind is CalibrationDatasetKind.SYNTHETIC:
        return SyntheticDataset(sample_shape, config.num_samples, seed)
    assert config.path is not None  # guaranteed by CalibrationConfig validation
    if kind is CalibrationDatasetKind.IMAGES:
        return ImageFolderDataset(config.path, config.glob)
    return NumpyDataset(config.path, config.glob)
