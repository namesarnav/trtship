from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from trtship.calibration import (
    CACHE_FILE,
    METADATA_FILE,
    CalibrationMetadata,
    ImageFolderDataset,
    NumpyDataset,
    Preprocessor,
    SyntheticDataset,
    compatibility_problems,
    iter_batches,
    open_dataset,
    read_cache_dir,
    sample_shape,
    select_samples,
    write_cache_dir,
)
from trtship.calibration.dataset import DatasetIdentity
from trtship.config import CalibrationConfig, PreprocessingConfig
from trtship.errors import CalibrationError
from trtship.specs import TensorSpec
from trtship.utils.hashing import sha256_bytes
from trtship.utils.timeutil import utc_now


def save_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


@pytest.fixture
def images(tmp_path: Path) -> Path:
    root = tmp_path / "imgs"
    save_image(root / "b.png", (0, 255, 0))
    save_image(root / "a.png", (255, 0, 0))
    save_image(root / "sub" / "c.jpg", (0, 0, 255))
    (root / "notes.txt").write_text("not an image")
    return root


# --------------------------------------------------------------------------- image datasets


def test_image_dataset_lists_sorted_images_recursively(images: Path) -> None:
    dataset = ImageFolderDataset(images)
    assert [p.name for p in dataset.files] == ["a.png", "b.png", "c.jpg"] or [
        p.relative_to(images).as_posix() for p in dataset.files
    ] == ["a.png", "b.png", "sub/c.jpg"]
    assert len(dataset) == 3
    sample = dataset.load(0)
    assert sample.shape == (8, 8, 3)
    assert sample.dtype == np.uint8
    assert tuple(sample[0, 0]) == (255, 0, 0)


def test_image_dataset_glob_and_errors(images: Path, tmp_path: Path) -> None:
    assert len(ImageFolderDataset(images, "*.png")) == 2
    with pytest.raises(CalibrationError, match="no images"):
        ImageFolderDataset(images, "*.tiff")
    with pytest.raises(CalibrationError, match="not found"):
        ImageFolderDataset(tmp_path / "missing")
    (images / "broken.png").write_bytes(b"not a png")
    broken = ImageFolderDataset(images, "broken.png")
    with pytest.raises(CalibrationError, match="cannot read calibration image"):
        broken.load(0)


def test_image_identity_follows_content_not_location(images: Path, tmp_path: Path) -> None:
    original = ImageFolderDataset(images).identity()
    assert original.kind == "images"
    assert original.items == 3
    assert original.representative

    moved = tmp_path / "elsewhere"
    images.rename(moved)
    assert ImageFolderDataset(moved).identity().fingerprint == original.fingerprint

    save_image(moved / "a.png", (254, 0, 0))  # one pixel value changes
    assert ImageFolderDataset(moved).identity().fingerprint != original.fingerprint
    save_image(moved / "a.png", (255, 0, 0))
    save_image(moved / "d.png", (1, 2, 3))  # a file is added
    assert ImageFolderDataset(moved).identity().fingerprint != original.fingerprint


# --------------------------------------------------------------------------- numpy datasets


def test_numpy_dataset_from_an_array_file(tmp_path: Path) -> None:
    data = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    np.save(tmp_path / "d.npy", data)
    dataset = NumpyDataset(tmp_path / "d.npy")
    assert len(dataset) == 4
    np.testing.assert_array_equal(dataset.load(2), data[2])
    assert dataset.identity().kind == "numpy"
    assert dataset.identity().items == 4


def test_numpy_dataset_from_npz_and_directory(tmp_path: Path) -> None:
    data = np.ones((3, 5), dtype=np.float32)
    np.savez(tmp_path / "d.npz", data=data)
    assert len(NumpyDataset(tmp_path / "d.npz")) == 3
    np.savez(tmp_path / "wrong.npz", other=data)
    with pytest.raises(CalibrationError, match="no 'data' array"):
        NumpyDataset(tmp_path / "wrong.npz")

    folder = tmp_path / "samples"
    folder.mkdir()
    for i in range(3):
        np.save(folder / f"{i}.npy", np.full((5,), i, dtype=np.float32))
    per_sample = NumpyDataset(folder)
    assert len(per_sample) == 3
    assert per_sample.load(2).tolist() == [2.0] * 5


def test_numpy_dataset_errors_and_identity(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="not found"):
        NumpyDataset(tmp_path / "nope.npy")
    np.save(tmp_path / "flat.npy", np.arange(5))
    with pytest.raises(CalibrationError, match="samples along its first axis"):
        NumpyDataset(tmp_path / "flat.npy")
    (tmp_path / "empty").mkdir()
    with pytest.raises(CalibrationError, match=r"no \.npy files"):
        NumpyDataset(tmp_path / "empty")

    np.save(tmp_path / "a.npy", np.zeros((2, 3)))
    np.save(tmp_path / "b.npy", np.ones((2, 3)))
    assert NumpyDataset(tmp_path / "a.npy").identity().fingerprint != (
        NumpyDataset(tmp_path / "b.npy").identity().fingerprint
    )


# --------------------------------------------------------------------------- synthetic data


def test_synthetic_data_is_deterministic_and_never_claims_to_be_representative() -> None:
    a, b = SyntheticDataset((3, 4), 5, seed=1), SyntheticDataset((3, 4), 5, seed=1)
    np.testing.assert_array_equal(a.load(2), b.load(2))
    assert not np.array_equal(a.load(2), a.load(3))
    assert not np.array_equal(a.load(2), SyntheticDataset((3, 4), 5, seed=2).load(2))
    identity = a.identity()
    assert identity.representative is False
    assert identity.kind == "synthetic"
    assert identity.fingerprint != SyntheticDataset((3, 4), 5, seed=2).identity().fingerprint
    assert identity.fingerprint != SyntheticDataset((3, 5), 5, seed=1).identity().fingerprint


def calibration_config(**kw: Any) -> CalibrationConfig:
    return CalibrationConfig.model_validate(kw, context={"base_dir": Path.cwd()})


def test_open_dataset_dispatches_on_kind(images: Path, tmp_path: Path) -> None:
    assert isinstance(
        open_dataset(calibration_config(dataset="images", path=str(images)), (3, 8, 8), 0),
        ImageFolderDataset,
    )
    np.save(tmp_path / "d.npy", np.zeros((2, 3)))
    assert isinstance(
        open_dataset(calibration_config(dataset="numpy", path=str(tmp_path / "d.npy")), (3,), 0),
        NumpyDataset,
    )
    synthetic = open_dataset(
        calibration_config(dataset="synthetic", allow_synthetic=True, num_samples=7), (3,), 5
    )
    assert isinstance(synthetic, SyntheticDataset)
    assert len(synthetic) == 7


# --------------------------------------------------------------------------- preprocessing


def spec(shape: list[int | str], dtype: str = "float32") -> TensorSpec:
    return TensorSpec(name="image", dtype=dtype, shape=shape)


def solid(color: tuple[int, int, int], size: tuple[int, int] = (6, 4)) -> np.ndarray:
    array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    array[:] = color
    return array


def test_sample_shape() -> None:
    assert sample_shape(spec(["batch", 3, 32, 32])) == (3, 32, 32)
    assert sample_shape(spec(["batch", 4, "seq"]), {"seq": 16}) == (4, 16)
    with pytest.raises(CalibrationError, match="axis 2 is dynamic"):
        sample_shape(spec(["batch", 4, "seq"]))


def test_rescale_layout_and_normalization() -> None:
    config = PreprocessingConfig.model_validate({"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.25, 0.5]})
    prep = Preprocessor(config, spec(["batch", 3, 6, 4]), (3, 6, 4))
    out = prep(solid((255, 0, 0)))
    assert out.shape == (3, 6, 4)  # CHW
    assert out.dtype == np.float32
    assert out[0, 0, 0] == pytest.approx((1.0 - 0.5) / 0.5)  # R
    assert out[1, 0, 0] == pytest.approx((0.0 - 0.5) / 0.25)  # G
    assert out[2, 0, 0] == pytest.approx((0.0 - 0.5) / 0.5)  # B


def test_bgr_channel_order_and_custom_rescale() -> None:
    config = PreprocessingConfig.model_validate({"channel_order": "bgr", "rescale": 1.0})
    out = Preprocessor(config, spec(["batch", 3, 6, 4]), (3, 6, 4))(solid((10, 20, 30)))
    assert out[:, 0, 0].tolist() == [30.0, 20.0, 10.0]


def test_resize_and_center_crop() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[2:6, 2:6] = (255, 255, 255)  # a white square in the middle
    config = PreprocessingConfig.model_validate({"center_crop": (4, 4)})
    out = Preprocessor(config, spec(["batch", 3, 4, 4]), (3, 4, 4))(image)
    assert out.min() == pytest.approx(1.0)  # the crop is exactly the white square

    resized = PreprocessingConfig.model_validate({"resize": (2, 2)})
    result = Preprocessor(resized, spec(["batch", 3, 2, 2]), (3, 2, 2))(solid((255, 255, 255)))
    assert result.shape == (3, 2, 2)
    assert result.min() == pytest.approx(1.0)

    with pytest.raises(CalibrationError, match="larger than the image"):
        Preprocessor(
            PreprocessingConfig.model_validate({"center_crop": (16, 16)}),
            spec(["batch", 3, 16, 16]),
            (3, 16, 16),
        )(image)


def test_grayscale_inputs_average_the_channels() -> None:
    out = Preprocessor(PreprocessingConfig(), spec(["batch", 1, 6, 4]), (1, 6, 4))(
        solid((255, 0, 0))
    )
    assert out.shape == (1, 6, 4)
    assert out[0, 0, 0] == pytest.approx(1 / 3)


def test_rgba_alpha_is_dropped() -> None:
    rgba = np.zeros((6, 4, 4), dtype=np.uint8)
    rgba[..., :3] = (255, 0, 0)
    rgba[..., 3] = 7
    out = Preprocessor(PreprocessingConfig(), spec(["batch", 3, 6, 4]), (3, 6, 4))(rgba)
    assert out[:, 0, 0].tolist() == [1.0, 0.0, 0.0]


def test_preprocessing_errors() -> None:
    plain = Preprocessor(PreprocessingConfig(), spec(["batch", 3, 6, 4]), (3, 6, 4))
    with pytest.raises(CalibrationError, match=r"expects \[3, 6, 4\]"):
        plain(solid((1, 2, 3), (5, 4)))
    bad_mean = PreprocessingConfig.model_validate({"mean": [0.5, 0.5], "std": [1.0, 1.0]})
    with pytest.raises(CalibrationError, match="mean has 2 values for 3 channels"):
        Preprocessor(bad_mean, spec(["batch", 3, 6, 4]), (3, 6, 4))(solid((1, 2, 3)))


def test_non_image_samples_pass_through_with_dtype_cast() -> None:
    prep = Preprocessor(PreprocessingConfig(), spec(["batch", 5], "float16"), (5,))
    out = prep(np.arange(5, dtype=np.float64))
    assert out.dtype == np.float16
    assert out.tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(CalibrationError, match="expects"):
        prep(np.arange(4, dtype=np.float32))


# --------------------------------------------------------------------------- sampling


def test_selection_is_deterministic_sorted_and_seeded() -> None:
    a = select_samples(100, 10, 5, seed=3)
    assert a == select_samples(100, 10, 5, seed=3)
    assert list(a.indices) == sorted(a.indices)
    assert len(set(a.indices)) == 10
    assert a.indices != select_samples(100, 10, 5, seed=4).indices
    assert (a.batches, a.sample_count) == (2, 10)


def test_all_samples_are_used_when_the_dataset_is_small() -> None:
    selection = select_samples(6, 512, 2, seed=0)
    assert selection.indices == (0, 1, 2, 3, 4, 5)


def test_a_trailing_partial_batch_is_dropped() -> None:
    selection = select_samples(10, 10, 4, seed=0)
    assert (selection.batches, selection.sample_count) == (2, 8)


def test_selection_errors() -> None:
    with pytest.raises(CalibrationError, match="empty"):
        select_samples(0, 5, 1, seed=0)
    with pytest.raises(CalibrationError, match="cannot fill one batch of 8"):
        select_samples(5, 5, 8, seed=0)


def test_batches_have_the_right_shape_and_order(tmp_path: Path) -> None:
    data = np.arange(7 * 3, dtype=np.float32).reshape(7, 3)
    np.save(tmp_path / "d.npy", data)
    dataset = NumpyDataset(tmp_path / "d.npy")
    selection = select_samples(7, 7, 3, seed=0)
    prep = Preprocessor(PreprocessingConfig(), spec(["batch", 3]), (3,))
    batches = list(iter_batches(dataset, selection, prep))
    assert [b.shape for b in batches] == [(3, 3), (3, 3)]  # 7 samples -> 2 full batches
    np.testing.assert_array_equal(batches[0], data[:3])
    np.testing.assert_array_equal(batches[1], data[3:6])


# --------------------------------------------------------------------------- cache


def metadata_for(cache: bytes, **overrides: Any) -> CalibrationMetadata:
    fields: dict[str, Any] = {
        "created_at": utc_now(),
        "dataset": DatasetIdentity(kind="numpy", name="d", fingerprint="f" * 64, items=10),
        "requested_samples": 8,
        "sample_count": 8,
        "batch_size": 4,
        "num_batches": 2,
        "seed": 0,
        "method": "entropy2",
        "preprocessing": {"rescale": 0.5},
        "input_name": "x",
        "input_dtype": "float32",
        "sample_shape": [16],
        "tensorrt_version": "10.3.0.26",
        "cuda_version": "12.4",
        "gpu": "Test GPU",
        "model_weights_sha256": "a" * 64,
        "source_onnx_sha256": "b" * 64,
        "cache_sha256": sha256_bytes(cache),
        "representative": True,
    }
    fields.update(overrides)
    return CalibrationMetadata.model_validate(fields)


def test_cache_directory_round_trip(tmp_path: Path) -> None:
    cache = b"TRT-CALIBRATION-CACHE\nscales..."
    metadata = metadata_for(cache)
    target = tmp_path / "cal"
    write_cache_dir(target, cache, metadata)
    assert sorted(p.name for p in target.iterdir()) == [CACHE_FILE, METADATA_FILE]
    read_cache, read_meta = read_cache_dir(target)
    assert read_cache == cache
    assert read_meta == metadata


def test_cache_directory_integrity(tmp_path: Path) -> None:
    cache = b"scales"
    target = tmp_path / "cal"
    write_cache_dir(target, cache, metadata_for(cache))
    with pytest.raises(CalibrationError, match="refusing to overwrite"):
        write_cache_dir(target, cache, metadata_for(cache))
    (target / CACHE_FILE).write_bytes(b"tampered")
    with pytest.raises(CalibrationError, match="does not match its metadata"):
        read_cache_dir(target)
    with pytest.raises(CalibrationError, match="not a calibration cache"):
        read_cache_dir(tmp_path)
    (target / METADATA_FILE).write_text("{broken")
    with pytest.raises(CalibrationError, match="unreadable calibration metadata"):
        read_cache_dir(target)
    with pytest.raises(CalibrationError, match="does not describe the cache bytes"):
        write_cache_dir(tmp_path / "other", b"real", metadata_for(b"different"))


def test_compatibility_reports_every_mismatch() -> None:
    meta = metadata_for(b"c")
    same = {
        "dataset_fingerprint": "f" * 64,
        "sample_count": 8,
        "batch_size": 4,
        "method": "entropy2",
        "tensorrt_version": "10.3.0.26",
    }
    assert compatibility_problems(meta, same) == []
    changed = {
        **same,
        "dataset_fingerprint": "0" * 64,
        "batch_size": 8,
        "tensorrt_version": "10.4.0",
    }
    problems = compatibility_problems(meta, changed)
    assert len(problems) == 3
    assert any(p.startswith("dataset_fingerprint:") for p in problems)
    assert any("batch_size: cache has 4, current run needs 8" in p for p in problems)
    assert any("tensorrt_version" in p for p in problems)
