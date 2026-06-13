"""Running INT8 calibration: data -> TensorRT calibrator -> cache directory."""

from __future__ import annotations

import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trtship.calibration.cache import (
    CalibrationMetadata,
    compatibility_problems,
    read_cache_dir,
    write_cache_dir,
)
from trtship.calibration.calibrator import DeviceBuffers, TorchDeviceBuffers, make_calibrator
from trtship.calibration.dataset import CalibrationDataset, DatasetIdentity, open_dataset
from trtship.calibration.preprocess import Preprocessor, sample_shape
from trtship.calibration.sampling import Selection, iter_batches, select_samples
from trtship.config import CalibrationConfig, Precision, TrtshipConfig
from trtship.errors import CalibrationError
from trtship.logging import get_logger
from trtship.models import ModelSignature, make_input, resolve_symbol_sizes
from trtship.specs import TensorSpec
from trtship.tensorrt import build_engine, load_tensorrt
from trtship.utils.hashing import sha256_bytes, sha256_file
from trtship.utils.timeutil import utc_now

log = get_logger(__name__)


def calibration_input(config: TrtshipConfig) -> TensorSpec:
    """The model input the calibration data feeds."""
    calibration = config.calibration
    assert calibration is not None
    if calibration.input_name is not None:
        return config.model.input(calibration.input_name)
    if len(config.model.inputs) != 1:
        names = [s.name for s in config.model.inputs]
        raise CalibrationError(
            f"the model has several inputs {names}; say which one the data feeds",
            hint="Set calibration.input_name.",
        )
    return config.model.inputs[0]


def _batch_sizes(config: TrtshipConfig, spec: TensorSpec, batch_size: int) -> dict[str, int]:
    """Symbol sizes for a calibration batch: profile ``opt`` shapes, with the calibrated input's
    leading (batch) axis set to the calibration batch size."""
    sizes = resolve_symbol_sizes(config.model, config.tensorrt.profiles, "opt")
    lead = spec.shape[0]
    if isinstance(lead, str):
        sizes[lead] = batch_size
    elif lead != batch_size:
        raise CalibrationError(
            f"input {spec.name!r} has a fixed batch axis of {lead} but calibration.batch_size is "
            f"{batch_size}",
            hint="Set calibration.batch_size to the input's fixed batch size.",
        )
    return sizes


def expected_metadata_fields(
    config: TrtshipConfig,
    spec: TensorSpec,
    shape: tuple[int, ...],
    fingerprint: str,
    sample_count: int,
    seed: int,
    weights_sha256: str,
    onnx_sha256: str,
    tensorrt_version: str,
) -> dict[str, Any]:
    calibration = config.calibration
    assert calibration is not None
    return {
        "dataset_fingerprint": fingerprint,
        "sample_count": sample_count,
        "batch_size": calibration.batch_size,
        "method": calibration.method,
        "preprocessing": calibration.preprocessing.model_dump(mode="json"),
        "input_name": spec.name,
        "sample_shape": list(shape),
        "seed": seed,
        "model_weights_sha256": weights_sha256,
        "source_onnx_sha256": onnx_sha256,
        "tensorrt_version": tensorrt_version,
    }


@dataclass
class Prepared:
    """Everything derived from the configuration before TensorRT is involved."""

    calibration: CalibrationConfig
    spec: TensorSpec
    sizes: dict[str, int]
    shape: tuple[int, ...]
    seed: int
    dataset: CalibrationDataset
    identity: DatasetIdentity
    selection: Selection


def prepare(config: TrtshipConfig) -> Prepared:
    """Open the dataset and choose the samples. Needs no GPU."""
    calibration = config.calibration
    if calibration is None:
        raise CalibrationError("calibration is not configured", hint="Add a `calibration` section.")
    spec = calibration_input(config)
    seed = calibration.seed if calibration.seed is not None else config.seed
    sizes = _batch_sizes(config, spec, calibration.batch_size)
    shape = sample_shape(spec, sizes)
    dataset = open_dataset(calibration, shape, seed)
    identity = dataset.identity()
    selection = select_samples(len(dataset), calibration.num_samples, calibration.batch_size, seed)
    return Prepared(calibration, spec, sizes, shape, seed, dataset, identity, selection)


def calibrate(
    onnx_path: Path,
    signature: ModelSignature,
    config: TrtshipConfig,
    output_dir: Path,
    *,
    model_weights_sha256: str,
    trt: Any | None = None,
    device: DeviceBuffers | None = None,
) -> CalibrationMetadata:
    """Calibrate ``onnx_path`` for INT8 and write the cache directory ``output_dir``.

    Real calibration data is run through TensorRT, which computes the scales; nothing here invents
    them. An existing cache (``calibration.cache_path``) is adopted only if it verifiably matches
    the current model, data, preprocessing, and TensorRT version.
    """
    prepared = prepare(config)
    calibration, spec, sizes, shape, seed = (
        prepared.calibration,
        prepared.spec,
        prepared.sizes,
        prepared.shape,
        prepared.seed,
    )
    dataset, identity, selection = prepared.dataset, prepared.identity, prepared.selection
    log.info(
        "calibrating on %s: %d samples in %d batches of %d",
        identity.name,
        selection.sample_count,
        selection.batches,
        selection.batch_size,
    )

    trt = trt if trt is not None else load_tensorrt(purpose="INT8 calibration")
    onnx_sha = sha256_file(onnx_path)
    expected = expected_metadata_fields(
        config, spec, shape, identity.fingerprint, selection.sample_count, seed,
        model_weights_sha256, onnx_sha, str(trt.__version__),
    )  # fmt: skip

    if calibration.cache_path is not None:
        return _adopt_existing(calibration.cache_path, output_dir, expected)

    preprocessor = Preprocessor(calibration.preprocessing, spec, shape)
    other_inputs = {
        other.name: make_input(other, sizes, seed=seed).numpy()
        for other in config.model.inputs
        if other.name != spec.name
    }
    calibrator = make_calibrator(
        trt,
        calibration.method,
        batches=iter_batches(dataset, selection, preprocessor),
        batch_size=calibration.batch_size,
        input_name=spec.name,
        other_inputs=other_inputs,
        device=device if device is not None else TorchDeviceBuffers(),
    )
    scratch = output_dir.with_name(f".{output_dir.name}.{secrets.token_hex(4)}.build")
    scratch.mkdir(parents=True)
    try:
        # TensorRT calibrates while building; the engine itself is discarded (`build` rebuilds
        # from the cache, so final engines never depend on calibration-time state).
        build_engine(
            onnx_path, signature, config, Precision.INT8, scratch / "calibration.plan",
            calibrator=calibrator, trt=trt,
        )  # fmt: skip
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if calibrator.cache_bytes is None:
        raise CalibrationError(
            "TensorRT finished without writing a calibration cache",
            hint="Check that the model has layers that can run in INT8.",
        )
    if calibrator.batches_served < 1:
        raise CalibrationError("TensorRT consumed no calibration batches")

    metadata = CalibrationMetadata(
        created_at=utc_now(),
        dataset=identity,
        requested_samples=calibration.num_samples,
        sample_count=selection.sample_count,
        batch_size=calibration.batch_size,
        num_batches=selection.batches,
        seed=seed,
        method=calibration.method,
        preprocessing=calibration.preprocessing.model_dump(mode="json"),
        input_name=spec.name,
        input_dtype=spec.dtype.value,
        sample_shape=list(shape),
        tensorrt_version=str(trt.__version__),
        cuda_version=_cuda_version(),
        gpu=_gpu_name(),
        model_weights_sha256=model_weights_sha256,
        source_onnx_sha256=onnx_sha,
        cache_sha256=sha256_bytes(calibrator.cache_bytes),
        representative=identity.representative,
    )
    write_cache_dir(output_dir, calibrator.cache_bytes, metadata)
    return metadata


def _adopt_existing(
    source: Path, output_dir: Path, expected: dict[str, Any]
) -> CalibrationMetadata:
    cache, metadata = read_cache_dir(source)
    problems = compatibility_problems(metadata, expected)
    if problems:
        raise CalibrationError(
            f"the calibration cache at {source} does not match this run",
            hint="Remove calibration.cache_path to calibrate afresh.",
            details={"mismatches": problems},
        )
    write_cache_dir(output_dir, cache, metadata)
    return metadata


def _cuda_version() -> str | None:
    import torch  # noqa: PLC0415

    return getattr(torch.version, "cuda", None)


def _gpu_name() -> str | None:
    import torch  # noqa: PLC0415

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
