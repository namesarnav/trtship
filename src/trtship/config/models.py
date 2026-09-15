"""Pydantic models for the trtship YAML configuration.

Path semantics: *input* paths (model weights, datasets) resolve against the directory of the
config file; *output* paths (run root, cache, model repository) resolve against the current
working directory. Both are made absolute during validation so a config snapshot is unambiguous.
"""

from __future__ import annotations

import ipaddress
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationInfo,
    field_validator,
    model_validator,
)

from trtship.specs import IDENTIFIER, TensorSpec
from trtship.utils.hashing import sha256_json

# Config-facing name for the shared tensor spec.
InputSpec = TensorSpec

_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FACTORY = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")


def _resolve_input_path(value: Path, info: ValidationInfo) -> Path:
    path = value.expanduser()
    base = (info.context or {}).get("base_dir")
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return path


def _resolve_output_path(value: Path) -> Path:
    return value.expanduser().absolute()


InputPath = Annotated[Path, AfterValidator(_resolve_input_path)]
OutputPath = Annotated[Path, AfterValidator(_resolve_output_path)]


class _Base(BaseModel):
    # validate_default: path defaults must pass through the same resolution as user-supplied values.
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


# --------------------------------------------------------------------------- enums


class Precision(StrEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


class ModelKind(StrEnum):
    MODULE = "module"  # factory builds an nn.Module (random init)
    CHECKPOINT = "checkpoint"  # factory builds an nn.Module, path holds a state_dict
    TORCHSCRIPT = "torchscript"  # path holds a TorchScript archive


# --------------------------------------------------------------------------- model


class ModelConfig(_Base):
    name: str
    kind: ModelKind
    path: InputPath | None = None
    factory: str | None = None
    factory_kwargs: dict[str, Any] = Field(default_factory=dict)
    # Directories added to sys.path before the factory is imported (relative to the config file).
    python_path: list[InputPath] = Field(default_factory=list)
    inputs: list[TensorSpec] = Field(min_length=1)
    output_names: list[str] | None = None
    trust_source: bool = False

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _MODEL_NAME.match(value):
            raise ValueError(
                "must start with a letter/digit and contain only letters, digits, '_', '.', '-'"
            )
        return value

    @field_validator("factory")
    @classmethod
    def _valid_factory(cls, value: str | None) -> str | None:
        if value is not None and not _FACTORY.match(value):
            raise ValueError("must look like 'package.module:callable'")
        return value

    @model_validator(mode="after")
    def _check_kind_requirements(self) -> ModelConfig:
        if self.kind is ModelKind.MODULE:
            if self.factory is None:
                raise ValueError("kind 'module' requires 'factory'")
            if self.path is not None:
                raise ValueError(
                    "kind 'module' does not use 'path'; use kind 'checkpoint' for weights"
                )
        if self.kind is ModelKind.CHECKPOINT and (self.factory is None or self.path is None):
            raise ValueError("kind 'checkpoint' requires both 'factory' and 'path'")
        if self.kind is ModelKind.TORCHSCRIPT:
            if self.path is None:
                raise ValueError("kind 'torchscript' requires 'path'")
            if self.factory is not None:
                raise ValueError("kind 'torchscript' does not use 'factory'")
        names = [spec.name for spec in self.inputs]
        if len(set(names)) != len(names):
            raise ValueError("input names must be unique")
        if self.output_names is not None:
            if len(set(self.output_names)) != len(self.output_names):
                raise ValueError("output_names must be unique")
            if not all(IDENTIFIER.match(n) for n in self.output_names):
                raise ValueError("output_names must be identifiers")
        return self

    def input(self, name: str) -> TensorSpec:
        for spec in self.inputs:
            if spec.name == name:
                return spec
        raise KeyError(name)


# --------------------------------------------------------------------------- pipeline stages


class ExportConfig(_Base):
    opset: int = Field(default=17, ge=11, le=23)
    constant_folding: bool = True
    dynamo: bool = False  # torch.export-based exporter instead of TorchScript tracing


OptimizePass = Literal[
    "extract_constants",
    "eliminate_identity",
    "deduplicate_initializers",
    "eliminate_dead_nodes",
    "eliminate_unused_initializers",
    "infer_shapes",
]


class OptimizeConfig(_Base):
    enabled: bool = True
    passes: list[OptimizePass] | None = None  # None runs every pass, in the canonical order

    @field_validator("passes")
    @classmethod
    def _unique_passes(cls, value: list[OptimizePass] | None) -> list[OptimizePass] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("passes must be unique")
        return value


class ShapeRange(_Base):
    min: list[StrictInt] = Field(min_length=1)
    opt: list[StrictInt] = Field(min_length=1)
    max: list[StrictInt] = Field(min_length=1)

    @model_validator(mode="after")
    def _ordered(self) -> ShapeRange:
        if not (len(self.min) == len(self.opt) == len(self.max)):
            raise ValueError("min/opt/max must have the same rank")
        for axis, (lo, mid, hi) in enumerate(zip(self.min, self.opt, self.max, strict=True)):
            if lo < 1:
                raise ValueError(f"axis {axis}: dims must be >= 1")
            if not lo <= mid <= hi:
                raise ValueError(f"axis {axis}: require min <= opt <= max, got {lo}/{mid}/{hi}")
        return self


class OptimizationProfile(_Base):
    """A TensorRT optimization profile: a shape range per dynamic input."""

    inputs: dict[str, ShapeRange] = Field(min_length=1)


class TensorRTConfig(_Base):
    precisions: list[Precision] = Field(default_factory=lambda: [Precision.FP32], min_length=1)
    workspace_mb: int = Field(default=4096, ge=16)
    optimization_level: int = Field(default=3, ge=0, le=5)
    profiles: list[OptimizationProfile] = Field(default_factory=list)
    timing_cache_path: OutputPath | None = None
    device_index: int = Field(default=0, ge=0)
    # An INT8 build also enables FP16 so layers without an INT8 kernel need not fall back to FP32.
    int8_fp16_fallback: bool = True

    @field_validator("precisions")
    @classmethod
    def _unique_precisions(cls, value: list[Precision]) -> list[Precision]:
        if len(set(value)) != len(value):
            raise ValueError("precisions must be unique")
        return value


class CalibrationDatasetKind(StrEnum):
    IMAGES = "images"
    NUMPY = "numpy"
    SYNTHETIC = "synthetic"


class PreprocessingConfig(_Base):
    resize: tuple[int, int] | None = None  # (height, width)
    center_crop: tuple[int, int] | None = None
    rescale: float = 1.0 / 255.0
    mean: list[float] | None = None
    std: list[float] | None = None
    channel_order: Literal["rgb", "bgr"] = "rgb"

    @model_validator(mode="after")
    def _check(self) -> PreprocessingConfig:
        if self.std is not None and any(s == 0 for s in self.std):
            raise ValueError("std must be non-zero")
        if (self.mean is None) != (self.std is None):
            raise ValueError("mean and std must be given together")
        if self.mean is not None and self.std is not None and len(self.mean) != len(self.std):
            raise ValueError("mean and std must have the same length")
        return self


class CalibrationConfig(_Base):
    dataset: CalibrationDatasetKind
    path: InputPath | None = None
    glob: str = "*"
    input_name: str | None = None  # which model input the dataset feeds; None = the only input
    num_samples: int = Field(default=512, ge=1)
    batch_size: int = Field(default=8, ge=1)
    method: Literal["entropy2", "minmax"] = "entropy2"
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    seed: int | None = None  # None inherits the top-level seed
    cache_path: OutputPath | None = None
    allow_synthetic: bool = False

    @model_validator(mode="after")
    def _check(self) -> CalibrationConfig:
        if self.dataset is CalibrationDatasetKind.SYNTHETIC:
            if not self.allow_synthetic:
                raise ValueError(
                    "synthetic calibration data is not representative of real inputs and yields "
                    "unreliable INT8 scales; set 'allow_synthetic: true' to opt in explicitly"
                )
        elif self.path is None:
            raise ValueError(f"dataset '{self.dataset.value}' requires 'path'")
        return self


class Tolerance(_Base):
    """Acceptance thresholds for comparing a candidate's outputs against a reference."""

    atol: float = Field(ge=0)
    rtol: float = Field(ge=0)
    cosine_min: float = Field(ge=0, le=1)
    top1_agreement_min: float | None = Field(default=None, ge=0, le=1)
    topk: int = Field(default=5, ge=1)
    topk_agreement_min: float | None = Field(default=None, ge=0, le=1)


def _default_tolerances() -> dict[Precision, Tolerance]:
    # Starting points, not guarantees: quantized precisions are compared mainly on direction
    # (cosine) and decision agreement, since element-wise error is dominated by quantization noise.
    return {
        Precision.FP32: Tolerance(atol=1e-4, rtol=1e-3, cosine_min=0.99999, top1_agreement_min=1.0),
        Precision.FP16: Tolerance(atol=5e-2, rtol=5e-2, cosine_min=0.999, top1_agreement_min=0.99),
        Precision.INT8: Tolerance(atol=5e-1, rtol=1e-1, cosine_min=0.98, top1_agreement_min=0.95),
    }


class ValidationConfig(_Base):
    num_samples: int = Field(default=8, ge=1)
    seed: int | None = None
    onnx_tolerance: Tolerance = Field(
        default_factory=lambda: Tolerance(atol=1e-4, rtol=1e-3, cosine_min=0.99999)
    )
    tolerances: dict[Precision, Tolerance] = Field(default_factory=_default_tolerances)

    @field_validator("tolerances")
    @classmethod
    def _fill_missing(cls, value: dict[Precision, Tolerance]) -> dict[Precision, Tolerance]:
        return {**_default_tolerances(), **value}


class BenchmarkConfig(_Base):
    warmup_iters: int = Field(default=50, ge=0)
    iters: int = Field(default=500, ge=1)
    batch_sizes: list[int] = Field(default_factory=lambda: [1], min_length=1)
    concurrency: list[int] = Field(default_factory=lambda: [1], min_length=1)
    precisions: list[Precision] | None = None  # None benchmarks every built precision
    seed: int | None = None
    # Also measure the ONNX model on ONNX Runtime's CPU provider, as a labelled baseline.
    include_onnx_baseline: bool = True

    @field_validator("batch_sizes", "concurrency")
    @classmethod
    def _positive(cls, value: list[int]) -> list[int]:
        if any(v < 1 for v in value):
            raise ValueError("values must be >= 1")
        return value


class DynamicBatchingConfig(_Base):
    preferred_batch_sizes: list[int] = Field(default_factory=list)
    max_queue_delay_us: int = Field(default=100, ge=0)


class TritonConfig(_Base):
    repository_dir: OutputPath = Field(default_factory=lambda: Path("model_repository"))
    model_version: int = Field(default=1, ge=1)
    precision: Precision | None = None  # engine to package; None requires a single built precision
    max_batch_size: int | None = Field(default=None, ge=0)  # None derives from the engine profile
    instance_count: int = Field(default=1, ge=1)
    instance_gpus: list[int] = Field(default_factory=lambda: [0], min_length=1)
    dynamic_batching: DynamicBatchingConfig | None = None
    # The TensorRT plan must be loaded by the same TensorRT version that built it, so there is
    # deliberately no default image: pick the Triton release that ships your TensorRT version.
    image: str | None = None
    container_name: str = "trtship-triton"
    # Published ports are bound to this address; the default keeps the server local to the host.
    bind_address: str = "127.0.0.1"
    http_port: int = Field(default=8000, ge=1, le=65535)
    grpc_port: int = Field(default=8001, ge=1, le=65535)
    metrics_port: int = Field(default=8002, ge=1, le=65535)
    startup_timeout_s: float = Field(default=180.0, gt=0)

    @field_validator("bind_address")
    @classmethod
    def _valid_address(cls, value: str) -> str:
        try:
            ipaddress.ip_address(value)
        except ValueError as exc:
            raise ValueError("must be an IPv4 or IPv6 address") from exc
        return value

    @field_validator("image")
    @classmethod
    def _valid_image(cls, value: str | None) -> str | None:
        # The value becomes a docker argument: a leading "-" would be read as an option
        # (`--privileged`), so only an image reference is accepted.
        if value is not None and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/:@-]*", value):
            raise ValueError(
                "must be an image reference such as nvcr.io/nvidia/tritonserver:<tag>, with no "
                "spaces and not starting with '-'"
            )
        return value

    @field_validator("container_name")
    @classmethod
    def _valid_container_name(cls, value: str) -> str:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", value):
            raise ValueError("must be a valid Docker container name")
        return value

    @model_validator(mode="after")
    def _distinct_ports(self) -> TritonConfig:
        ports = [self.http_port, self.grpc_port, self.metrics_port]
        if len(set(ports)) != len(ports):
            raise ValueError("http_port, grpc_port and metrics_port must be distinct")
        return self


class ArtifactsConfig(_Base):
    root: OutputPath = Field(default_factory=lambda: Path("runs"))
    cache_dir: OutputPath = Field(default_factory=lambda: Path(".trtship-cache"))
    reuse_cache: bool = True


# --------------------------------------------------------------------------- root


class TrtshipConfig(_Base):
    schema_version: Literal[1] = 1
    seed: int = 0
    model: ModelConfig
    export: ExportConfig = Field(default_factory=ExportConfig)
    optimize: OptimizeConfig = Field(default_factory=OptimizeConfig)
    tensorrt: TensorRTConfig = Field(default_factory=TensorRTConfig)
    calibration: CalibrationConfig | None = None
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    triton: TritonConfig = Field(default_factory=TritonConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)

    @model_validator(mode="after")
    def _cross_checks(self) -> TrtshipConfig:
        built = set(self.tensorrt.precisions)
        if Precision.INT8 in built and self.calibration is None:
            raise ValueError(
                "tensorrt.precisions includes int8, which requires a 'calibration' section"
            )
        if self.benchmark.precisions is not None and not set(self.benchmark.precisions) <= built:
            raise ValueError("benchmark.precisions must be a subset of tensorrt.precisions")
        if self.triton.precision is not None and self.triton.precision not in built:
            raise ValueError("triton.precision must be one of tensorrt.precisions")
        if self.calibration is not None and self.calibration.input_name is not None:
            names = {spec.name for spec in self.model.inputs}
            if self.calibration.input_name not in names:
                raise ValueError(
                    f"calibration.input_name {self.calibration.input_name!r} is not a model input"
                )
        self._check_profiles()
        return self

    def _check_profiles(self) -> None:
        for index, profile in enumerate(self.tensorrt.profiles):
            for name, rng in profile.inputs.items():
                try:
                    spec = self.model.input(name)
                except KeyError:
                    raise ValueError(
                        f"tensorrt.profiles[{index}] references unknown input {name!r}"
                    ) from None
                if len(rng.min) != len(spec.shape):
                    raise ValueError(
                        f"tensorrt.profiles[{index}].{name}: rank {len(rng.min)} does not match "
                        f"declared rank {len(spec.shape)}"
                    )
                for axis, declared in enumerate(spec.shape):
                    if (
                        isinstance(declared, int)
                        and not rng.min[axis] == rng.opt[axis] == rng.max[axis] == declared
                    ):
                        raise ValueError(
                            f"tensorrt.profiles[{index}].{name}: axis {axis} is static "
                            f"({declared}) but the profile varies it"
                        )

    def warnings(self) -> list[str]:
        """Non-fatal problems worth surfacing in `config validate`."""
        found: list[str] = []
        profiled = {name for profile in self.tensorrt.profiles for name in profile.inputs}
        for spec in self.model.inputs:
            if spec.is_dynamic and spec.name not in profiled:
                found.append(
                    f"input {spec.name!r} has dynamic dims "
                    f"{sorted(spec.dynamic_axes.values())} but no tensorrt.profiles entry; "
                    "engine builds will fail without one"
                )
        if self.tensorrt.precisions and self.model.trust_source:
            found.append("model.trust_source is true: model loading may execute arbitrary code")
        return found


def config_hash(config: TrtshipConfig) -> str:
    """Stable hash of the fully resolved configuration."""
    return sha256_json(config.model_dump(mode="json"))
