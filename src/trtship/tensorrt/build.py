"""TensorRT engine building from an ONNX model.

Written against the TensorRT 10 Python API with compatibility for 8.6+ (the differences handled
are explicit-batch network creation and the engine-introspection API). The ``trt`` module is a
parameter so the translation logic (configuration -> builder calls, error reporting) can be
unit-tested with a fake; the real thing runs only where TensorRT and a supported GPU exist.
"""

from __future__ import annotations

import re
import secrets
import time
from pathlib import Path
from typing import Any

import onnx
from pydantic import BaseModel, ConfigDict, Field

from trtship.config import Precision, TrtshipConfig
from trtship.errors import EngineBuildError, EnvironmentUnavailableError
from trtship.logging import get_logger
from trtship.models import ModelSignature
from trtship.tensorrt.engine_info import EngineInfo, describe_engine
from trtship.tensorrt.loader import TrtVersion, check_supported, load_tensorrt, parse_version
from trtship.tensorrt.profiles import ProfileShapes, build_profile_shapes
from trtship.utils.fs import atomic_write_bytes, publish_new
from trtship.utils.hashing import sha256_file

log = get_logger(__name__)

_MAX_LOG_LINES = 200
_QDQ_OPS = frozenset({"QuantizeLinear", "DequantizeLinear"})
_UNSUPPORTED_PATTERNS = (
    re.compile(r"No importer registered for op:\s*(\w+)"),
    re.compile(r"Unsupported ONNX (?:operator|op)[^:]*:\s*(\w+)", re.IGNORECASE),
    re.compile(r"getPluginCreator could not find plugin:?\s*(\w+)"),
)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TimingCacheInfo(_Frozen):
    path: str | None
    loaded_bytes: int
    saved_bytes: int


class BuildResult(_Frozen):
    precision: Precision
    path: str
    sha256: str
    size_bytes: int
    build_time_s: float
    tensorrt_version: str
    builder_flags: list[str]
    workspace_mb: int
    optimization_level: int | None
    profiles: list[dict[str, dict[str, list[int]]]]
    calibrator: str | None
    timing_cache: TimingCacheInfo
    engine: EngineInfo
    warnings: list[str] = Field(default_factory=list)
    builder_log: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- diagnostics


def parser_errors(parser: Any) -> list[dict[str, Any]]:
    """Collect every error the ONNX parser recorded."""
    errors = []
    for index in range(int(parser.num_errors)):
        error = parser.get_error(index)
        errors.append(
            {
                "index": index,
                "code": str(error.code()),
                "node": int(error.node()),
                "description": str(error.desc()),
            }
        )
    return errors


def unsupported_operators(errors: list[dict[str, Any]]) -> list[str]:
    """Operator names the parser reported as unsupported, de-duplicated, in order."""
    found: dict[str, None] = {}
    for error in errors:
        for pattern in _UNSUPPORTED_PATTERNS:
            for match in pattern.finditer(error["description"]):
                found.setdefault(match.group(1), None)
    return list(found)


def _logger(trt: Any, sink: list[str]) -> Any:
    class CapturingLogger(trt.ILogger):  # type: ignore[misc]
        def __init__(self) -> None:
            trt.ILogger.__init__(self)

        def log(self, severity: Any, msg: str) -> None:
            level = int(severity)
            if level <= int(trt.ILogger.Severity.WARNING):
                if len(sink) < _MAX_LOG_LINES:
                    sink.append(f"[{getattr(severity, 'name', level)}] {msg}")
            else:
                log.debug("tensorrt: %s", msg)

    return CapturingLogger()


def _select_cuda_device(index: int) -> None:
    """Make ``index`` the current CUDA device (TensorRT builds on the current device)."""
    import torch  # noqa: PLC0415 - only needed on GPU machines

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if index >= count:
            raise EnvironmentUnavailableError(
                f"tensorrt.device_index is {index} but only {count} GPU(s) exist"
            )
        torch.cuda.set_device(index)
    elif index != 0:
        raise EnvironmentUnavailableError(
            f"cannot select GPU {index}: selecting a device needs a CUDA-enabled PyTorch",
            hint="Install a CUDA build of PyTorch, or use tensorrt.device_index: 0.",
        )


def has_explicit_quantization(onnx_path: Path) -> bool:
    """Whether the model already carries Q/DQ nodes (so INT8 needs no calibrator)."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    return any(node.op_type in _QDQ_OPS for node in model.graph.node)


# --------------------------------------------------------------------------- build steps


def _create_network(trt: Any, builder: Any, version: TrtVersion) -> Any:
    flags = 0
    if version.major < 10:  # TensorRT 10 networks are always explicit-batch
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    return builder.create_network(flags)


def _parse_onnx(trt: Any, network: Any, logger: Any, onnx_path: Path) -> None:
    parser = trt.OnnxParser(network, logger)
    if parser.parse_from_file(str(onnx_path)):
        return
    errors = parser_errors(parser)
    operators = unsupported_operators(errors)
    summary = "; ".join(e["description"] for e in errors[:3]) or "no details reported"
    hint = (
        f"TensorRT cannot import operator(s): {', '.join(operators)}. Replace them in the model, "
        "try a different export.opset, or provide a TensorRT plugin."
        if operators
        else "Check the parser errors in the details; validate the ONNX model first."
    )
    raise EngineBuildError(
        f"TensorRT could not parse {onnx_path.name}: {summary}",
        hint=hint,
        details={"parser_errors": errors, "unsupported_operators": operators},
    )


def _check_network_io(network: Any, signature: ModelSignature) -> None:
    names = [network.get_input(i).name for i in range(int(network.num_inputs))]
    expected = [spec.name for spec in signature.inputs]
    if sorted(names) != sorted(expected):
        raise EngineBuildError(
            f"the parsed network's inputs {names} differ from the model's {expected}",
            hint="The ONNX file does not match the configuration; re-export it.",
        )


def _shapes_json(shapes: ProfileShapes) -> dict[str, dict[str, list[int]]]:
    return {
        name: {"min": list(low), "opt": list(opt), "max": list(high)}
        for name, (low, opt, high) in shapes.items()
    }


def _add_profiles(trt: Any, builder: Any, cfg: Any, profiles: list[ProfileShapes]) -> None:
    for index, shapes in enumerate(profiles):
        profile = builder.create_optimization_profile()
        for name, (low, opt, high) in shapes.items():
            profile.set_shape(name, low, opt, high)
        if not profile.is_valid():
            raise EngineBuildError(
                f"optimization profile {index} is not valid for this network",
                hint="Check that every dynamic input has min <= opt <= max and matches the model.",
                details={
                    "profile": index,
                    "shapes": _shapes_json(shapes),
                },
            )
        cfg.add_optimization_profile(profile)


def _precision_flags(
    trt: Any,
    builder: Any,
    cfg: Any,
    precision: Precision,
    *,
    int8_fp16_fallback: bool,
    warnings: list[str],
) -> list[str]:
    flags: list[str] = []
    wants_fp16 = precision is Precision.FP16 or (precision is Precision.INT8 and int8_fp16_fallback)
    if wants_fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
        flags.append("FP16")
        if getattr(builder, "platform_has_fast_fp16", True) is False:
            warnings.append("this GPU has no fast FP16 path; the FP16 engine will not be faster")
    if precision is Precision.INT8:
        cfg.set_flag(trt.BuilderFlag.INT8)
        flags.append("INT8")
        if getattr(builder, "platform_has_fast_int8", True) is False:
            warnings.append("this GPU has no fast INT8 path; the INT8 engine will not be faster")
    return flags


class _Configured:
    """The builder configuration plus what was chosen, for the result record."""

    def __init__(self, cfg: Any, flags: list[str], level: int | None) -> None:
        self.cfg = cfg
        self.flags = flags
        self.level = level
        self.timing_cache: Any | None = None
        self.cache_path: Path | None = None
        self.loaded_cache_bytes = 0


def _configure(
    trt: Any,
    builder: Any,
    settings: Any,
    precision: Precision,
    profiles: list[ProfileShapes],
    calibrator: Any | None,
    warnings: list[str],
) -> _Configured:
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, settings.workspace_mb * 1024 * 1024)
    level: int | None = None
    if hasattr(cfg, "builder_optimization_level"):
        cfg.builder_optimization_level = settings.optimization_level
        level = settings.optimization_level
    flags = _precision_flags(
        trt,
        builder,
        cfg,
        precision,
        int8_fp16_fallback=settings.int8_fp16_fallback,
        warnings=warnings,
    )
    if calibrator is not None and precision is Precision.INT8:
        cfg.int8_calibrator = calibrator
    _add_profiles(trt, builder, cfg, profiles)

    configured = _Configured(cfg, flags, level)
    if settings.timing_cache_path is not None:
        configured.cache_path = settings.timing_cache_path
        existing = configured.cache_path.read_bytes() if configured.cache_path.is_file() else b""
        configured.loaded_cache_bytes = len(existing)
        configured.timing_cache = cfg.create_timing_cache(existing)
        cfg.set_timing_cache(configured.timing_cache, False)
    return configured


def _build_plan(
    builder: Any, network: Any, cfg: Any, precision: Precision, builder_log: list[str]
) -> tuple[bytes, float]:
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, cfg)
    elapsed = round(time.perf_counter() - started, 3)
    if serialized is None:
        raise EngineBuildError(
            f"TensorRT failed to build a {precision.value} engine",
            hint="See the builder log in the details; common causes are unsupported layers, too "
            "small a workspace (tensorrt.workspace_mb), or an invalid optimization profile.",
            details={"builder_log": builder_log, "precision": precision.value},
        )
    return bytes(serialized), elapsed


def _persist_plan(plan: bytes, output_path: Path) -> str:
    """Write the plan next to its destination and publish it without overwriting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f".{output_path.name}.{secrets.token_hex(4)}.tmp")
    try:
        tmp.write_bytes(plan)
        digest = sha256_file(tmp)
        publish_new(tmp, output_path)
    finally:
        tmp.unlink(missing_ok=True)
    return digest


def _inspect_plan(trt: Any, logger: Any, plan: bytes) -> EngineInfo:
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    if engine is None:
        raise EngineBuildError("the engine was built but could not be deserialized for inspection")
    return describe_engine(trt, engine)


def build_engine(
    onnx_path: Path,
    signature: ModelSignature,
    config: TrtshipConfig,
    precision: Precision,
    output_path: Path,
    *,
    calibrator: Any | None = None,
    trt: Any | None = None,
) -> BuildResult:
    """Build a TensorRT plan at ``precision`` and write it to ``output_path`` (must not exist).

    Raises :class:`EnvironmentUnavailableError` without a usable GPU/TensorRT (never falls back to
    the CPU) and :class:`EngineBuildError` for parse/build failures, with the unsupported operators
    in ``details``.
    """
    settings = config.tensorrt
    trt = trt if trt is not None else load_tensorrt(purpose="building a TensorRT engine")
    version = parse_version(str(trt.__version__))
    check_supported(version)
    if output_path.exists():
        raise EngineBuildError(f"refusing to overwrite existing engine: {output_path}")
    profiles = build_profile_shapes(signature, settings.profiles)
    needs_calibration = precision is Precision.INT8 and calibrator is None
    if needs_calibration and not has_explicit_quantization(onnx_path):
        raise EngineBuildError(
            "INT8 needs a calibrator or a model with Q/DQ nodes, and neither was given",
            hint="Run the calibrate stage (a `calibration` config section) before building INT8.",
        )
    _select_cuda_device(settings.device_index)

    builder_log: list[str] = []
    warnings: list[str] = []
    logger = _logger(trt, builder_log)
    builder = trt.Builder(logger)
    network = _create_network(trt, builder, version)
    _parse_onnx(trt, network, logger, onnx_path)
    _check_network_io(network, signature)

    configured = _configure(trt, builder, settings, precision, profiles, calibrator, warnings)
    plan, elapsed = _build_plan(builder, network, configured.cfg, precision, builder_log)
    saved = 0
    if configured.timing_cache is not None and configured.cache_path is not None:
        blob = bytes(configured.timing_cache.serialize())
        atomic_write_bytes(configured.cache_path, blob)
        saved = len(blob)

    info = _inspect_plan(trt, logger, plan)
    digest = _persist_plan(plan, output_path)
    log.info(
        "built %s engine %s (%d bytes, %.1fs)",
        precision.value,
        output_path.name,
        len(plan),
        elapsed,
    )
    return BuildResult(
        precision=precision,
        path=str(output_path),
        sha256=digest,
        size_bytes=len(plan),
        build_time_s=elapsed,
        tensorrt_version=str(trt.__version__),
        builder_flags=configured.flags,
        workspace_mb=settings.workspace_mb,
        optimization_level=configured.level,
        profiles=[_shapes_json(p) for p in profiles],
        calibrator=type(calibrator).__name__ if calibrator is not None else None,
        timing_cache=TimingCacheInfo(
            path=str(configured.cache_path) if configured.cache_path else None,
            loaded_bytes=configured.loaded_cache_bytes,
            saved_bytes=saved,
        ),
        engine=info,
        warnings=warnings,
        builder_log=builder_log,
    )
