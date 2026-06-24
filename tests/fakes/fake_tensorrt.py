"""A fake of the parts of the TensorRT Python API that trtship calls.

**This is not TensorRT.** It exists so trtship's *translation* logic (configuration -> builder
calls, error reporting, engine description) can be unit-tested without a GPU. It records every
call so tests can assert exactly what would have been sent to the real builder. What it cannot
tell us is whether real TensorRT accepts those calls; that is what the ``@pytest.mark.tensorrt``
tests in ``tests/gpu`` are for, and they only run on a machine with TensorRT and a GPU.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

Triple = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


class Severity(enum.IntEnum):
    INTERNAL_ERROR = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    VERBOSE = 4


class ILogger:
    Severity = Severity

    def __init__(self) -> None:
        pass

    def log(self, severity: int, msg: str) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class IInt8EntropyCalibrator2:
    def __init__(self) -> None:
        pass


class IInt8MinMaxCalibrator:
    def __init__(self) -> None:
        pass


class DataType(enum.Enum):
    FLOAT = 0
    HALF = 1
    INT8 = 2
    INT32 = 3
    BOOL = 4
    UINT8 = 5
    INT64 = 6
    BF16 = 7  # a type trtship does not support


class TensorIOMode(enum.Enum):
    NONE = 0
    INPUT = 1
    OUTPUT = 2


class BuilderFlag(enum.Enum):
    FP16 = 1
    INT8 = 2


class MemoryPoolType(enum.Enum):
    WORKSPACE = 0


class NetworkDefinitionCreationFlag(enum.IntEnum):
    EXPLICIT_BATCH = 0


@dataclass
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: DataType = DataType.FLOAT


@dataclass
class ParserError:
    message: str
    node_index: int = 0

    def code(self) -> str:
        return "ErrorCode.UNSUPPORTED_NODE"

    def desc(self) -> str:
        return self.message

    def node(self) -> int:
        return self.node_index


class FakeNetwork:
    def __init__(self, input_names: list[str]) -> None:
        self._inputs = input_names

    @property
    def num_inputs(self) -> int:
        return len(self._inputs)

    def input_names(self) -> list[str]:
        return list(self._inputs)

    def get_input(self, index: int) -> Any:
        return SimpleNamespace(name=self._inputs[index])


class FakeOnnxParser:
    def __init__(self, network: FakeNetwork, logger: Any, calls: FakeCalls) -> None:
        self._errors = list(calls.parse_errors)
        self._calls = calls

    def parse_from_file(self, path: str) -> bool:
        self._calls.parsed.append(path)
        return not self._errors

    @property
    def num_errors(self) -> int:
        return len(self._errors)

    def get_error(self, index: int) -> ParserError:
        return self._errors[index]


class FakeProfile:
    def __init__(self, valid: bool) -> None:
        self.shapes: dict[str, Triple] = {}
        self._valid = valid

    def set_shape(self, name: str, min: Any, opt: Any, max: Any) -> None:
        self.shapes[name] = (tuple(min), tuple(opt), tuple(max))

    def is_valid(self) -> bool:
        return self._valid


class FakeTimingCache:
    def __init__(self, initial: bytes) -> None:
        self.initial = initial

    def serialize(self) -> bytes:
        return self.initial + b"|tactics"


class FakeBuilderConfig:
    def __init__(self, calls: FakeCalls, *, optimization_level: bool) -> None:
        self._calls = calls
        self.flags: list[BuilderFlag] = []
        self.profiles: list[FakeProfile] = []
        self.int8_calibrator: Any = None
        self.timing_cache: FakeTimingCache | None = None
        self.calibration_profile: FakeProfile | None = None
        self.workspace_bytes: int | None = None
        if optimization_level:
            self.builder_optimization_level = 3

    def set_memory_pool_limit(self, kind: MemoryPoolType, size: int) -> None:
        assert kind is MemoryPoolType.WORKSPACE
        self.workspace_bytes = size

    def set_flag(self, flag: BuilderFlag) -> None:
        self.flags.append(flag)

    def add_optimization_profile(self, profile: FakeProfile) -> int:
        self.profiles.append(profile)
        return len(self.profiles) - 1

    def set_calibration_profile(self, profile: FakeProfile) -> bool:
        self.calibration_profile = profile
        return True

    def create_timing_cache(self, blob: bytes) -> FakeTimingCache:
        return FakeTimingCache(blob)

    def set_timing_cache(self, cache: FakeTimingCache, ignore_mismatch: bool) -> bool:
        self.timing_cache = cache
        return True


@dataclass
class FakeCalls:
    """Everything the fake was asked to do, for assertions."""

    parse_errors: list[ParserError] = field(default_factory=list)
    parsed: list[str] = field(default_factory=list)
    network_flags: list[int] = field(default_factory=list)
    configs: list[FakeBuilderConfig] = field(default_factory=list)
    builds: int = 0
    calibration_batches: list[list[int]] = field(default_factory=list)
    used_cache: bool = False


class FakeBuilder:
    def __init__(self, logger: Any, calls: FakeCalls, options: FakeOptions) -> None:
        self._logger = logger
        self._calls = calls
        self._options = options
        self.platform_has_fast_fp16 = options.fast_fp16
        self.platform_has_fast_int8 = options.fast_int8

    def create_network(self, flags: int = 0) -> FakeNetwork:
        self._calls.network_flags.append(flags)
        return FakeNetwork(self._options.network_inputs)

    def create_builder_config(self) -> FakeBuilderConfig:
        cfg = FakeBuilderConfig(
            self._calls, optimization_level=self._options.has_optimization_level
        )
        self._calls.configs.append(cfg)
        return cfg

    def create_optimization_profile(self) -> FakeProfile:
        return FakeProfile(valid=not self._options.invalid_profile)

    def _simulate_calibration(self, calibrator: Any, network: FakeNetwork) -> None:
        """What TensorRT does with a calibrator: use its cache, or pull batches then write one."""
        if calibrator.read_calibration_cache() is not None:
            self._calls.used_cache = True
            return
        served = 0
        while (pointers := calibrator.get_batch(network.input_names())) is not None:
            self._calls.calibration_batches.append(list(pointers))
            served += 1
        if served and self._options.calibration_writes_cache:
            calibrator.write_calibration_cache(b"FAKE-SCALES:" + str(served).encode())

    def build_serialized_network(
        self, network: FakeNetwork, cfg: FakeBuilderConfig
    ) -> bytes | None:
        self._calls.builds += 1
        self._logger.log(Severity.WARNING, "Tactic Device request: 512MB")
        self._logger.log(Severity.INFO, "chatter that must not be captured")
        calibrator = cfg.int8_calibrator
        if calibrator is not None and BuilderFlag.INT8 in cfg.flags:
            self._simulate_calibration(calibrator, network)
        if self._options.build_fails:
            self._logger.log(Severity.ERROR, "Layer 'conv1' has no valid tactics")
            return None
        return (
            b"FAKEPLAN"
            + json.dumps(
                {
                    "flags": sorted(f.name for f in cfg.flags),
                    "workspace": cfg.workspace_bytes,
                    "profiles": [
                        {k: [list(x) for x in v] for k, v in p.shapes.items()} for p in cfg.profiles
                    ],
                    "calibrated": cfg.int8_calibrator is not None,
                },
                sort_keys=True,
            ).encode()
        )


class FakeExecutionContext:
    """Executes by calling ``options.compute`` on the host arrays behind the bound addresses."""

    def __init__(self, engine: Any, options: FakeOptions) -> None:
        self._engine = engine
        self._options = options
        self.input_shapes: dict[str, tuple[int, ...]] = {}
        self.addresses: dict[str, int] = {}
        self.executions = 0

    def set_input_shape(self, name: str, shape: tuple[int, ...]) -> bool:
        low, _, high = self._engine.get_tensor_profile_shape(name, 0)
        ok = all(lo <= d <= hi for lo, d, hi in zip(low, shape, high, strict=True))
        if ok:
            self.input_shapes[name] = tuple(shape)
        return ok

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        declared = self._engine.get_tensor_shape(name)
        if -1 not in declared:
            return tuple(declared)
        if self._options.data_dependent_outputs:
            return tuple(declared)
        first = next(iter(self.input_shapes.values()), (1,))
        return tuple(
            first[0] if d == -1 and i == 0 else (d if d != -1 else 1)
            for i, d in enumerate(declared)
        )

    def set_tensor_address(self, name: str, pointer: int) -> bool:
        self.addresses[name] = pointer
        return True

    def execute_async_v3(self, stream: int) -> bool:
        self.executions += 1
        if self._options.execute_fails:
            return False
        memory = self._options.memory
        inputs = {
            t.name: memory.by_ptr[self.addresses[t.name]] for t in self._options.engine_inputs
        }
        results = self._options.compute(inputs)
        for spec in self._options.engine_outputs:
            memory.by_ptr[self.addresses[spec.name]][...] = results[spec.name]
        return True


class FakeEngine:
    """Engine using the TensorRT >= 8.5 tensor API."""

    def __init__(self, options: FakeOptions, profiles: list[FakeProfile]) -> None:
        self._options = options
        self._profiles = profiles
        self.num_optimization_profiles = max(len(profiles), 1)
        self._tensors = [*options.engine_inputs, *options.engine_outputs]
        self._input_names = {t.name for t in options.engine_inputs}
        self.contexts: list[FakeExecutionContext] = []

    @property
    def num_io_tensors(self) -> int:
        return len(self._tensors)

    def get_tensor_name(self, index: int) -> str:
        return self._tensors[index].name

    def get_tensor_mode(self, name: str) -> TensorIOMode:
        return TensorIOMode.INPUT if name in self._input_names else TensorIOMode.OUTPUT

    def get_tensor_dtype(self, name: str) -> DataType:
        return next(t for t in self._tensors if t.name == name).dtype

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return next(t for t in self._tensors if t.name == name).shape

    def get_tensor_profile_shape(self, name: str, profile: int) -> Triple:
        return self._profiles[profile].shapes[name]

    def create_execution_context(self) -> FakeExecutionContext | None:
        if self._options.context_fails:
            return None
        context = FakeExecutionContext(self, self._options)
        self.contexts.append(context)
        return context


class FakeBindingsEngine:
    """Engine using the pre-8.5 bindings API."""

    def __init__(self, options: FakeOptions, profiles: list[FakeProfile]) -> None:
        self._profiles = profiles
        self.num_optimization_profiles = max(len(profiles), 1)
        self._bindings = [*options.engine_inputs, *options.engine_outputs]
        self._n_inputs = len(options.engine_inputs)

    @property
    def num_bindings(self) -> int:
        return len(self._bindings)

    def get_binding_name(self, index: int) -> str:
        return self._bindings[index].name

    def binding_is_input(self, index: int) -> bool:
        return index < self._n_inputs

    def get_binding_dtype(self, index: int) -> DataType:
        return self._bindings[index].dtype

    def get_binding_shape(self, index: int) -> tuple[int, ...]:
        return self._bindings[index].shape

    def get_profile_shape(self, profile: int, index: int) -> Triple:
        return self._profiles[profile].shapes[self._bindings[index].name]


class FakeRuntime:
    def __init__(self, logger: Any, calls: FakeCalls, options: FakeOptions) -> None:
        self._calls = calls
        self._options = options

    def deserialize_cuda_engine(self, plan: bytes) -> Any:
        if self._options.deserialize_fails:
            return None
        profiles = self._calls.configs[-1].profiles
        cls = FakeEngine if self._options.tensor_api else FakeBindingsEngine
        return cls(self._options, profiles)


@dataclass
class FakeOptions:
    version: str = "10.3.0.26"
    network_inputs: list[str] = field(default_factory=lambda: ["x"])
    engine_inputs: list[TensorSpec] = field(default_factory=lambda: [TensorSpec("x", (-1, 16))])
    engine_outputs: list[TensorSpec] = field(
        default_factory=lambda: [TensorSpec("output", (-1, 4))]
    )
    parse_errors: list[ParserError] = field(default_factory=list)
    build_fails: bool = False
    invalid_profile: bool = False
    deserialize_fails: bool = False
    fast_fp16: bool = True
    fast_int8: bool = True
    has_optimization_level: bool = True
    tensor_api: bool = True
    calibration_writes_cache: bool = True
    memory: Any = None  # a HostMemory shared with the executor under test
    compute: Any = None  # dict[str, ndarray] -> dict[str, ndarray], run by execute_async_v3
    execute_fails: bool = False
    context_fails: bool = False
    data_dependent_outputs: bool = False


def make_fake_trt(options: FakeOptions | None = None) -> tuple[Any, FakeCalls]:
    """A module-like object exposing the API surface trtship uses, plus its call recorder."""
    opts = options or FakeOptions()
    calls = FakeCalls(parse_errors=list(opts.parse_errors))

    module = SimpleNamespace(
        __version__=opts.version,
        ILogger=ILogger,
        IInt8EntropyCalibrator2=IInt8EntropyCalibrator2,
        IInt8MinMaxCalibrator=IInt8MinMaxCalibrator,
        DataType=DataType,
        TensorIOMode=TensorIOMode,
        BuilderFlag=BuilderFlag,
        MemoryPoolType=MemoryPoolType,
        NetworkDefinitionCreationFlag=NetworkDefinitionCreationFlag,
        Builder=lambda logger: FakeBuilder(logger, calls, opts),
        OnnxParser=lambda network, logger: FakeOnnxParser(network, logger, calls),
        Runtime=lambda logger: FakeRuntime(logger, calls, opts),
    )
    return module, calls


class HostBuffers:
    """A :class:`trtship.calibration.DeviceBuffers` that keeps 'device' memory on the host, so the
    calibration data path can be tested without a GPU. It records every upload."""

    def __init__(self) -> None:
        self.uploads: list[Any] = []

    def upload(self, array: Any) -> tuple[object, int]:
        copy = np.ascontiguousarray(array).copy()
        self.uploads.append(copy)
        return copy, int(copy.ctypes.data)


class HostMemory:
    """A :class:`trtship.tensorrt.DeviceMemory` whose 'device' buffers are host arrays, addressable
    by their pointers so the fake execution context can read and write them."""

    stream = 0

    def __init__(self) -> None:
        self.by_ptr: dict[int, Any] = {}

    def allocate(self, shape: tuple[int, ...], dtype: Any) -> tuple[Any, int]:
        array = np.zeros(shape, dtype=dtype)
        pointer = int(array.ctypes.data)
        self.by_ptr[pointer] = array
        return array, pointer

    def upload(self, handle: Any, array: Any) -> None:
        handle[...] = array

    def download(self, handle: Any) -> Any:
        return handle.copy()

    def synchronize(self) -> None:
        return None
