from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from trtship.config import (
    Precision,
    TrtshipConfig,
    apply_overrides,
    config_hash,
    dump_config_yaml,
    env_overrides,
    load_config,
)
from trtship.errors import ConfigError

Writer = Callable[[dict[str, Any], str], Path]


def test_minimal_config_loads_with_defaults(
    config_dict: dict[str, Any], write_config: Writer
) -> None:
    cfg = load_config(write_config(config_dict, "config.yaml"))
    assert cfg.model.name == "tiny"
    assert cfg.export.opset == 17
    assert cfg.tensorrt.precisions == [Precision.FP32]
    assert cfg.model.inputs[0].dynamic_axes == {0: "batch"}
    assert (
        cfg.validation.tolerances[Precision.INT8].cosine_min
        < cfg.validation.tolerances[Precision.FP32].cosine_min
    )
    assert cfg.warnings() == []


def test_unknown_keys_are_rejected(config_dict: dict[str, Any], write_config: Writer) -> None:
    config_dict["tensorrt"]["worksapce_mb"] = 1
    with pytest.raises(ConfigError) as info:
        load_config(write_config(config_dict, "c.yaml"))
    assert "tensorrt.worksapce_mb" in str(info.value)
    assert info.value.details["errors"][0]["loc"] == "tensorrt.worksapce_mb"


def test_all_errors_are_reported_together(write_config: Writer) -> None:
    data = {"model": {"name": "bad name!", "kind": "module", "inputs": []}}
    with pytest.raises(ConfigError) as info:
        load_config(write_config(data, "c.yaml"))
    locs = {e["loc"] for e in info.value.details["errors"]}
    assert {"model.name", "model.inputs"} <= locs


@pytest.mark.parametrize(
    ("mutation", "fragment"),
    [
        (lambda c: c["model"].update(kind="checkpoint"), "requires both 'factory' and 'path'"),
        (lambda c: c["model"].update(kind="torchscript", factory=None), "requires 'path'"),
        (lambda c: c["model"].update(factory="not a factory"), "package.module:callable"),
        (lambda c: c["model"]["inputs"].append(dict(c["model"]["inputs"][0])), "unique"),
        (lambda c: c["model"]["inputs"][0].update(shape=[0, 3]), ">= 1"),
        (lambda c: c["model"]["inputs"][0].update(shape=[True, 3]), "shape"),
        (lambda c: c["model"]["inputs"][0].update(shape=["not-ident!", 3]), "identifier"),
        (lambda c: c["model"]["inputs"][0].update(dtype="float64"), "dtype"),
        (lambda c: c["tensorrt"].update(precisions=["fp32", "fp32"]), "unique"),
        (lambda c: c["tensorrt"].update(precisions=["int8"]), "requires a 'calibration' section"),
        (lambda c: c["tensorrt"].update(workspace_mb=1), "workspace_mb"),
        (lambda c: c.setdefault("export", {}).update(opset=3), "opset"),
        (lambda c: c.update(benchmark={"precisions": ["fp16"]}), "subset"),
        (lambda c: c.update(triton={"precision": "fp16"}), "one of tensorrt.precisions"),
        (lambda c: c.update(triton={"http_port": 9, "grpc_port": 9}), "distinct"),
    ],
)
def test_validation_failures(
    config_dict: dict[str, Any],
    write_config: Writer,
    mutation: Callable[[dict[str, Any]], object],
    fragment: str,
) -> None:
    mutation(config_dict)
    with pytest.raises(ConfigError) as info:
        load_config(write_config(config_dict, "c.yaml"))
    assert fragment in str(info.value)


@pytest.mark.parametrize(
    ("profile", "fragment"),
    [
        ({"x": {"min": [1, 16], "opt": [8, 16], "max": [4, 16]}}, "min <= opt <= max"),
        ({"x": {"min": [1, 16, 1], "opt": [1, 16, 1], "max": [1, 16, 1]}}, "rank"),
        ({"x": {"min": [1, 8], "opt": [2, 8], "max": [4, 8]}}, "static (16)"),
        ({"nope": {"min": [1], "opt": [1], "max": [1]}}, "unknown input"),
        ({"x": {"min": [1, 16], "opt": [4, 16]}}, "max"),
    ],
)
def test_profile_validation(
    config_dict: dict[str, Any], write_config: Writer, profile: dict[str, Any], fragment: str
) -> None:
    config_dict["tensorrt"]["profiles"] = [{"inputs": profile}]
    with pytest.raises(ConfigError) as info:
        load_config(write_config(config_dict, "c.yaml"))
    assert fragment in str(info.value)


def test_dynamic_input_without_profile_warns(
    config_dict: dict[str, Any], write_config: Writer
) -> None:
    config_dict["tensorrt"]["profiles"] = []
    cfg = load_config(write_config(config_dict, "c.yaml"))
    assert any("no tensorrt.profiles entry" in w for w in cfg.warnings())


def test_trust_source_warns(config_dict: dict[str, Any], write_config: Writer) -> None:
    config_dict["model"]["trust_source"] = True
    cfg = load_config(write_config(config_dict, "c.yaml"))
    assert any("arbitrary code" in w for w in cfg.warnings())


def test_int8_requires_calibration_and_synthetic_needs_opt_in(
    config_dict: dict[str, Any], write_config: Writer
) -> None:
    config_dict["tensorrt"]["precisions"] = ["fp32", "int8"]
    config_dict["calibration"] = {"dataset": "synthetic"}
    with pytest.raises(ConfigError, match="not representative"):
        load_config(write_config(config_dict, "c.yaml"))
    config_dict["calibration"]["allow_synthetic"] = True
    cfg = load_config(write_config(config_dict, "c.yaml"))
    assert cfg.calibration is not None
    assert cfg.calibration.method == "entropy2"


def test_calibration_dataset_needs_path(config_dict: dict[str, Any], write_config: Writer) -> None:
    config_dict["tensorrt"]["precisions"] = ["int8"]
    config_dict["calibration"] = {"dataset": "images"}
    with pytest.raises(ConfigError, match="requires 'path'"):
        load_config(write_config(config_dict, "c.yaml"))


def test_calibration_input_name_must_exist(
    config_dict: dict[str, Any], write_config: Writer
) -> None:
    config_dict["calibration"] = {"dataset": "numpy", "path": "d.npy", "input_name": "ghost"}
    with pytest.raises(ConfigError, match="not a model input"):
        load_config(write_config(config_dict, "c.yaml"))


def test_preprocessing_mean_std_must_pair(
    config_dict: dict[str, Any], write_config: Writer
) -> None:
    config_dict["calibration"] = {
        "dataset": "numpy",
        "path": "d.npy",
        "preprocessing": {"mean": [0.5]},
    }
    with pytest.raises(ConfigError, match="together"):
        load_config(write_config(config_dict, "c.yaml"))


def test_partial_tolerance_overrides_keep_other_precision_defaults(
    config_dict: dict[str, Any], write_config: Writer
) -> None:
    config_dict["validation"] = {
        "tolerances": {"fp16": {"atol": 0.1, "rtol": 0.1, "cosine_min": 0.9}}
    }
    cfg = load_config(write_config(config_dict, "c.yaml"))
    assert cfg.validation.tolerances[Precision.FP16].atol == 0.1
    assert set(cfg.validation.tolerances) == set(Precision)


def test_input_paths_resolve_against_config_dir_and_output_paths_against_cwd(
    config_dict: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_dir = tmp_path / "configs" / "examples"
    cfg_dir.mkdir(parents=True)
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    config_dict["model"].update(kind="torchscript", path="../../models/m.pt")
    path = cfg_dir / "c.yaml"
    path.write_text(yaml.safe_dump(config_dict))
    cfg = load_config(path)
    assert cfg.model.path is not None
    assert cfg.model.path.resolve() == (tmp_path / "models" / "m.pt").resolve()
    assert cfg.artifacts.root == workdir / "runs"
    assert cfg.artifacts.root.is_absolute()


def test_overrides_apply_in_order_env_then_cli(
    config_dict: dict[str, Any], write_config: Writer
) -> None:
    path = write_config(config_dict, "c.yaml")
    cfg = load_config(
        path,
        environ={"TRTSHIP__TENSORRT__WORKSPACE_MB": "2048", "TRTSHIP__SEED": "7", "OTHER": "x"},
        overrides=["tensorrt.workspace_mb=512", "export.opset=18"],
    )
    assert cfg.tensorrt.workspace_mb == 512  # CLI beats env
    assert cfg.seed == 7  # env beats file default
    assert cfg.export.opset == 18


def test_override_values_are_yaml_typed() -> None:
    merged = apply_overrides({}, ["a.b=[1, 2]", "a.c=true", "a.d=null", "a.e=hello world"])
    assert merged == {"a": {"b": [1, 2], "c": True, "d": None, "e": "hello world"}}


@pytest.mark.parametrize("bad", ["novalue", "=x", ".a=1", "a..b=1", "a.b=[unclosed"])
def test_bad_overrides_are_config_errors(bad: str) -> None:
    with pytest.raises(ConfigError):
        apply_overrides({}, [bad])


def test_override_cannot_descend_into_a_scalar() -> None:
    with pytest.raises(ConfigError, match="not a mapping"):
        apply_overrides({"a": 1}, ["a.b=2"])


def test_env_overrides_translation() -> None:
    assert env_overrides(
        {"TRTSHIP__A__B_C": "1", "TRTSHIP_LOG_LEVEL": "DEBUG", "TRTSHIP__": "x"}
    ) == ["a.b_c=1"]


def test_yaml_errors_include_location(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  name: [unclosed\n")
    with pytest.raises(ConfigError, match=r"line \d+"):
        load_config(path)


def test_missing_and_non_mapping_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        load_config(path)
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(ConfigError, match="model"):  # empty file -> missing required 'model'
        load_config(empty)


def test_dump_and_reload_round_trips(
    config_dict: dict[str, Any], write_config: Writer, tmp_path: Path
) -> None:
    cfg = load_config(write_config(config_dict, "c.yaml"))
    snapshot = tmp_path / "snapshot.yaml"
    snapshot.write_text(dump_config_yaml(cfg))
    again = load_config(snapshot)
    assert again == cfg
    assert config_hash(again) == config_hash(cfg)


def test_config_hash_changes_with_content(
    config_dict: dict[str, Any], write_config: Writer
) -> None:
    a = load_config(write_config(config_dict, "a.yaml"))
    b = load_config(write_config(config_dict, "b.yaml"), overrides=["seed=1"])
    assert config_hash(a) != config_hash(b)


def test_config_is_immutable(config_dict: dict[str, Any], write_config: Writer) -> None:
    cfg = load_config(write_config(config_dict, "c.yaml"))
    with pytest.raises(Exception, match="frozen"):
        cfg.seed = 5  # type: ignore[misc]


def test_direct_model_validate_without_context(config_dict: dict[str, Any]) -> None:
    cfg = TrtshipConfig.model_validate(config_dict)
    assert cfg.model.name == "tiny"
