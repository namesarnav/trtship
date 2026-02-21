"""Loading, overriding, and serializing configuration."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from trtship.config.models import TrtshipConfig
from trtship.errors import ConfigError

ENV_PREFIX = "TRTSHIP__"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc.strerror or exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        problem = getattr(exc, "problem", None) or str(exc)
        raise ConfigError(f"invalid YAML in {path}{where}: {problem}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def _set_path(data: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    if not all(keys):
        raise ConfigError(f"invalid override key {dotted!r}")
    cursor = data
    for key in keys[:-1]:
        nxt = cursor.setdefault(key, {})
        if not isinstance(nxt, dict):
            raise ConfigError(
                f"override {dotted!r}: {key!r} is a {type(nxt).__name__}, not a mapping"
            )
        cursor = nxt
    cursor[keys[-1]] = value


def apply_overrides(data: Mapping[str, Any], assignments: Iterable[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` assignments; values are parsed as YAML."""
    merged: dict[str, Any] = yaml.safe_load(yaml.safe_dump(dict(data))) or {}
    for assignment in assignments:
        key, sep, raw = assignment.partition("=")
        if not sep or not key.strip():
            raise ConfigError(
                f"invalid override {assignment!r}", hint="Use the form section.key=value"
            )
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"override {assignment!r}: value is not valid YAML") from exc
        _set_path(merged, key.strip(), value)
    return merged


def env_overrides(environ: Mapping[str, str] | None = None) -> list[str]:
    """Translate ``TRTSHIP__SECTION__KEY=value`` variables into dotted override assignments."""
    source = os.environ if environ is None else environ
    return [
        f"{name[len(ENV_PREFIX) :].lower().replace('__', '.')}={value}"
        for name, value in sorted(source.items())
        if name.startswith(ENV_PREFIX) and len(name) > len(ENV_PREFIX)
    ]


def _format_errors(exc: ValidationError) -> list[dict[str, str]]:
    return [
        {
            "loc": ".".join(str(part) for part in err["loc"]) or "<root>",
            "msg": err["msg"].removeprefix("Value error, "),
            "type": err["type"],
        }
        for err in exc.errors(include_url=False)
    ]


def load_config(
    path: Path,
    *,
    overrides: Iterable[str] = (),
    environ: Mapping[str, str] | None = None,
) -> TrtshipConfig:
    """Load and validate a config file.

    Precedence, lowest to highest: defaults < YAML file < ``TRTSHIP__*`` environment
    < ``overrides``.
    """
    path = path.expanduser()
    data = _read_yaml(path)
    data = apply_overrides(data, [*env_overrides(environ), *overrides])
    try:
        return TrtshipConfig.model_validate(data, context={"base_dir": path.resolve().parent})
    except ValidationError as exc:
        errors = _format_errors(exc)
        lines = "\n".join(f"  {e['loc']}: {e['msg']}" for e in errors)
        plural = "s" if len(errors) != 1 else ""
        raise ConfigError(
            f"invalid configuration in {path} ({len(errors)} error{plural}):\n{lines}",
            details={"errors": errors, "path": str(path)},
        ) from exc


def dump_config_yaml(config: TrtshipConfig) -> str:
    """Serialize a resolved config. Re-loading the result yields an equal config."""
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
