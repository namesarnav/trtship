"""JSON Schema for the configuration, kept in the repository and checked for drift in tests."""

from __future__ import annotations

import json
from typing import Any

from trtship.config.models import TrtshipConfig


def config_schema() -> dict[str, Any]:
    schema = TrtshipConfig.model_json_schema()
    schema["title"] = "trtship configuration"
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def config_schema_json() -> str:
    return json.dumps(config_schema(), indent=2, sort_keys=True) + "\n"
