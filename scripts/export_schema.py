"""Write the configuration JSON Schema to configs/schemas/trtship.schema.json."""

from __future__ import annotations

from pathlib import Path

from trtship.config import config_schema_json

TARGET = Path(__file__).resolve().parent.parent / "configs" / "schemas" / "trtship.schema.json"

if __name__ == "__main__":
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(config_schema_json(), encoding="utf-8")
    print(f"wrote {TARGET}")
