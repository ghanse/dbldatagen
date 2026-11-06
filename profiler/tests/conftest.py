import json
from pathlib import Path
from typing import Any

import pytest


SPEC_DIR = Path(__file__).resolve().parents[1] / "spec"
EXAMPLES = sorted((SPEC_DIR / "examples").glob("*.json"))


@pytest.fixture
def minimal_profile() -> dict[str, Any]:
    """Returns the smallest valid profile as a JSON object."""
    return {
        "spec_version": "0.1.0",
        "created_at": "2026-11-06T12:00:00Z",
        "engine": {"name": "test", "version": "0"},
        "tables": [
            {
                "name": "t",
                "row_count": 2,
                "columns": [
                    {
                        "name": "id",
                        "logical_type": "int64",
                        "source_type": "bigint",
                        "nullable": False,
                        "null_fraction": 0.0,
                    }
                ],
            }
        ],
    }


def read_json(path: Path) -> dict[str, Any]:
    """Returns the parsed JSON content of `path`."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
