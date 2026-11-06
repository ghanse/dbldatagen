from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from conftest import EXAMPLES, SPEC_DIR, read_json
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from dbldatagen_profiler.model import SPEC_VERSION, load_profile, render_json_schema


SCHEMA_PATH = SPEC_DIR / "profile.schema.json"
REGENERATE = "uv run --package dbldatagen-profiler dbldatagen-profile schema -o profiler/spec/profile.schema.json"


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    """Returns a JSON Schema validator for the committed profile schema."""
    schema = read_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_committed_schema_is_current() -> None:
    assert SCHEMA_PATH.read_text(encoding="utf-8") == render_json_schema(), f"schema is stale; run: {REGENERATE}"


def test_schema_names_spec_version() -> None:
    assert SPEC_VERSION in read_json(SCHEMA_PATH)["$comment"]


def test_there_are_examples() -> None:
    assert {path.name for path in EXAMPLES} >= {"single_table.json", "two_tables_fk.json"}


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.name)
def test_example_matches_schema_and_model(validator: Draft202012Validator, path: Path) -> None:
    validator.validate(read_json(path))
    load_profile(path.read_text(encoding="utf-8"), strict=True)


def test_schema_accepts_unknown_fields(validator: Draft202012Validator, minimal_profile: dict[str, Any]) -> None:
    minimal_profile["future_block"] = {"a": 1}
    minimal_profile["tables"][0]["future_table_field"] = True
    minimal_profile["tables"][0]["columns"][0]["future_column_field"] = [1, 2]
    validator.validate(minimal_profile)
    assert "additionalProperties" not in SCHEMA_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize(("version", "ok"), [("0.1.0", True), ("0.9.4", True), ("1.0.0", False), ("10.1.0", False)])
def test_schema_pins_major_version(
    validator: Draft202012Validator, minimal_profile: dict[str, Any], version: str, ok: bool
) -> None:
    minimal_profile["spec_version"] = version
    assert validator.is_valid(minimal_profile) is ok


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("engine"),
        lambda p: p["tables"][0].update(row_count=-1),
        lambda p: p["tables"][0]["columns"][0].update(logical_type="float128"),
        lambda p: p["tables"][0].update(columns=[]),
        lambda p: p["tables"][0].update(primary_key={"columns": [], "confidence": 1.0}),
    ],
)
def test_schema_rejects_invalid_profile(
    validator: Draft202012Validator, minimal_profile: dict[str, Any], mutate: Callable[[dict[str, Any]], object]
) -> None:
    validator.validate(minimal_profile)
    mutate(minimal_profile)
    with pytest.raises(ValidationError):
        validator.validate(minimal_profile)
