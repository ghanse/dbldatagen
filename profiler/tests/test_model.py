import json
from pathlib import Path
from typing import Any

import pytest
from conftest import EXAMPLES, read_json
from pydantic import ValidationError

from dbldatagen_profiler.model import ColumnProfile, Profile, dump_profile, load_profile


def column(**overrides: object) -> dict[str, Any]:
    """Returns a valid column JSON object with `overrides` applied."""
    base = {"name": "c", "logical_type": "int64", "source_type": "bigint", "nullable": True, "null_fraction": 0.0}
    return {**base, **overrides}


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.name)
def test_example_round_trip(path: Path) -> None:
    profile = load_profile(path.read_text(encoding="utf-8"))
    text = dump_profile(profile)
    assert load_profile(text) == profile
    assert json.loads(text) == json.loads(dump_profile(load_profile(read_json(path))))


def test_minimal_profile_omits_unset_blocks(minimal_profile: dict[str, Any]) -> None:
    profile = load_profile(minimal_profile)
    dumped = json.loads(dump_profile(profile))
    assert set(dumped) == {"spec_version", "created_at", "engine", "tables"}
    assert dumped["tables"][0]["columns"][0] == minimal_profile["tables"][0]["columns"][0]


def test_quantile_values_keep_their_json_type() -> None:
    stats = {"quantiles": [{"q": 0.5, "value": 4}, {"q": 0.9, "value": 4.5}]}
    parsed = ColumnProfile.model_validate(column(numeric=stats))
    assert parsed.numeric is not None and parsed.numeric.quantiles is not None
    assert [type(q.value) for q in parsed.numeric.quantiles] == [int, float]


@pytest.mark.parametrize(
    "overrides",
    [
        {"logical_type": "decimal", "source_type": "decimal(10,2)"},
        {"precision": 10, "scale": 2},
        {"logical_type": "decimal", "precision": 2, "scale": 3},
        {"nullable": False, "null_fraction": 0.1},
        {"null_fraction": 1.5},
        {"logical_type": "array", "children": [column(name="item")]},
        {"logical_type": "map", "children": [column(name="key")]},
        {"logical_type": "struct", "children": []},
        {"logical_type": "struct", "children": [column(), column()]},
        {"children": [column()]},
        {"string": {"length": {"min": 1}}},
        {"logical_type": "string", "numeric": {"min": 1}},
        {"logical_type": "boolean", "boolean": {"true_fraction": 2}},
        {"numeric": {"histogram": {"edges": [0, 1, 2], "counts": [1]}}},
        {"numeric": {"histogram": {"edges": [2, 1], "counts": [1]}}},
        {"cardinality": "huge"},
        {"unknown_field": 1},
    ],
)
def test_invalid_column_is_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ColumnProfile.model_validate(column(**overrides))


@pytest.mark.parametrize(
    "table_overrides",
    [
        {"columns": [column(name="id"), column(name="id")]},
        {"primary_key": {"columns": ["missing"], "confidence": 1.0}},
        {
            "check_constraints": [
                {"kind": "comparison", "left": "id", "operator": "<", "right": "missing", "satisfied_fraction": 1}
            ]
        },
        {"check_constraints": [{"kind": "between", "column": "id", "satisfied_fraction": 1}]},
        {"dependencies": [{"columns": ["id", "id"], "method": "cramers_v", "value": -0.5}]},
        {"dependencies": [{"columns": ["id"], "method": "pearson", "value": 0.5}]},
    ],
)
def test_invalid_table_is_rejected(minimal_profile: dict[str, Any], table_overrides: dict[str, Any]) -> None:
    minimal_profile["tables"][0].update(table_overrides)
    with pytest.raises(ValidationError):
        load_profile(minimal_profile)


def relationship(**overrides: object) -> dict[str, Any]:
    """Returns a valid self-referencing relationship JSON object with `overrides` applied."""
    base = {"child_table": "t", "child_columns": ["id"], "parent_table": "t", "parent_columns": ["id"], "coverage": 1}
    return {**base, **overrides}


@pytest.mark.parametrize(
    "rel",
    [
        relationship(parent_table="missing"),
        relationship(child_columns=["nope"]),
        relationship(parent_columns=["id", "id"]),
        relationship(coverage=1.5),
    ],
)
def test_invalid_relationship_is_rejected(minimal_profile: dict[str, Any], rel: dict[str, Any]) -> None:
    minimal_profile["relationships"] = [rel]
    with pytest.raises(ValidationError):
        load_profile(minimal_profile)


def test_duplicate_table_names_are_rejected(minimal_profile: dict[str, Any]) -> None:
    minimal_profile["tables"].append(minimal_profile["tables"][0])
    with pytest.raises(ValidationError):
        load_profile(minimal_profile)


def test_created_at_requires_utc_offset(minimal_profile: dict[str, Any]) -> None:
    minimal_profile["created_at"] = "2026-11-06T12:00:00"
    with pytest.raises(ValidationError):
        Profile.model_validate(minimal_profile)
