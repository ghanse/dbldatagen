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
    assert set(dumped) == {"spec_version", "created_at", "engine", "privacy", "tables"}
    assert dumped["privacy"] == {"min_value_count": 10, "extremes": "exact", "enumerations": "exact"}
    assert dumped["tables"][0]["columns"][0] == minimal_profile["tables"][0]["columns"][0]


def test_quantile_values_keep_their_json_type() -> None:
    stats = {"quantiles": [{"q": 0.5, "value": 4}, {"q": 0.9, "value": 4.5}]}
    parsed = ColumnProfile.model_validate(column(logical_type="float64", source_type="double", numeric=stats))
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
        {"logical_type": "float64", "numeric": {"histogram": {"edges": [0, 1, 2], "counts": [1]}}},
        {"logical_type": "float64", "numeric": {"histogram": {"edges": [2, 1], "counts": [1]}}},
        {
            "logical_type": "decimal",
            "precision": 4,
            "scale": 1,
            "numeric": {"histogram": {"edges": ["2.0", "1.0"], "counts": [1]}},
        },
        {"collection_length": {"min": 1}},
        {"logical_type": "float64", "numeric": {"histogram": {"edges": [True, 2], "counts": [1]}}},
        {"cardinality": "huge"},
    ],
)
def test_invalid_column_is_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ColumnProfile.model_validate(column(**overrides))


@pytest.mark.parametrize(
    "table_overrides",
    [
        {"columns": [column(name="id"), column(name="id")]},
        {"columns": []},
        {"primary_key": {"columns": ["id", "id"], "confidence": 1.0}},
        {"unique_keys": [{"columns": [], "confidence": 1.0}]},
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


@pytest.mark.parametrize(
    ("logical_type", "valid", "invalid"),
    [
        ("int64", ["9223372036854775807", "-1", "0"], [9223372036854775807, 1, "1.0", "01", "1e3"]),
        ("uint64", ["18446744073709551615"], [1]),
        ("decimal", ["12345678901234567890.12", "-0.5", "3"], [0.5, 3, "1.", ".5", "NaN"]),
        ("int32", [1, -7], ["1", 1.5, True]),
        ("float64", [1, 1.5], ["1.5", False]),
    ],
)
def test_numeric_literals_follow_logical_type_encoding(
    logical_type: str, valid: list[object], invalid: list[object]
) -> None:
    extra = {"precision": 38, "scale": 2} if logical_type == "decimal" else {}
    for value in valid:
        stats = {"min": value, "max": value, "quantiles": [{"q": 0.5, "value": value}]}
        ColumnProfile.model_validate(column(logical_type=logical_type, numeric=stats, **extra))
    for value in invalid:
        with pytest.raises(ValidationError):
            ColumnProfile.model_validate(column(logical_type=logical_type, numeric={"max": value}, **extra))


def test_decimal_histogram_edges_are_lossless() -> None:
    edges = ["0.10", "12345678901234567890.10", "12345678901234567890.11"]
    histogram = {"edges": edges, "counts": [1, 1]}
    parsed = ColumnProfile.model_validate(
        column(logical_type="decimal", precision=38, scale=2, numeric={"histogram": histogram})
    )
    assert parsed.numeric is not None and parsed.numeric.histogram is not None
    assert parsed.numeric.histogram.edges == edges
    assert parsed.numeric.histogram.sampled is False


@pytest.mark.parametrize(
    ("logical_type", "valid", "invalid"),
    [
        ("date", ["2026-11-06"], ["2026-11-06T00:00:00", "11/06/2026", 20261106]),
        ("time", ["08:30:00", "08:30:00.123456789"], ["8:30", "08:30:00Z"]),
        ("timestamp", ["2026-11-06T08:30:00Z", "2026-11-06T08:30:00.5-07:00"], ["2026-11-06T08:30:00", "2026-11-06"]),
        ("timestamp_ntz", ["2026-11-06T08:30:00.000001"], ["2026-11-06T08:30:00Z", "2026-11-06T08:30:00+00:00"]),
    ],
)
def test_temporal_literals_follow_iso_format(logical_type: str, valid: list[object], invalid: list[object]) -> None:
    for value in valid:
        ColumnProfile.model_validate(column(logical_type=logical_type, temporal={"min": value}))
    for value in invalid:
        with pytest.raises(ValidationError):
            ColumnProfile.model_validate(column(logical_type=logical_type, temporal={"min": value}))


@pytest.mark.parametrize(
    ("constraint", "ok"),
    [
        ({"kind": "range", "column": "id", "min": "0", "max": "99"}, True),
        ({"kind": "range", "column": "id", "min": 0}, False),
        ({"kind": "enumeration", "column": "id", "values": ["1", "2"], "min_value_count": 10}, True),
        ({"kind": "enumeration", "column": "id", "values": [1, 2], "min_value_count": 10}, False),
    ],
)
def test_constraint_literals_follow_column_type(
    minimal_profile: dict[str, Any], constraint: dict[str, Any], ok: bool
) -> None:
    minimal_profile["tables"][0]["check_constraints"] = [{**constraint, "satisfied_fraction": 1.0}]
    if ok:
        load_profile(minimal_profile)
    else:
        with pytest.raises(ValidationError):
            load_profile(minimal_profile)


def test_expression_dialect_defaults_to_spark(minimal_profile: dict[str, Any]) -> None:
    constraint = {"kind": "expression", "expression": "id > 0", "satisfied_fraction": 1.0}
    minimal_profile["tables"][0]["check_constraints"] = [constraint]
    table = load_profile(minimal_profile).tables[0]
    assert table.check_constraints is not None
    assert table.check_constraints[0].model_dump()["dialect"] == "spark"


def test_fan_out_quantiles_are_integer_counts(minimal_profile: dict[str, Any]) -> None:
    fan_out = {"min": 0, "max": 3, "mean": 1.0, "quantiles": [{"q": 0.5, "value": 1.5}]}
    minimal_profile["relationships"] = [relationship(fan_out=fan_out)]
    with pytest.raises(ValidationError):
        load_profile(minimal_profile)
