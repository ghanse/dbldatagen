from typing import Any

import pytest
from pyspark.sql import SparkSession

from dbldatagen_profiler.engines.spark import profile_tables
from dbldatagen_profiler.model import Profile
from profiler_bench.catalog import Manifest, load, materialize
from profiler_bench.catalog.manifest import Check, ComparisonCheck, EnumerationCheck, ExpressionCheck, RangeCheck
from profiler_bench.fidelity import StructureScore, score_structure
from profiler_bench.fidelity.scores import StructureCategory


def _baseline(name: str, spark: SparkSession) -> tuple[Profile, Manifest]:
    return profile_tables(load(name, "xs", 0, spark)), materialize(name, "xs", 0).manifest


def _bound(value: object, column: dict[str, Any]) -> object:
    return str(value) if column["logical_type"] in ("int64", "decimal") and value is not None else value


def _constraint(check: Check, columns: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if isinstance(check, RangeCheck):
        column = columns[check.column]
        return {
            "kind": "range",
            "column": check.column,
            "min": _bound(check.min, column),
            "max": _bound(check.max, column),
            "satisfied_fraction": 1.0,
        }
    if isinstance(check, EnumerationCheck):
        return {
            "kind": "enumeration",
            "column": check.column,
            "values": list(check.values),
            "min_value_count": 10,
            "satisfied_fraction": 1.0,
        }
    if isinstance(check, ComparisonCheck):
        return {
            "kind": "comparison",
            "left": check.left,
            "operator": check.op,
            "right": check.right,
            "satisfied_fraction": 1.0,
        }
    assert isinstance(check, ExpressionCheck)
    return {"kind": "expression", "expression": check.sql, "satisfied_fraction": 1.0}


def _oracle(profile: Profile, truth: Manifest) -> Profile:
    """Returns `profile` with every planted key, foreign key, check, dependency, and format added."""
    data = profile.model_dump(mode="json")
    data["privacy"]["allow_expression_constraints"] = True
    for table in data["tables"]:
        planted = truth.table(table["name"])
        columns = {c["name"]: c for c in table["columns"]}
        if planted.primary_key:
            table["primary_key"] = {"columns": list(planted.primary_key), "confidence": 1.0}
        table["unique_keys"] = [{"columns": list(k), "confidence": 1.0} for k in planted.unique_keys] or None
        table["check_constraints"] = [_constraint(c, columns) for c in planted.checks] or None
        table["dependencies"] = [
            {"columns": list(d.columns), "method": d.measure, "value": d.value} for d in planted.dependencies
        ] or None
        for column in planted.columns:
            if column.formats:
                stats = columns[column.name].get("string") or {}
                stats["patterns"] = [{"regex": f.regex, "coverage": f.share} for f in column.formats]
                columns[column.name]["string"] = stats
    data["relationships"] = [
        {
            "child_table": fk.child_table,
            "child_columns": list(fk.child_columns),
            "parent_table": fk.parent_table,
            "parent_columns": list(fk.parent_columns),
            "coverage": fk.coverage,
        }
        for fk in truth.foreign_keys
    ]
    return Profile.model_validate(data)


@pytest.fixture(scope="module")
def retail(spark: SparkSession) -> tuple[Profile, Manifest]:
    return _baseline("retail", spark)


def _category(score: StructureScore, table: str, name: str) -> StructureCategory:
    return next(t for t in score.tables if t.name == table).categories[name]


def test_baseline_reports_missing_structure_as_not_recovered(retail: tuple[Profile, Manifest]) -> None:
    profile, truth = retail
    score = score_structure(profile, truth)
    keys = _category(score, "orders", "keys")
    assert keys.recall == 0.0 and keys.precision is None
    assert {i.status for i in keys.items} == {"not recovered"}
    fds = _category(score, "customers", "functional_dependencies")
    assert fds.items[0].note == "the profile states no functional dependencies"
    assert _category(score, "orders", "null_rates").recall == 1.0
    assert score.score < 0.6


def test_oracle_profile_recovers_planted_structure(retail: tuple[Profile, Manifest]) -> None:
    profile, truth = retail
    score = score_structure(_oracle(profile, truth), truth)
    for table in score.tables:
        for name, category in table.categories.items():
            if name != "functional_dependencies":
                assert category.recall == 1.0, (table.name, name)
                assert category.precision in (None, 1.0), (table.name, name)
    assert score.score > 0.85


def test_strings_oracle_recovers_patterns_and_cramers_v(spark: SparkSession) -> None:
    profile, truth = _baseline("strings", spark)
    score = score_structure(_oracle(profile, truth), truth)
    assert _category(score, "contacts", "patterns").recall == 1.0
    assert _category(score, "contacts", "dependencies").recall == 1.0


def test_over_general_regex_is_not_equivalent(retail: tuple[Profile, Manifest]) -> None:
    profile, truth = retail
    data = _oracle(profile, truth).model_dump(mode="json")
    sku = next(c for t in data["tables"] if t["name"] == "products" for c in t["columns"] if c["name"] == "sku")
    sku["string"]["patterns"] = [{"regex": "[A-Z0-9-]+", "coverage": 1.0}]
    patterns = _category(score_structure(Profile.model_validate(data), truth), "products", "patterns")
    assert patterns.recall == 0.0
    assert patterns.precision == 0.0


def test_wrong_range_and_partial_enumeration_get_partial_credit(retail: tuple[Profile, Manifest]) -> None:
    profile, truth = retail
    data = _oracle(profile, truth).model_dump(mode="json")
    customers = next(t for t in data["tables"] if t["name"] == "customers")
    for check in customers["check_constraints"]:
        if check["kind"] == "enumeration" and check["column"] == "segment":
            check["values"] = ["consumer", "enterprise"]
        if check["kind"] == "range" and check["column"] == "signup_date":
            check["min"] = "2010-01-01"
    checks = _category(score_structure(Profile.model_validate(data), truth), "customers", "checks")
    credits = {i.target.split(" ")[1]: i.credit for i in checks.items}
    assert credits["segment"] == pytest.approx(2 / 3)
    assert credits["signup_date"] == 0.0
    assert credits["country"] == 1.0


def test_structure_score_round_trips(retail: tuple[Profile, Manifest]) -> None:
    profile, truth = retail
    score = score_structure(profile, truth)
    assert StructureScore.model_validate_json(score.model_dump_json()) == score
    assert score_structure(profile, truth).model_dump_json() == score.model_dump_json()
