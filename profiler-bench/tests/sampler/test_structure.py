from __future__ import annotations

import math

import numpy as np
import pytest
from pyspark.sql import SparkSession

from dbldatagen_profiler.model import (
    ComparisonConstraint,
    DependenceMethod,
    Dependency,
    DistinctCount,
    EnumerationConstraint,
    ExpressionConstraint,
    FanOutStats,
    KeyCandidate,
    LogicalType,
    NumericStats,
    Quantile,
    RangeConstraint,
    Relationship,
    StringStats,
    TableProfile,
    ValueFrequency,
)
from profiler_bench.sampler import sample

from .profiles import column, profile


def test_composite_primary_and_unique_keys_are_unique(spark: SparkSession) -> None:
    table = TableProfile(
        name="t",
        row_count=1_000,
        columns=[
            column("tenant", LogicalType.INT32, distinct_count=DistinctCount(value=5, exact=True)),
            column("id", LogicalType.INT64, distinct_count=DistinctCount(value=50, exact=True)),
            column("region", LogicalType.STRING, distinct_count=DistinctCount(value=3, exact=True)),
            column("code", LogicalType.STRING, distinct_count=DistinctCount(value=30, exact=True)),
        ],
        primary_key=KeyCandidate(columns=["tenant", "id"], confidence=1),
        unique_keys=[KeyCandidate(columns=["region", "code"], confidence=1)],
    )
    frame = sample(profile(table), spark, seed=20).tables["t"]

    assert frame.select("tenant", "id").distinct().count() == 1_000
    assert frame.select("region", "code").distinct().count() == 1_000


def test_fk_integrity_partial_coverage_and_fanout(spark: SparkSession) -> None:
    parent = TableProfile(
        name="parent",
        row_count=100,
        columns=[column("id", LogicalType.INT64)],
        primary_key=KeyCandidate(columns=["id"], confidence=1),
    )
    child = TableProfile(name="child", row_count=1_000, columns=[column("parent_id", LogicalType.INT64)])
    relationship = Relationship(
        child_table="child",
        child_columns=["parent_id"],
        parent_table="parent",
        parent_columns=["id"],
        coverage=0.8,
        fan_out=FanOutStats(
            min=0,
            max=20,
            mean=8,
            stddev=2,
            quantiles=[Quantile(q=0.5, value=8)],
        ),
    )
    configured = profile(parent, child).model_copy(update={"relationships": [relationship]})
    result = sample(configured, spark, seed=21)
    parents = result.tables["parent"]
    children = result.tables["child"]
    matched = children.join(parents, children.parent_id == parents.id, "inner")
    fanout = matched.groupBy("id").count()
    counts = [row["count"] for row in parents.join(fanout, "id", "left").fillna(0, ["count"]).collect()]

    assert matched.count() / children.count() == pytest.approx(0.8, abs=0.001)
    assert children.join(parents, children.parent_id == parents.id, "left_anti").count() == 200
    assert float(np.mean(counts)) == pytest.approx(8, abs=0.01)
    assert float(np.median(counts)) == pytest.approx(8, abs=2)


def test_range_enumeration_and_comparison_constraints_hold(spark: SparkSession) -> None:
    table = TableProfile(
        name="t",
        row_count=2_000,
        columns=[
            column("a", LogicalType.INT32, numeric=NumericStats(min=-100, max=100)),
            column("b", LogicalType.INT32, numeric=NumericStats(min=-100, max=100)),
            column("kind", LogicalType.STRING),
        ],
        check_constraints=[
            RangeConstraint(column="a", min=10, max=20, satisfied_fraction=1),
            EnumerationConstraint(column="kind", values=["x", "y", "z"], min_value_count=10, satisfied_fraction=1),
            ComparisonConstraint(left="a", operator="<=", right="b", satisfied_fraction=1),
        ],
    )
    frame = sample(profile(table), spark, seed=22).tables["t"]

    assert frame.filter("a < 10 OR a > 20").count() == 0
    assert frame.filter("kind NOT IN ('x', 'y', 'z')").count() == 0
    assert frame.filter("a > b").count() == 0


def test_infeasible_expression_is_reported_not_raised(spark: SparkSession) -> None:
    table = TableProfile(
        name="t",
        row_count=100,
        columns=[column("x", LogicalType.INT32, numeric=NumericStats(min=1, max=2))],
        check_constraints=[ExpressionConstraint(expression="x < 0", satisfied_fraction=0)],
    )

    result = sample(profile(table), spark, seed=23)

    assert result.tables["t"].count() == 100
    assert result.report.unmet_constraints == {"t": ("x < 0",)}
    assert "check_constraints.expression" in result.report.columns["t.x"].unmet


@pytest.mark.parametrize(
    ("method", "target", "tolerance"),
    [(DependenceMethod.PEARSON, 0.65, 0.04), (DependenceMethod.SPEARMAN, -0.55, 0.04)],
)
def test_numeric_copula_recovers_target(
    spark: SparkSession, method: DependenceMethod, target: float, tolerance: float
) -> None:
    stats = NumericStats(min=-5, max=5, mean=0, stddev=1)
    table = TableProfile(
        name="t",
        row_count=5_000,
        columns=[column("x", LogicalType.FLOAT64, numeric=stats), column("y", LogicalType.FLOAT64, numeric=stats)],
        dependencies=[Dependency(columns=("x", "y"), method=method, value=target)],
    )
    values = sample(profile(table), spark, seed=24).tables["t"].collect()
    x = np.asarray([row.x for row in values])
    y = np.asarray([row.y for row in values])
    if method == DependenceMethod.SPEARMAN:
        x = np.argsort(np.argsort(x))
        y = np.argsort(np.argsort(y))

    assert float(np.corrcoef(x, y)[0, 1]) == pytest.approx(target, abs=tolerance)


def test_inconsistent_correlation_matrix_is_repaired_and_reported(spark: SparkSession) -> None:
    stats = NumericStats(min=-5, max=5, mean=0, stddev=1)
    table = TableProfile(
        name="t",
        row_count=1_000,
        columns=[column(name, LogicalType.FLOAT64, numeric=stats) for name in ("x", "y", "z")],
        dependencies=[
            Dependency(columns=("x", "y"), method=DependenceMethod.PEARSON, value=0.9),
            Dependency(columns=("x", "z"), method=DependenceMethod.PEARSON, value=0.9),
            Dependency(columns=("y", "z"), method=DependenceMethod.PEARSON, value=-0.9),
        ],
    )

    result = sample(profile(table), spark, seed=25)

    assert result.tables["t"].count() == 1_000
    assert all("dependencies.nearest_psd" in result.report.columns[f"t.{name}"].approximated for name in "xyz")


def _cramers_v(left: list[str], right: list[str]) -> float:
    left_values = sorted(set(left))
    right_values = sorted(set(right))
    observed = np.asarray(
        [[sum(a == x and b == y for a, b in zip(left, right, strict=True)) for y in right_values] for x in left_values],
        dtype=float,
    )
    expected = observed.sum(axis=1, keepdims=True) @ observed.sum(axis=0, keepdims=True) / observed.sum()
    return math.sqrt(float((((observed - expected) ** 2 / expected).sum()) / observed.sum()))


def test_cramers_v_target_is_approximated(spark: SparkSession) -> None:
    categorical = StringStats(
        top_values=[ValueFrequency(value="a", frequency=0.5), ValueFrequency(value="b", frequency=0.5)]
    )
    table = TableProfile(
        name="t",
        row_count=6_000,
        columns=[
            column("left", LogicalType.STRING, string=categorical),
            column("right", LogicalType.STRING, string=categorical),
        ],
        dependencies=[Dependency(columns=("left", "right"), method=DependenceMethod.CRAMERS_V, value=0.6)],
    )
    rows = sample(profile(table), spark, seed=26).tables["t"].collect()

    assert _cramers_v([row.left for row in rows], [row.right for row in rows]) == pytest.approx(0.6, abs=0.06)


def test_report_classifies_used_ignored_and_unmet_fields(spark: SparkSession) -> None:
    stats = NumericStats(
        min=-10,
        max=10,
        mean=0,
        stddev=2,
        quantiles=[Quantile(q=0.5, value=0)],
    )
    table = TableProfile(
        name="t",
        row_count=200,
        columns=[column("x", LogicalType.FLOAT64, numeric=stats), column("kind", LogicalType.STRING)],
        dependencies=[Dependency(columns=("x", "kind"), method=DependenceMethod.CORRELATION_RATIO, value=0.2)],
        check_constraints=[ExpressionConstraint(expression="x > 100", satisfied_fraction=0)],
    )

    report = sample(profile(table), spark, seed=27).report

    assert "numeric.quantiles" in report.columns["t.x"].used
    assert {"numeric.min", "numeric.max", "dependencies.correlation_ratio"} <= set(report.columns["t.x"].ignored)
    assert "dependencies.correlation_ratio" in report.columns["t.kind"].ignored
    assert "check_constraints.expression" in report.columns["t.x"].unmet
    assert report.unmet_constraints == {"t": ("x > 100",)}
