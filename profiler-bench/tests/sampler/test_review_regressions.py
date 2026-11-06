from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from dbldatagen_profiler.model import (
    ComparisonConstraint,
    DependenceMethod,
    Dependency,
    DistinctCount,
    ExpressionConstraint,
    Histogram,
    KeyCandidate,
    LogicalType,
    NumericStats,
    Quantile,
    RangeConstraint,
    Relationship,
    TableProfile,
)
from profiler_bench.sampler import core as sampler_core
from profiler_bench.sampler import sample

from .profiles import column, profile


def test_large_exact_marginals_do_not_round_through_float(spark: SparkSession) -> None:
    low = 9_007_199_254_740_993
    integer = column(
        "big",
        LogicalType.INT64,
        numeric=NumericStats(histogram=Histogram(edges=[str(low), str(low + 5), str(low + 10)], counts=[1, 1])),
    )
    decimal = column(
        "amount",
        LogicalType.DECIMAL,
        precision=20,
        scale=2,
        numeric=NumericStats(
            quantiles=[
                Quantile(q=0, value="900719925474099.01"),
                Quantile(q=1, value="900719925474099.99"),
            ]
        ),
    )
    rows = (
        sample(profile(TableProfile(name="t", row_count=3_000, columns=[integer, decimal])), spark, seed=30)
        .tables["t"]
        .collect()
    )

    integers = {row.big for row in rows}
    decimals = [row.amount for row in rows]
    assert integers <= set(range(low, low + 11))
    assert low + 10 in integers
    assert all(Decimal("900719925474099.01") <= value <= Decimal("900719925474099.99") for value in decimals)
    assert all(value.as_tuple().exponent == -2 for value in decimals)


def test_histogram_clamps_copula_probability_one(spark: SparkSession, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sampler_core, "_normal_cdf", lambda values: np.ones(len(values)))
    stats = NumericStats(histogram=Histogram(edges=[0.0, 1.0, 2.0], counts=[1, 1]))
    table = TableProfile(
        name="t",
        row_count=100,
        columns=[column("x", LogicalType.FLOAT64, numeric=stats), column("y", LogicalType.FLOAT64, numeric=stats)],
        dependencies=[Dependency(columns=("x", "y"), method=DependenceMethod.PEARSON, value=0.5)],
    )

    rows = sample(profile(table), spark, seed=31).tables["t"].collect()

    assert len(rows) == 100
    assert all(1 <= row.x <= 2 and 1 <= row.y <= 2 for row in rows)


def test_fk_coverage_uses_non_null_keys_and_preserves_nulls(spark: SparkSession) -> None:
    parent = TableProfile(
        name="parent",
        row_count=100,
        columns=[column("id", LogicalType.INT64)],
        primary_key=KeyCandidate(columns=["id"], confidence=1),
    )
    child = TableProfile(
        name="child",
        row_count=1_000,
        columns=[column("parent_id", LogicalType.INT64, nullable=True, null_fraction=0.2)],
    )
    relationship = Relationship(
        child_table="child",
        child_columns=["parent_id"],
        parent_table="parent",
        parent_columns=["id"],
        coverage=0.5,
    )
    configured = profile(parent, child).model_copy(update={"relationships": [relationship]})
    result = sample(configured, spark, seed=32)
    parents = result.tables["parent"]
    children = result.tables["child"]

    assert children.filter("parent_id IS NULL").count() == 200
    assert (
        children.filter("parent_id IS NOT NULL").join(parents, children.parent_id == parents.id, "inner").count() == 400
    )


def test_expression_selection_is_row_id_deterministic(spark: SparkSession) -> None:
    table = TableProfile(
        name="t",
        row_count=500,
        columns=[column("x", LogicalType.FLOAT64, numeric=NumericStats(min=0, max=1))],
        check_constraints=[ExpressionConstraint(expression="x >= 0.5", satisfied_fraction=0.5)],
    )

    spark.conf.set("spark.sql.shuffle.partitions", "2")
    first = sample(profile(table), spark, seed=33).tables["t"].collect()
    spark.conf.set("spark.sql.shuffle.partitions", "7")
    second = sample(profile(table), spark, seed=33).tables["t"].collect()

    assert first == second
    assert all(row.x >= 0.5 for row in first)


def test_cardinality_pool_preserves_copula_rank_dependence(spark: SparkSession) -> None:
    stats = NumericStats(min=-4, max=4, mean=0, stddev=1)
    table = TableProfile(
        name="t",
        row_count=5_000,
        columns=[
            column("x", LogicalType.FLOAT64, numeric=stats, distinct_count=DistinctCount(value=10, exact=True)),
            column("y", LogicalType.FLOAT64, numeric=stats, distinct_count=DistinctCount(value=10, exact=True)),
        ],
        dependencies=[Dependency(columns=("x", "y"), method=DependenceMethod.SPEARMAN, value=0.8)],
    )
    rows = sample(profile(table), spark, seed=34).tables["t"].collect()
    x = pd.Series([row.x for row in rows]).rank().to_numpy()
    y = pd.Series([row.y for row in rows]).rank().to_numpy()

    assert len(set(x)) == 10
    assert len(set(y)) == 10
    assert float(np.corrcoef(x, y)[0, 1]) == pytest.approx(0.8, abs=0.06)


@pytest.mark.parametrize("operator", ["<", ">"])
def test_strict_comparisons_repair_to_strict_inequality(spark: SparkSession, operator: str) -> None:
    table = TableProfile(
        name="t",
        row_count=500,
        columns=[
            column("a", LogicalType.INT32, numeric=NumericStats(min=10, max=10)),
            column("b", LogicalType.INT32, numeric=NumericStats(min=10, max=10)),
        ],
        check_constraints=[ComparisonConstraint(left="a", operator=operator, right="b", satisfied_fraction=1)],
    )
    frame = sample(profile(table), spark, seed=35).tables["t"]

    assert frame.filter(f"NOT (a {operator} b)").count() == 0


def test_constraints_then_keys_and_fk_preserve_all_invariants(spark: SparkSession) -> None:
    parent = TableProfile(
        name="parent",
        row_count=100,
        columns=[column("id", LogicalType.INT64, numeric=NumericStats(min="1", max="20"))],
        primary_key=KeyCandidate(columns=["id"], confidence=1),
        check_constraints=[RangeConstraint(column="id", min="1", max="100000", satisfied_fraction=1)],
    )
    child = TableProfile(
        name="child",
        row_count=1_000,
        columns=[
            column("parent_id", LogicalType.INT64),
            column("line", LogicalType.INT32, numeric=NumericStats(min=1, max=5)),
        ],
        primary_key=KeyCandidate(columns=["parent_id", "line"], confidence=1),
        check_constraints=[RangeConstraint(column="line", min=1, max=100_000, satisfied_fraction=1)],
    )
    relationship = Relationship(
        child_table="child",
        child_columns=["parent_id"],
        parent_table="parent",
        parent_columns=["id"],
        coverage=1,
    )
    configured = profile(parent, child).model_copy(update={"relationships": [relationship]})
    result = sample(configured, spark, seed=36)
    parents = result.tables["parent"]
    children = result.tables["child"]

    assert parents.select("id").distinct().count() == 100
    assert parents.filter("id < 1 OR id > 100000").count() == 0
    assert children.select("parent_id", "line").distinct().count() == 1_000
    assert children.filter("line < 1 OR line > 100000").count() == 0
    assert children.join(parents, children.parent_id == parents.id, "left_anti").count() == 0
