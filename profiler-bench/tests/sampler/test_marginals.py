from __future__ import annotations

import math
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import types as T

from dbldatagen_profiler.model import (
    DistinctCount,
    Histogram,
    LengthStats,
    LogicalType,
    NumericStats,
    Quantile,
    StringStats,
    TableProfile,
    TemporalStats,
    ValueFrequency,
)
from profiler_bench.sampler import sample

from .profiles import column, profile


@pytest.mark.parametrize(
    ("name", "stats", "expected_mean", "mean_tolerance", "expected_std", "std_tolerance"),
    [
        (
            "histogram",
            NumericStats(histogram=Histogram(edges=[0.0, 10.0, 20.0], counts=[1, 3])),
            12.5,
            0.35,
            None,
            None,
        ),
        (
            "quantiles",
            NumericStats(quantiles=[Quantile(q=0, value=0.0), Quantile(q=0.5, value=10.0), Quantile(q=1, value=20.0)]),
            10.0,
            0.25,
            None,
            None,
        ),
        (
            "clipped_normal",
            NumericStats(min=-4.0, max=4.0, mean=0.5, stddev=0.8),
            0.5,
            0.05,
            0.8,
            0.05,
        ),
        ("uniform", NumericStats(min=2.0, max=8.0), 5.0, 0.08, math.sqrt(3), 0.08),
    ],
)
def test_numeric_marginal_paths_recover_statistics(
    spark: SparkSession,
    name: str,
    stats: NumericStats,
    expected_mean: float,
    mean_tolerance: float,
    expected_std: float | None,
    std_tolerance: float | None,
) -> None:
    result = sample(
        profile(TableProfile(name="t", row_count=5_000, columns=[column("x", LogicalType.FLOAT64, numeric=stats)])),
        spark,
        seed=12,
    )
    values = np.asarray([row.x for row in result.tables["t"].collect()])

    assert float(values.mean()) == pytest.approx(expected_mean, abs=mean_tolerance)
    if expected_std is not None and std_tolerance is not None:
        assert float(values.std(ddof=1)) == pytest.approx(expected_std, abs=std_tolerance)
    if name == "histogram":
        assert float(np.mean(values >= 10)) == pytest.approx(0.75, abs=0.03)
    if name == "quantiles":
        assert np.quantile(values, [0.25, 0.5, 0.75]).tolist() == pytest.approx([5, 10, 15], abs=0.35)
    fields = result.report.columns["t.x"].used + result.report.columns["t.x"].approximated
    assert any(name in field for field in fields)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (LogicalType.INT32, 0),
        (LogicalType.FLOAT64, 0.0),
        (LogicalType.DATE, date(1970, 1, 1)),
        (LogicalType.TIMESTAMP_NTZ, datetime(1970, 1, 1)),
    ],
)
def test_type_defaults_are_valid(spark: SparkSession, kind: LogicalType, expected: object) -> None:
    frame = sample(profile(TableProfile(name="t", row_count=20, columns=[column("x", kind)])), spark, seed=2).tables[
        "t"
    ]

    values = {row.x for row in frame.collect()}

    assert values == {expected}


def test_boolean_default_is_balanced(spark: SparkSession) -> None:
    values = (
        sample(
            profile(TableProfile(name="t", row_count=2_000, columns=[column("x", LogicalType.BOOLEAN)])), spark, seed=2
        )
        .tables["t"]
        .collect()
    )

    assert sum(row.x for row in values) / len(values) == pytest.approx(0.5, abs=0.04)


def test_integer_histogram_rounds_and_stays_in_range(spark: SparkSession) -> None:
    stats = NumericStats(histogram=Histogram(edges=[1, 5, 10], counts=[1, 1]))
    frame = sample(
        profile(TableProfile(name="t", row_count=2_000, columns=[column("x", LogicalType.INT32, numeric=stats)])),
        spark,
        seed=3,
    ).tables["t"]

    values = [row.x for row in frame.collect()]

    assert all(isinstance(value, int) and 1 <= value <= 10 for value in values)
    assert 10 in values
    assert np.mean(values) == pytest.approx(5.0, abs=0.2)


def test_decimal_preserves_precision_scale_and_distribution(spark: SparkSession) -> None:
    stats = NumericStats(min="10.00", max="20.00")
    frame = sample(
        profile(
            TableProfile(
                name="t",
                row_count=1_000,
                columns=[column("amount", LogicalType.DECIMAL, precision=8, scale=2, numeric=stats)],
            )
        ),
        spark,
        seed=4,
    ).tables["t"]
    values = [row.amount for row in frame.collect()]

    assert frame.schema["amount"].dataType == T.DecimalType(8, 2)
    assert all(isinstance(value, Decimal) and value.as_tuple().exponent == -2 for value in values)
    assert float(np.mean([float(value) for value in values])) == pytest.approx(15.0, abs=0.2)


def test_decimal_histogram_uses_half_open_bins_and_closed_last_bin(spark: SparkSession) -> None:
    stats = NumericStats(histogram=Histogram(edges=["10.00", "10.02", "10.04"], counts=[1, 1]))
    rows = (
        sample(
            profile(
                TableProfile(
                    name="t",
                    row_count=2_000,
                    columns=[column("amount", LogicalType.DECIMAL, precision=6, scale=2, numeric=stats)],
                )
            ),
            spark,
            seed=41,
        )
        .tables["t"]
        .collect()
    )
    values = {row.amount for row in rows}

    assert Decimal("10.02") in values
    assert Decimal("10.04") in values
    assert values <= {Decimal("10.00"), Decimal("10.01"), Decimal("10.02"), Decimal("10.03"), Decimal("10.04")}


@pytest.mark.parametrize(
    ("kind", "stats", "expected"),
    [
        (
            LogicalType.DATE,
            TemporalStats(quantiles=[Quantile(q=0, value="2020-01-01"), Quantile(q=1, value="2020-01-11")]),
            date(2020, 1, 6),
        ),
        (
            LogicalType.TIMESTAMP_NTZ,
            TemporalStats(min="2020-01-01T00:00:00", max="2020-01-03T00:00:00"),
            datetime(2020, 1, 2),
        ),
    ],
)
def test_temporal_marginals_recover_midpoint(
    spark: SparkSession, kind: LogicalType, stats: TemporalStats, expected: date | datetime
) -> None:
    frame = sample(
        profile(TableProfile(name="t", row_count=3_000, columns=[column("x", kind, temporal=stats)])),
        spark,
        seed=5,
    ).tables["t"]
    values = [row.x for row in frame.collect()]
    midpoint = sorted(values)[len(values) // 2]
    tolerance = 1 if kind == LogicalType.DATE else 7_200

    if kind == LogicalType.DATE:
        assert abs((midpoint - expected).days) <= tolerance
    else:
        assert abs((midpoint - expected).total_seconds()) <= tolerance


def test_top_values_other_mass_and_distinct_count(spark: SparkSession) -> None:
    stats = StringStats(
        top_values=[ValueFrequency(value="red", frequency=0.3), ValueFrequency(value="blue", frequency=0.2)],
        length=LengthStats(min=8, max=8),
    )
    frame = sample(
        profile(
            TableProfile(
                name="t",
                row_count=5_000,
                columns=[
                    column(
                        "color",
                        LogicalType.STRING,
                        string=stats,
                        nullable=True,
                        null_fraction=0.2,
                        distinct_count=DistinctCount(value=12, exact=True),
                    )
                ],
            )
        ),
        spark,
        seed=6,
    ).tables["t"]
    counts = {row.color: row["count"] for row in frame.groupBy("color").count().collect()}
    nulls = counts.pop(None)
    total = sum(counts.values())
    observed = [counts["red"] / total, counts["blue"] / total, 1 - (counts["red"] + counts["blue"]) / total]
    tvd = 0.5 * sum(abs(actual - target) for actual, target in zip(observed, [0.3, 0.2, 0.5], strict=True))

    assert tvd <= 0.02
    assert nulls == 1_000
    assert counts["red"] == 1_200
    assert counts["blue"] == 800
    assert len(counts) == pytest.approx(12, abs=1)


def test_non_finite_values_are_reproduced(spark: SparkSession) -> None:
    stats = NumericStats(min=float("-inf"), max=float("inf"), mean=0.0, stddev=1.0)
    values = (
        sample(
            profile(TableProfile(name="t", row_count=1_000, columns=[column("x", LogicalType.FLOAT64, numeric=stats)])),
            spark,
            seed=7,
        )
        .tables["t"]
        .collect()
    )
    numbers = [row.x for row in values]

    assert float("-inf") in numbers
    assert float("inf") in numbers
