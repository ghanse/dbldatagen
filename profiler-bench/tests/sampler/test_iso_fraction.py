"""Regression: odd-length fractional timestamps must round-trip on Python 3.10."""

from __future__ import annotations

from datetime import datetime

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from dbldatagen_profiler.engines.spark import profile_dataframe
from dbldatagen_profiler.model import LogicalType, TableProfile, TemporalStats
from profiler_bench.sampler import sample

from .profiles import column, profile


@pytest.mark.parametrize(
    ("kind", "minimum", "maximum"),
    [
        (LogicalType.TIMESTAMP, "2024-02-04T16:27:43.7Z", "2024-02-04T16:27:43.791440Z"),
        (LogicalType.TIMESTAMP, "2024-02-04T16:27:43.791Z", "2024-02-04T16:27:43.79144Z"),
        (LogicalType.TIMESTAMP, "2024-02-04T16:27:43Z", "2024-02-04T16:27:43.791440Z"),
        (LogicalType.TIMESTAMP_NTZ, "2024-02-04T16:27:43.7", "2024-02-04T16:27:43.791440"),
        (LogicalType.TIMESTAMP_NTZ, "2024-02-04T16:27:43.79144", "2024-02-04T16:27:43.791440"),
        (LogicalType.TIMESTAMP_NTZ, "2024-02-04T16:27:43", "2024-02-04T16:27:43.79144"),
    ],
)
def test_sampler_accepts_odd_fraction_lengths(
    spark: SparkSession, kind: LogicalType, minimum: str, maximum: str
) -> None:
    """Odd fractional lengths must parse on Python 3.10 (the original fidelity crash)."""
    stats = TemporalStats(min=minimum, max=maximum)
    result = sample(
        profile(TableProfile(name="t", row_count=20, columns=[column("moment", kind, temporal=stats)])),
        spark,
        seed=7,
    )
    values = [row.moment for row in result.tables["t"].collect()]
    assert len(values) == 20
    assert all(isinstance(value, datetime) for value in values)


def test_profile_sample_round_trip_spark_fraction(spark: SparkSession) -> None:
    """Profiles a value Spark would cast as '.79144' and samples from that profile."""
    source = spark.sql(
        """
        SELECT TIMESTAMP '2024-02-04 16:27:43.79144' AS ts,
               TIMESTAMP_NTZ '2024-02-04 16:27:43.79144' AS ntz
        """
    )
    built = profile_dataframe("fractions", source)
    ts = built.tables[0].columns[0]
    ntz = built.tables[0].columns[1]
    assert ts.temporal is not None and ntz.temporal is not None
    assert ts.temporal.min == "2024-02-04T16:27:43.791440Z"
    assert ntz.temporal.min == "2024-02-04T16:27:43.791440"

    reconstructed = sample(built, spark, seed=11).tables["fractions"]
    rendered = reconstructed.select(
        F.date_format("ts", "yyyy-MM-dd'T'HH:mm:ss.SSSSSS").alias("ts"),
        F.col("ntz"),
    ).collect()
    assert len(rendered) == 1
    assert rendered[0].ts == "2024-02-04T16:27:43.791440"
    assert rendered[0].ntz == datetime(2024, 2, 4, 16, 27, 43, 791440)
