from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from dbldatagen_profiler.engines.spark import profile_tables
from dbldatagen_profiler.model import ColumnProfile, EngineInfo, LogicalType, Profile, TableProfile, load_profile
from profiler_bench.catalog import load
from profiler_bench.sampler import sample
from profiler_bench.sampler.regex import generate_regex


EXAMPLES = Path(__file__).parents[3] / "profiler" / "spec" / "examples"


def test_shipped_profiles_preserve_rows_schema_keys_and_relationships(spark: SparkSession) -> None:
    profile = load_profile((EXAMPLES / "two_tables_fk.json").read_text())
    result = sample(profile, spark, seed=17, scale=0.1)

    assert {name: frame.count() for name, frame in result.tables.items()} == {"customers": 100, "orders": 500}
    customers = result.tables["customers"]
    orders = result.tables["orders"]
    assert customers.schema == T.StructType(
        [
            T.StructField("customer_id", T.LongType(), False),
            T.StructField(
                "address",
                T.StructType([T.StructField("city", T.StringType(), True), T.StructField("zip", T.StringType(), True)]),
                True,
            ),
        ]
    )
    assert customers.select("customer_id").distinct().count() == 100
    assert orders.join(customers, "customer_id", "left_anti").count() == 0
    assert orders.filter("shipped_on < to_date(ordered_at)").count() == 0


def test_same_seed_is_deterministic_and_partition_independent(spark: SparkSession) -> None:
    profile = load_profile((EXAMPLES / "single_table.json").read_text())
    first = sample(profile, spark, seed=4, scale=0.02).tables["customers"]
    second = sample(profile, spark, seed=4, scale=0.02).tables["customers"]

    assert first.orderBy("customer_id").collect() == second.repartition(7).orderBy("customer_id").collect()


def test_marginals_nulls_cardinality_and_report(spark: SparkSession) -> None:
    profile = load_profile((EXAMPLES / "single_table.json").read_text())
    result = sample(profile, spark, seed=9, scale=0.1)
    frame = result.tables["customers"]
    stats = frame.agg(
        F.avg("balance").alias("mean"),
        F.sum(F.col("email").isNull().cast("int")).alias("nulls"),
        F.countDistinct("tier").alias("tiers"),
    ).first()

    assert stats is not None
    assert -1000 <= stats.mean <= 50000
    assert stats.nulls / 1000 == pytest.approx(profile.tables[0].columns[1].null_fraction, abs=0.015)
    assert stats.tiers <= 4
    balance = result.report.columns["customers.balance"]
    assert "numeric.histogram" in balance.used
    assert "numeric.quantiles" in balance.ignored
    assert "check_constraints.expression" in balance.used


def test_supported_regex_subset_fullmatches() -> None:
    patterns = [
        r"[A-Z]{3}-\d{2,4}",
        r"(cat|dog)s?",
        r"\w+\s[0-9]*",
        r"ab+c.*",
        r"[^0-9]{2}",
    ]
    rng = np.random.default_rng(10)

    for pattern in patterns:
        assert all(re.fullmatch(pattern, generate_regex(pattern, rng, max_length=24)) for _ in range(50))


def test_profiler_to_sampler_catalog_smoke(spark: SparkSession) -> None:
    source = load("numeric", "xs", seed=3, spark=spark)
    profile = profile_tables(source)
    reconstructed = sample(profile, spark, seed=3)

    assert reconstructed.tables.keys() == source.keys()
    for name, source_frame in source.items():
        assert reconstructed.tables[name].schema == source_frame.schema
        assert reconstructed.tables[name].count() == source_frame.count()


def test_every_logical_type_constructs(spark: SparkSession) -> None:
    def column(name: str, kind: LogicalType, **values) -> ColumnProfile:
        return ColumnProfile(
            name=name,
            logical_type=kind,
            source_type=values.pop("source_type", kind.value),
            nullable=kind == LogicalType.NULL,
            null_fraction=1.0 if kind == LogicalType.NULL else 0.0,
            **values,
        )

    primitives = [
        column(kind.value, kind)
        for kind in LogicalType
        if kind not in {LogicalType.DECIMAL, LogicalType.STRUCT, LogicalType.ARRAY, LogicalType.MAP}
    ]
    columns = [
        *primitives,
        column("decimal", LogicalType.DECIMAL, precision=12, scale=2),
        column("struct", LogicalType.STRUCT, children=[column("x", LogicalType.INT32)]),
        column("array", LogicalType.ARRAY, children=[column("element", LogicalType.STRING)]),
        column(
            "map",
            LogicalType.MAP,
            children=[column("key", LogicalType.STRING), column("value", LogicalType.INT64)],
        ),
    ]
    profile = Profile(
        spec_version="0.1.0",
        created_at=datetime.now(timezone.utc),
        engine=EngineInfo(name="test", version="1"),
        tables=[TableProfile(name="types", row_count=3, columns=columns)],
    )

    frame = sample(profile, spark, seed=1).tables["types"]

    assert frame.count() == 3
    assert frame.schema.names == [item.name for item in columns]
