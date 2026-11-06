from collections.abc import Callable
from typing import Any

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from profiler_bench.catalog import DATASETS, Manifest, load, materialize, verify


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_manifest_is_true_of_its_data(name: str, spark: SparkSession) -> None:
    truth = materialize(name, "xs", 0).manifest
    assert verify(truth, load(name, "xs", 0, spark)) == []


def test_load_reads_both_timestamp_kinds(spark: SparkSession) -> None:
    events = load("temporal", "xs", 0, spark)["events"]
    types = {f.name: f.dataType.simpleString() for f in events.schema.fields}
    assert types["event_ts"] == "timestamp"
    assert types["local_ts"] == "timestamp_ntz"
    assert types["event_date"] == "date"


def _edit(truth: Manifest, table: str, column: str, **update: object) -> Manifest:
    data = truth.model_dump()
    (target,) = [t for t in data["tables"] if t["name"] == table]
    (entry,) = [c for c in target["columns"] if c["name"] == column]
    entry.update(update)
    return Manifest.model_validate(data)


def _edit_table(truth: Manifest, table: str, key: str, edit: Callable[[Any], Any]) -> Manifest:
    data = truth.model_dump()
    (target,) = [t for t in data["tables"] if t["name"] == table]
    target[key] = edit(target[key])
    return Manifest.model_validate(data)


def _rho(dependencies: list[dict[str, Any]], value: float) -> list[dict[str, Any]]:
    return [{**d, "value": value} if d["measure"] == "pearson" and d["value"] > 0.5 else d for d in dependencies]


@pytest.mark.parametrize(
    ("name", "tamper", "expected"),
    [
        ("numeric", lambda m: _edit(m, "measurements", "normal", null_rate=0.2), "normal: null rate"),
        (
            "numeric",
            lambda m: _edit(
                m, "measurements", "normal", distribution={"family": "normal", "params": {"mean": 49.0, "std": 10.0}}
            ),
            "normal: mean",
        ),
        ("numeric", lambda m: _edit_table(m, "measurements", "dependencies", lambda d: _rho(d, 0.3)), "pearson"),
        (
            "strings",
            lambda m: _edit(
                m,
                "contacts",
                "postal_code",
                formats=[{"regex": "[0-9]{5}", "share": 0.6}, {"regex": "[0-9]{5}-[0-9]{4}", "share": 0.4}],
            ),
            "format '[0-9]{5}' share",
        ),
        ("strings", lambda m: _edit(m, "contacts", "city", unique=True), "duplicate values"),
        ("edge", lambda m: _edit(m, "edge_cases", "int64_extremes", max_value=2**62), "max is not"),
        ("edge", lambda m: _edit(m, "edge_cases", "special_floats", special_values={"nan": 0.1}), "nan share"),
        (
            "nested",
            lambda m: _edit(m, "profiles", "tags", length={"family": "poisson", "params": {"lam": 3.0}}),
            "tags: length",
        ),
    ],
)
def test_verify_detects_false_claims(
    name: str, tamper: Callable[[Manifest], Manifest], expected: str, spark: SparkSession
) -> None:
    failures = verify(tamper(materialize(name, "xs", 0).manifest), load(name, "xs", 0, spark))
    assert any(expected in f for f in failures), failures


@pytest.mark.parametrize(
    ("table", "tamper", "expected"),
    [
        ("customers", lambda df: df.where("customer_id > 10"), "coverage"),
        (
            "order_items",
            lambda df: df.withColumn("line_total", (F.col("line_total") + 1).cast("decimal(12,2)")),
            "line_total = unit_price",
        ),
        ("orders", lambda df: df.withColumn("ship_date", F.date_sub("order_date", 1)), "order_date` <= `ship_date"),
        ("products", lambda df: df.union(df.limit(1)), "primary key"),
    ],
)
def test_verify_detects_broken_data(
    table: str, tamper: Callable[[DataFrame], DataFrame], expected: str, spark: SparkSession
) -> None:
    frames = load("retail", "xs", 0, spark)
    frames[table] = tamper(frames[table])
    failures = verify(materialize("retail", "xs", 0).manifest, frames)
    assert any(expected in f for f in failures), failures


def test_verify_detects_broken_order(spark: SparkSession) -> None:
    frames = load("temporal", "xs", 0, spark)
    frames["events"] = frames["events"].orderBy(F.desc("event_id")).coalesce(1)
    failures = verify(materialize("temporal", "xs", 0).manifest, frames)
    assert any("event_id: values do not strictly increase" in f for f in failures), failures


def test_verify_requires_a_utc_session(spark: SparkSession) -> None:
    frames = load("numeric", "xs", 0, spark)
    spark.conf.set("spark.sql.session.timeZone", "America/New_York")
    try:
        with pytest.raises(ValueError, match="UTC"):
            verify(materialize("numeric", "xs", 0).manifest, frames)
    finally:
        spark.conf.set("spark.sql.session.timeZone", "UTC")
