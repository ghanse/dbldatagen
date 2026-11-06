import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from profiler_bench.catalog import DATASETS, Manifest, load, materialize, verify


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_manifest_is_true_of_its_data(name: str, spark: SparkSession) -> None:
    truth = materialize(name, "xs", 0).manifest
    assert verify(truth, load(name, "xs", 0, spark)) == []


@pytest.mark.parametrize("sample_seed", [1, 2])
@pytest.mark.parametrize("name", sorted(DATASETS))
def test_manifest_is_true_for_other_sample_seeds(name: str, sample_seed: int, spark: SparkSession) -> None:
    truth = materialize(name, "xs", 0, sample_seed=sample_seed).manifest
    assert truth == materialize(name, "xs", 0).manifest
    assert verify(truth, load(name, "xs", 0, spark, sample_seed=sample_seed)) == []


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


def _dist(family: str, **params: object) -> dict[str, Any]:
    return {"family": family, "params": params}


def _strength(truth: Manifest, table: str, measure: str, columns: tuple[str, str], value: float) -> Manifest:
    data = truth.model_dump()
    (target,) = [t for t in data["tables"] if t["name"] == table]
    (entry,) = [d for d in target["dependencies"] if d["measure"] == measure and tuple(d["columns"]) == columns]
    entry["value"] = value
    return Manifest.model_validate(data)


def _unique_fraction(truth: Manifest, value: float) -> Manifest:
    data = truth.model_dump()
    data["tables"][0]["quasi_identifiers"][0]["unique_fraction"] = value
    return Manifest.model_validate(data)


LOGNORMAL_MEAN = math.exp(3 + 0.75**2 / 2)
LOGNORMAL_STD = math.sqrt((math.exp(0.75**2) - 1) * math.exp(6 + 0.75**2))


@pytest.mark.parametrize(
    ("name", "tamper", "expected"),
    [
        ("numeric", lambda m: _edit(m, "measurements", "normal", null_rate=0.2), "normal: null rate"),
        (
            "numeric",
            lambda m: _edit(m, "measurements", "normal", distribution=_dist("normal", mean=49.0, std=10.0)),
            "normal: mean",
        ),
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
        ("nested", lambda m: _edit(m, "profiles", "tags", length=_dist("poisson", lam=3.0)), "tags: length"),
    ],
)
def test_verify_detects_false_claims(
    name: str, tamper: Callable[[Manifest], Manifest], expected: str, spark: SparkSession
) -> None:
    truth = tamper(materialize(name, "xs", 0).manifest)
    failures = verify(truth, load(name, "xs", 0, spark), tables=[t.name for t in truth.tables if t.name != "wide"])
    assert any(expected in f for f in failures), failures


@pytest.mark.parametrize(
    ("name", "tamper", "expected"),
    [
        (
            "numeric",
            lambda m: _edit(m, "measurements", "ratio", distribution=_dist("normal", mean=0.5, std=math.sqrt(1 / 12))),
            "ratio: KS",
        ),
        (
            "numeric",
            lambda m: _edit(m, "measurements", "bimodal", distribution=_dist("normal", mean=0.0, std=math.sqrt(6.55))),
            "bimodal: KS",
        ),
        (
            "numeric",
            lambda m: _edit(
                m, "measurements", "lognormal", distribution=_dist("normal", mean=LOGNORMAL_MEAN, std=LOGNORMAL_STD)
            ),
            "lognormal: KS",
        ),
        (
            "nested",
            lambda m: _edit(m, "profiles", "scores", length=_dist("negative_binomial", n=15.0, p=2.5 / (35 / 12))),
            "scores: length goodness of fit chi-square",
        ),
        (
            "nested",
            lambda m: _edit(m, "profiles", "tags", length=_dist("binomial", n=20, p=0.1)),
            "tags: length goodness of fit chi-square",
        ),
        (
            "strings",
            lambda m: _edit(m, "contacts", "city", distribution=_dist("zipf", s=1.2, n_values=5000)),
            "city: rank 1",
        ),
    ],
    ids=[
        "uniform-as-normal",
        "mixture-as-normal",
        "lognormal-as-normal",
        "uniform-as-negbin",
        "poisson-as-binomial",
        "zipf-s",
    ],
)
def test_verify_rejects_wrong_family_with_matched_moments(
    name: str, tamper: Callable[[Manifest], Manifest], expected: str, spark: SparkSession
) -> None:
    table = {"numeric": "measurements", "nested": "profiles", "strings": "contacts"}[name]
    failures = verify(tamper(materialize(name, "xs", 0).manifest), load(name, "xs", 0, spark), tables=[table])
    assert any(expected in f for f in failures), failures
    assert not any("mean" in f or "std" in f or "variance" in f for f in failures if expected.split(":")[0] in f)


@pytest.mark.parametrize(
    ("name", "tamper", "expected"),
    [
        ("numeric", lambda m: _strength(m, "measurements", "pearson", ("linear_a", "linear_b"), 0.75), "Fisher z"),
        ("numeric", lambda m: _strength(m, "measurements", "spearman", ("copula_x", "copula_y"), 0.62), "Fisher z"),
        ("numeric", lambda m: _strength(m, "measurements", "correlation_ratio", ("nl_x", "nl_y"), 0.97), "bootstrap"),
        ("strings", lambda m: _strength(m, "contacts", "cramers_v", ("tier", "channel"), 0.17), "bootstrap"),
        ("strings", lambda m: _unique_fraction(m, 0.98), "unique share"),
    ],
    ids=["pearson", "spearman", "correlation-ratio", "cramers-v", "quasi-identifier"],
)
def test_verify_rejects_misdeclared_strength(
    name: str, tamper: Callable[[Manifest], Manifest], expected: str, spark: SparkSession
) -> None:
    failures = verify(tamper(materialize(name, "xs", 0).manifest), load(name, "xs", 0, spark))
    assert len(failures) == 1, failures
    assert expected in failures[0]


@pytest.mark.parametrize(
    ("table", "tamper", "expected"),
    [
        ("customers", lambda df: df.where(F.col("customer_id") % 2 == 0), "coverage"),
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


def test_verify_rejects_undeclared_nan(spark: SparkSession) -> None:
    frames = load("numeric", "xs", 0, spark)
    frames["measurements"] = frames["measurements"].withColumn(
        "ratio", F.when(F.col("id") % 100 == 0, F.lit(float("nan"))).otherwise(F.col("ratio")).cast("float")
    )
    failures = verify(materialize("numeric", "xs", 0).manifest, frames)
    assert "measurements: ratio: 50 undeclared NaN or infinite values" in failures


def test_verify_reads_storage_order_across_files(spark: SparkSession, tmp_path: Path) -> None:
    instance = materialize("temporal", "s", 0)
    frames = load("temporal", "s", 0, spark)
    assert instance.manifest.table("events").files > 1
    assert verify(instance.manifest, frames) == []
    reversed_events = instance.read_table("events")[::-1]
    pq.write_table(reversed_events, tmp_path / "events.parquet")
    frames["events"] = spark.read.parquet(str(tmp_path / "events.parquet"))
    failures = verify(instance.manifest, frames)
    assert any("event_id: values do not strictly increase" in f for f in failures), failures


def test_verify_requires_a_utc_session(spark: SparkSession) -> None:
    frames = load("numeric", "xs", 0, spark)
    spark.conf.set("spark.sql.session.timeZone", "America/New_York")
    try:
        with pytest.raises(ValueError, match="UTC"):
            verify(materialize("numeric", "xs", 0).manifest, frames)
    finally:
        spark.conf.set("spark.sql.session.timeZone", "UTC")
