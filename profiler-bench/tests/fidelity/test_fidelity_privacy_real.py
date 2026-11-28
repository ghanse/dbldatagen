"""Privacy guard changes for real datasets: structural text, single counting, independent rows, and membership copy checks."""

import re

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from dbldatagen_profiler.engines.spark import profile_tables
from dbldatagen_profiler.model import Profile
from profiler_bench.catalog import load, materialize
from profiler_bench.catalog.manifest import Manifest
from profiler_bench.fidelity import PrivacyScore, privacy, score_privacy
from profiler_bench.fidelity.config import load_config
from profiler_bench.fidelity.privacy import (
    _membership_checks,
    _row_copy_checks,
    _structural_paths,
    independent_rows,
    shuffled_columns,
)


PLANTED = "regio"


@pytest.fixture(scope="module")
def real(spark: SparkSession) -> dict[str, DataFrame]:
    return load("retail", "xs", 0, spark)


@pytest.fixture(scope="module")
def manifest() -> Manifest:
    return materialize("retail", "xs", 0).manifest


@pytest.fixture(scope="module")
def profile(real: dict[str, DataFrame]) -> Profile:
    return profile_tables(real, detect_pii=False)


@pytest.fixture(scope="module")
def planted(real: dict[str, DataFrame]) -> dict[str, DataFrame]:
    """Real tables whose first emails are structural words: a column-name substring, an enum value, the engine."""
    first = [r["customer_id"] for r in real["customers"].orderBy("customer_id").limit(3).collect()]
    email = F.col("email")
    for customer, value in zip(first, (PLANTED, "pearson", "pyspark"), strict=True):
        email = F.when(F.col("customer_id") == customer, F.lit(value)).otherwise(email)
    return {**real, "customers": real["customers"].withColumn("email", email)}


def _email_hits(score: PrivacyScore) -> list[str]:
    detail = next(c for c in score.checks if c.name == "profile.sensitive_value_leaks").detail
    return [k for k in detail if k.startswith("sensitive_value_in_profile:customers.email:")]


def _edit(profile: Profile, edit: object) -> Profile:
    data = profile.model_dump(mode="json")
    edit(data)  # type: ignore[operator]
    return Profile.model_validate(data)


def _column(data: dict, table: str, column: str) -> dict:
    return next(c for t in data["tables"] if t["name"] == table for c in t["columns"] if c["name"] == column)


def _rename_column(data: dict, table: str, old: str, new: str) -> None:
    """Renames a top-level column and every table-level reference to it: keys, constraints, dependencies, QI sets."""
    entry = next(t for t in data["tables"] if t["name"] == table)
    _column(data, table, old)["name"] = new
    keys = [entry["primary_key"]] if entry.get("primary_key") else []
    lists = [
        *(key["columns"] for key in [*keys, *(entry.get("unique_keys") or [])]),
        *(dependency["columns"] for dependency in entry.get("dependencies") or []),
        *(item["columns"] for item in entry.get("quasi_identifier_sets") or []),
    ]
    for names in lists:
        names[:] = [new if name == old else name for name in names]
    for constraint in entry.get("check_constraints") or []:
        for field in ("column", "left", "right"):
            if constraint.get(field) == old:
                constraint[field] = new


def test_structural_text_is_verified_against_schema_model_and_engine(
    planted: dict[str, DataFrame], profile: Profile
) -> None:
    dump = profile.model_dump(mode="json")
    structural = _structural_paths(dump, planted)
    assert "$.engine.name" in structural
    assert any(re.fullmatch(r"\$\.tables\[\d+\]\.columns\[\d+\]\.name", p) for p in structural)
    assert any(p.endswith(".source_type") for p in structural)
    assert any(p.endswith(".logical_type") for p in structural)
    assert not any(".top_values" in p or ".patterns" in p for p in structural)
    assert _structural_paths(dump, {}) == frozenset()


def test_structural_words_in_real_values_are_not_leaks(
    spark: SparkSession, planted: dict[str, DataFrame], profile: Profile, manifest: Manifest
) -> None:
    score = score_privacy(planted, load("retail", "xs", 0, spark, sample_seed=3), profile, manifest)
    assert _email_hits(score) == []


@pytest.mark.parametrize("where", ["pattern", "top_value", "fake_column_name"])
def test_deliberate_copies_still_fail(
    spark: SparkSession, planted: dict[str, DataFrame], profile: Profile, manifest: Manifest, where: str
) -> None:
    def edit(data: dict) -> None:
        sku = _column(data, "products", "sku")
        if where == "pattern":
            sku["string"] = {**(sku.get("string") or {}), "patterns": [{"regex": f"x{PLANTED}x", "coverage": 0.5}]}
        elif where == "top_value":
            sku["string"] = {**(sku.get("string") or {}), "top_values": [{"value": PLANTED, "frequency": 0.1}]}
        else:
            _column(data, "products", "department")["string"] = None
            _rename_column(data, "products", "department", f"{PLANTED}nx")

    edited = _edit(profile, edit)
    if where == "fake_column_name":
        products = next(t for t in edited.tables if t.name == "products")
        assert any(f"{PLANTED}nx" in dependency.columns for dependency in products.dependencies or [])
    score = score_privacy(planted, load("retail", "xs", 0, spark, sample_seed=3), edited, manifest)
    assert _email_hits(score), where
    leaks = next(c for c in score.checks if c.name == "profile.sensitive_value_leaks")
    assert not leaks.passed
    assert not score.passed


def test_a_planted_value_counts_once(
    spark: SparkSession, planted: dict[str, DataFrame], profile: Profile, manifest: Manifest
) -> None:
    def edit(data: dict) -> None:
        sku = _column(data, "products", "sku")
        sku["string"] = {**(sku.get("string") or {}), "top_values": [{"value": f"zz{PLANTED}zz", "frequency": 0.1}]}

    score = score_privacy(planted, load("retail", "xs", 0, spark, sample_seed=3), _edit(profile, edit), manifest)
    check = next(c for c in score.checks if c.name == "profile.sensitive_value_leaks")
    hits = _email_hits(score)
    assert len(hits) == 1 and "top_values" in hits[0]
    assert check.detail["sensitive_value_in_profile"] == 1.0
    assert "sensitive_value_in_profile_text" not in check.detail
    assert check.value == sum(v for k, v in check.detail.items() if ":" not in k)


def test_independent_rows_drop_rows_seen_by_the_profiler(spark: SparkSession, real: dict[str, DataFrame]) -> None:
    customers = real["customers"]
    seen = customers.limit(100)
    kept = independent_rows(customers.union(seen), seen)
    assert kept.count() == customers.count() - 100
    assert kept.join(seen, list(customers.columns), "left_semi").count() == 0


def _unseen(spark: SparkSession, real: dict[str, DataFrame], sample_seed: int = 4) -> dict[str, DataFrame]:
    """Returns another sample of the generator with every row the profiled sample has removed."""
    return {
        name: independent_rows(f, real[name])
        for name, f in load("retail", "xs", 0, spark, sample_seed=sample_seed).items()
    }


def test_too_few_independent_rows_are_unmeasured_not_failed(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile, manifest: Manifest
) -> None:
    synth = load("retail", "xs", 0, spark, sample_seed=3)
    holdout = {name: frame.limit(50) for name, frame in _unseen(spark, real).items()}
    score = score_privacy(
        real, synth, profile, manifest, holdout=holdout, chance=_shuffled(real), min_independent_rows=100
    )
    unmeasured = {c.name for c in score.checks if not c.measured}
    assert {
        "customers.dcr_ratio",
        "customers.email.value_leak_rate_excess",
        "customers.copy_rate_excess",
        "orders.partial_copy_rate_excess",
    } <= unmeasured
    assert all(
        not c.passed and c.detail["insufficient_independent_rows"] == 1.0 for c in score.checks if not c.measured
    )
    assert not [c for c in score.checks if c.measured and not c.passed]
    assert not [c for c in score.checks if c.measured and c.name.endswith(privacy.COPY_CHECKS)]
    assert (score.status, score.passed) == ("unmeasured", False)


def test_the_synthetic_suite_status_is_pass_or_fail(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile, manifest: Manifest
) -> None:
    independent = score_privacy(real, load("retail", "xs", 0, spark, sample_seed=3), profile, manifest)
    copied = score_privacy(real, real, profile, manifest)
    assert (independent.status, independent.passed) == ("pass", True)
    assert (copied.status, copied.passed) == ("fail", False)
    assert all(c.measured for c in [*independent.checks, *copied.checks])


def _shuffled(real: dict[str, DataFrame], seed: int = 0) -> dict[str, DataFrame]:
    return {name: shuffled_columns(frame, seed, name) for name, frame in real.items()}


def test_shuffled_columns_keep_marginals_and_schema_and_break_rows(real: dict[str, DataFrame]) -> None:
    items = real["order_items"]
    shuffled, again = shuffled_columns(items, 0, "order_items"), shuffled_columns(items, 0, "order_items")
    assert shuffled.schema == items.schema
    assert shuffled.collect() == again.collect()
    assert shuffled.collect() != shuffled_columns(items, 1, "order_items").collect()
    for column in items.columns:
        assert sorted(r[0] for r in shuffled.select(column).collect()) == sorted(
            r[0] for r in items.select(column).collect()
        )
    assert shuffled.join(items, list(items.columns), "left_semi").count() < items.count() // 10


def _copy_checks(score: PrivacyScore) -> dict[str, bool]:
    return {c.name: c.passed for c in score.checks if c.name.endswith("copy_rate_excess")}


def _membership(
    spark: SparkSession, real: dict[str, DataFrame], synth: dict[str, DataFrame], profile: Profile, manifest: Manifest
) -> PrivacyScore:
    return score_privacy(
        real, synth, profile, manifest, holdout=_unseen(spark, real), chance=_shuffled(real), min_independent_rows=100
    )


def test_an_ideal_sampler_with_dependent_columns_passes_the_membership_test(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile, manifest: Manifest
) -> None:
    synth = load("retail", "xs", 0, spark, sample_seed=3)
    score = _membership(spark, real, synth, profile, manifest)
    checks = {c.name: c for c in score.checks}
    assert all(c.detail["membership"] == 1.0 for n, c in checks.items() if n.endswith("copy_rate_excess"))
    orders = checks["orders.partial_copy_rate_excess"]
    assert orders.passed or orders.detail.get("underpowered") == 1.0, orders
    assert not [n for n, c in checks.items() if n.endswith("copy_rate_excess") and c.measured and not c.passed]
    assert score.passed, [c for c in score.checks if c.measured and not c.passed]


def test_five_percent_verbatim_copies_fail_the_membership_test(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile, manifest: Manifest
) -> None:
    independent = load("retail", "xs", 0, spark, sample_seed=3)
    items = independent["order_items"]
    copied = real["order_items"].orderBy("order_id", "line_no").limit(items.count() // 20)
    kept = items.orderBy("order_id", "line_no").limit(items.count() - copied.count())
    score = _membership(spark, real, {**independent, "order_items": kept.unionByName(copied)}, profile, manifest)
    check = next(c for c in score.checks if c.name == "order_items.copy_rate_excess")
    assert check.measured and not check.passed and check.value >= 0.03, check
    assert not score.passed


def test_an_exact_copy_of_the_profiled_part_fails_the_membership_test(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile, manifest: Manifest
) -> None:
    score = _membership(spark, real, real, profile, manifest)
    assert not any(passed for name, passed in _copy_checks(score).items() if name.endswith(".copy_rate_excess"))
    assert not score.passed


def test_key_only_tables_use_the_shuffled_control(real: dict[str, DataFrame]) -> None:
    config = load_config()
    bridge = real["order_items"].select("order_id", "product_id")
    shuffled = shuffled_columns(bridge, 0, "bridge")
    columns = ["order_id", "product_id"]
    kept = _row_copy_checks(
        "bridge", bridge, shuffled_columns(bridge, 7, "bridge"), bridge, shuffled, columns, [], 0, config, 100
    )
    copied = _row_copy_checks("bridge", bridge, bridge, bridge, shuffled, columns, [], 0, config, 100)
    assert [c.name for c in kept] == ["bridge.copy_rate_excess"]
    assert kept[0].detail["shuffled_control"] == 1.0 and not (kept[0].measured and not kept[0].passed), kept[0]
    assert copied[0].measured and not copied[0].passed, copied[0]


def test_copy_checks_that_cannot_catch_a_five_percent_copy_are_underpowered(
    spark: SparkSession, real: dict[str, DataFrame]
) -> None:
    config = load_config()
    columns = ["segment", "country"]
    seen = real["customers"].select(*columns)
    unseen = _unseen(spark, real)["customers"].select(*columns)
    synth = load("retail", "xs", 0, spark, sample_seed=3)["customers"].select(*columns)
    checks = _membership_checks("customers", seen, synth, unseen, columns, 0, config, None)
    assert [c.name for c in checks] == ["customers.copy_rate_excess", "customers.partial_copy_rate_excess"]
    for check in checks:
        assert (check.measured, check.passed, check.detail["underpowered"]) == (False, False, 1.0), check
        assert check.detail["alternative_excess"] <= check.detail["alternative_required"]


def _frames(spark: SparkSession, rows: int, values: int, seed: int) -> DataFrame:
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({"a": rng.integers(0, values, rows), "b": rng.integers(0, values, rows)}).astype("int64")
    return spark.createDataFrame(data)


TABLES = {
    "large": {"seen": 4_000, "unseen": 4_000, "values": 10**9, "powered": True},
    "small": {"seen": 60, "unseen": 60, "values": 10**9, "powered": False},
    "high_repeat": {"seen": 4_000, "unseen": 4_000, "values": 3, "powered": False},
    "high_overlap": {"seen": 4_000, "unseen": 120, "values": 10**9, "powered": False},
}


@pytest.mark.parametrize("shape", list(TABLES))
def test_powered_checks_fail_a_real_five_percent_copy_and_unpowered_ones_never_pass(
    spark: SparkSession, shape: str
) -> None:
    spec = TABLES[shape]
    config, columns = load_config(), ["a", "b"]
    seen = _frames(spark, spec["seen"], spec["values"], 1).cache()
    unseen = _frames(spark, spec["unseen"], spec["values"], 2).cache()
    synth = _frames(spark, spec["seen"], spec["values"], 3).cache()
    rows = synth.count()
    copied = synth.limit(rows - rows // 20).unionByName(seen.orderBy("a", "b").limit(rows // 20))
    clean = _membership_checks(shape, seen, synth, unseen, columns, 0, config, None)
    mixed = _membership_checks(shape, seen, copied, unseen, columns, 0, config, None)
    assert [c.name for c in clean] == [f"{shape}.copy_rate_excess", f"{shape}.partial_copy_rate_excess"]
    for before, after in zip(clean, mixed, strict=True):
        assert not (before.measured and not before.passed), before
        if spec["powered"]:
            assert before.measured and before.passed, before
            assert after.measured and not after.passed, after
        else:
            assert (before.measured, before.passed, before.detail.get("underpowered")) == (False, False, 1.0), before
            assert not after.passed, after
            assert after.detail.get("underpowered") == 1.0 or after.measured, after


def test_a_value_in_identical_strings_keeps_every_path(spark: SparkSession) -> None:
    values = spark.createDataFrame([("john", "john", 4)], "raw string, v string, n int")
    texts = [("xxjohnxx", "$.a"), ("xxjohnxx", "$.b"), ("other", "$.c")]
    assert privacy._candidate_substring_hits(values, spark, [], texts, [], 4) == {"john": {"$.a", "$.b"}}
