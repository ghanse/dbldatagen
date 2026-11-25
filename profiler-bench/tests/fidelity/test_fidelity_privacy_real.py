"""Privacy guard changes for real datasets: structural text, single counting, independent rows, and membership copy checks."""

import re

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from dbldatagen_profiler.engines.spark import profile_tables
from dbldatagen_profiler.model import Profile
from profiler_bench.catalog import load, materialize
from profiler_bench.catalog.manifest import Manifest
from profiler_bench.fidelity import PrivacyScore, score_privacy
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
            department = _column(data, "products", "department")
            department["name"], department["string"] = f"{PLANTED}nx", None

    score = score_privacy(planted, load("retail", "xs", 0, spark, sample_seed=3), _edit(profile, edit), manifest)
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
    assert score.passed, [c for c in score.checks if c.measured and not c.passed]


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
        assert check.detail["detectable_copy_excess"] <= check.detail["required_excess"]
