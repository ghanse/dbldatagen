import json
import re
import time
from pathlib import Path

import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from dbldatagen_profiler.engines.spark import profile_tables
from dbldatagen_profiler.model import Profile
from profiler_bench.catalog import DATASETS, load, materialize
from profiler_bench.fidelity import PrivacyScore, privacy, regex, score_privacy, stats
from profiler_bench.fidelity.config import load_config


_EXAMPLES = Path(__file__).resolve().parents[3] / "profiler" / "spec" / "examples"


def _checks(score: PrivacyScore) -> dict[str, bool]:
    return {c.name: c.passed for c in score.checks}


@pytest.fixture(scope="module")
def real(spark: SparkSession) -> dict[str, DataFrame]:
    return load("retail", "xs", 0, spark)


@pytest.fixture(scope="module")
def profile(real: dict[str, DataFrame]) -> Profile:
    return profile_tables(real)


def test_independent_synth_passes(spark: SparkSession, real: dict[str, DataFrame], profile: Profile) -> None:
    score = score_privacy(
        real, load("retail", "xs", 0, spark, sample_seed=3), profile, materialize("retail", "xs", 0).manifest
    )
    assert score.passed, [c for c in score.checks if not c.passed]
    assert PrivacyScore.model_validate_json(score.model_dump_json()) == score


def test_copied_rows_fail_the_guard(real: dict[str, DataFrame], profile: Profile) -> None:
    score = score_privacy(real, real, profile, materialize("retail", "xs", 0).manifest)
    checks = _checks(score)
    assert not score.passed
    assert not checks["customers.copy_rate_excess"]
    assert not checks["customers.dcr_ratio"]
    assert not checks["customers.email.value_leak_rate_excess"]
    assert checks["profile.sensitive_value_leaks"]


def test_sensitive_values_in_the_profile_are_found(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile
) -> None:
    email = real["customers"].orderBy("customer_id").collect()[0]["email"]
    data = profile.model_dump(mode="json")
    customers = next(t for t in data["tables"] if t["name"] == "customers")
    column = next(c for c in customers["columns"] if c["name"] == "email")
    column["string"] = {
        "top_values": [{"value": email, "frequency": 0.5}],
        "patterns": [{"regex": re.escape(email.upper()) + "|" + re.escape(email), "coverage": 0.5}],
    }
    leaky = Profile.model_validate(data)
    score = score_privacy(
        real, load("retail", "xs", 0, spark, sample_seed=3), leaky, materialize("retail", "xs", 0).manifest
    )
    detail = {c.name: c for c in score.checks}
    assert detail["profile.sensitive_value_leaks"].detail["customers.email"] == 1.0
    assert detail["profile.k_threshold"].detail["customers.email top_values"] == 1.0
    assert detail["profile.sensitive_value_leaks"].detail["sensitive_pattern_not_generalized"] >= 1.0
    assert detail["profile.k_threshold"].detail["sensitive_pattern_not_generalized"] >= 1.0
    assert not score.passed


def test_regex_literals_leak_without_the_raw_value(real: dict[str, DataFrame]) -> None:
    email = real["customers"].collect()[0]["email"]
    assert email in privacy.regex.literals("x|" + re.escape(email))


def test_declared_quasi_identifier_sets_are_checked(spark: SparkSession) -> None:
    real = load("strings", "xs", 0, spark)
    truth = materialize("strings", "xs", 0).manifest
    profile = profile_tables(real)
    name = "contacts.quasi_identifiers[home_zip,birth_date,gender].match_rate_excess"
    assert not _checks(score_privacy(real, real, profile, truth))[name]
    independent = score_privacy(real, load("strings", "xs", 0, spark, sample_seed=3), profile, truth)
    assert independent.passed, [c for c in independent.checks if not c.passed]


def test_every_value_leak_class_is_checked(spark: SparkSession) -> None:
    real = load("strings", "xs", 0, spark)
    names = {
        c.name for c in score_privacy(real, real, profile_tables(real), materialize("strings", "xs", 0).manifest).checks
    }
    assert "contacts.session_token.value_leak_rate_excess" in names
    assert "contacts.comment.value_leak_rate_excess" in names
    assert not any("home_zip.value_leak" in n for n in names)


def _independent(spark: SparkSession) -> dict[str, DataFrame]:
    return load("retail", "xs", 0, spark, sample_seed=3)


def _with_patterns(
    profile: Profile,
    table: str,
    column: str,
    regexes: list[str],
    *,
    length_max: int | None = None,
) -> Profile:
    data = profile.model_dump(mode="json")
    target = next(c for t in data["tables"] if t["name"] == table for c in t["columns"] if c["name"] == column)
    string = {**(target.get("string") or {}), "patterns": [{"regex": r, "coverage": 0.5} for r in regexes]}
    if length_max is not None:
        string["length"] = {**(string.get("length") or {}), "min": 0, "max": length_max}
    target["string"] = string
    return Profile.model_validate(data)


def test_empty_synth_fails_every_table(real: dict[str, DataFrame], profile: Profile) -> None:
    score = score_privacy(real, {}, profile, materialize("retail", "xs", 0).manifest)
    assert not score.passed
    assert {c.name for c in score.checks if not c.passed} >= {f"{t}.synth_table_present" for t in real}


def test_missing_and_retyped_columns_fail_explicitly(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile
) -> None:
    synth = _independent(spark)
    synth["orders"] = synth["orders"].drop("status")
    synth["customers"] = synth["customers"].drop("email")
    synth["products"] = synth["products"].select(
        *(F.encode(F.col(c).cast("string"), "utf-8").alias(c) for c in synth["products"].columns)
    )
    synth["order_items"] = synth["order_items"].limit(0)
    checks = {c.name: c for c in score_privacy(real, synth, profile, materialize("retail", "xs", 0).manifest).checks}
    assert not checks["orders.synth_columns_present"].passed
    assert not checks["customers.email.value_leak_rate_excess"].passed
    assert checks["customers.email.value_leak_rate_excess"].detail == {"unmeasured": 1.0}
    assert checks["products.synth_columns_present"].value == 0.0
    assert not checks["products.copy_rate_excess"].passed
    assert not checks["products.dcr_ratio"].passed
    assert not checks["order_items.synth_rows_present"].passed


def test_obfuscated_literal_regexes_fail_structural_on_sensitive(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile
) -> None:
    email = real["customers"].orderBy("customer_id").collect()[0]["email"]
    grouped = "".join(f"({c})" if c.isalnum() else re.escape(c) for c in email)
    classes = "".join(f"[{c}]" if c.isalnum() else re.escape(c) for c in email)
    for pattern in (grouped, classes):
        leaky = _with_patterns(profile, "customers", "email", [pattern])
        score = score_privacy(real, _independent(spark), leaky, materialize("retail", "xs", 0).manifest)
        detail = {c.name: c for c in score.checks}
        assert detail["profile.sensitive_value_leaks"].detail["sensitive_pattern_not_generalized"] >= 1.0, pattern
        assert detail["profile.k_threshold"].detail["sensitive_pattern_not_generalized"] >= 1.0, pattern
        assert not score.passed


def test_rare_literal_in_a_pattern_fails_the_k_audit(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile
) -> None:
    sku = real["products"].orderBy("product_id").collect()[0]["sku"]
    pattern = "".join(f"({c})" for c in sku)
    leaky = _with_patterns(profile, "products", "sku", [pattern])
    score = score_privacy(real, _independent(spark), leaky, materialize("retail", "xs", 0).manifest)
    detail = {c.name: c for c in score.checks}
    assert detail["profile.k_threshold"].detail == {
        "products.sku pattern 0": 1.0,
        "products.sku pattern 0 literals": 1.0,
    }
    assert detail["profile.sensitive_value_leaks"].passed


def test_broad_patterns_are_not_flagged(spark: SparkSession, real: dict[str, DataFrame], profile: Profile) -> None:
    broad = _with_patterns(profile, "products", "sku", [r"[A-Z]{3}-\d{4}"])
    broad = _with_patterns(broad, "customers", "email", [r"[^\s]+"])
    score = score_privacy(real, _independent(spark), broad, materialize("retail", "xs", 0).manifest)
    assert score.passed, [c for c in score.checks if not c.passed]


def test_literal_email_pattern_fails_structural_on_sensitive(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile
) -> None:
    leaky = _with_patterns(profile, "customers", "email", [r"[a-z0-9]{3,12}@[a-z]{2,8}\.[a-z]{2,3}"])
    score = score_privacy(real, _independent(spark), leaky, materialize("retail", "xs", 0).manifest)
    detail = {c.name: c for c in score.checks}
    assert detail["profile.sensitive_value_leaks"].detail["sensitive_pattern_not_generalized"] >= 1.0
    assert detail["profile.k_threshold"].detail["sensitive_pattern_not_generalized"] >= 1.0
    assert not score.passed


def test_decoy_alternation_fails_structural_and_per_branch(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile
) -> None:
    decoy = r"(?:(j)(o)(h)(n)[@](x)[.](c)(o)(m)|[a-z0-9@.]{10})"
    sensitive = score_privacy(
        real,
        _independent(spark),
        _with_patterns(profile, "customers", "email", [decoy]),
        materialize("retail", "xs", 0).manifest,
    )
    detail = {c.name: c for c in sensitive.checks}
    assert detail["profile.sensitive_value_leaks"].detail["sensitive_pattern_not_generalized"] >= 1.0
    assert detail["profile.k_threshold"].detail["sensitive_pattern_not_generalized"] >= 1.0
    assert not sensitive.passed

    planted = real["products"].withColumn(
        "sku",
        F.when(F.row_number().over(Window.orderBy("product_id")) == 1, F.lit("john@x.com")).otherwise(F.col("sku")),
    )
    real_planted = {**real, "products": planted}
    non_sensitive = score_privacy(
        real_planted,
        _independent(spark),
        _with_patterns(profile, "products", "sku", [decoy]),
        materialize("retail", "xs", 0).manifest,
    )
    detail = {c.name: c for c in non_sensitive.checks}
    assert not non_sensitive.passed
    assert (
        detail["profile.k_threshold"].detail.get("products.sku pattern 0", 0.0) >= 1.0
        or detail["profile.k_threshold"].detail.get("products.sku pattern 0 literals", 0.0) >= 1.0
        or detail["profile.sensitive_value_leaks"].detail.get("customers.email", 0.0) >= 1.0
    ), detail["profile.k_threshold"].detail


def test_dfa_budget_rejects_adversarial_pattern(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile
) -> None:
    adversarial = r"(a|b)*a(a|b){25}"
    started = time.perf_counter()
    first = regex.count_matches(adversarial, 21, 29, max_dfa_states=10_000)
    second = regex.count_matches(adversarial, 21, 29, max_dfa_states=10_000)
    assert time.perf_counter() - started < 1.0
    assert first is regex.BUDGET_EXCEEDED
    assert second is regex.BUDGET_EXCEEDED

    # Sensitive column: literal `a` fails the structural rule.
    leaky = _with_patterns(profile, "customers", "email", [adversarial])
    score = score_privacy(real, _independent(spark), leaky, materialize("retail", "xs", 0).manifest)
    detail = {c.name: c for c in score.checks}
    assert detail["profile.sensitive_value_leaks"].detail["sensitive_pattern_not_generalized"] >= 1.0
    assert detail["profile.k_threshold"].detail["sensitive_pattern_not_generalized"] >= 1.0
    assert not score.passed

    # Non-sensitive column: the length-cap probe still hits the shared budget.
    budgeted = score_privacy(
        real,
        _independent(spark),
        _with_patterns(profile, "products", "sku", [adversarial]),
        materialize("retail", "xs", 0).manifest,
    )
    detail = {c.name: c for c in budgeted.checks}
    assert detail["profile.sensitive_value_leaks"].detail["regex_budget_exceeded"] >= 1.0
    assert detail["profile.k_threshold"].detail["regex_budget_exceeded"] >= 1.0
    assert not budgeted.passed


def test_unbounded_declared_length_does_not_hide_or_iterate(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile
) -> None:
    broad = _with_patterns(profile, "products", "sku", [r"a*"], length_max=10**12)
    started = time.perf_counter()
    risks = privacy._pattern_risks(broad, load_config(), set())
    assert time.perf_counter() - started < 1.0
    sku_risks = [r for (table, path, _), r in risks.items() if table == "products" and path == "sku"]
    assert sku_risks and all(not r.budget_exceeded for r in sku_risks)
    score = score_privacy(real, _independent(spark), broad, materialize("retail", "xs", 0).manifest)
    assert score.passed, [c for c in score.checks if not c.passed]
    assert "regex_budget_exceeded" not in {c.name: c for c in score.checks}["profile.k_threshold"].detail


def test_case_class_pattern_fails_structural_on_sensitive(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile
) -> None:
    for pattern in (
        r"[jJ][oO][hH][nN][@][xX][.][cC][oO][mM]",
        r"[jJ][oO][hH][nN]@x\.com.*",
    ):
        leaky = _with_patterns(profile, "customers", "email", [pattern], length_max=10**12)
        score = score_privacy(real, _independent(spark), leaky, materialize("retail", "xs", 0).manifest)
        detail = {c.name: c for c in score.checks}
        assert detail["profile.sensitive_value_leaks"].detail["sensitive_pattern_not_generalized"] >= 1.0, pattern
        assert not score.passed


def test_dialect_rejects_comments_flags_lookarounds_and_named_groups(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile
) -> None:
    for pattern in (
        r"(?#john@x.com)[^\s]+",
        r"(?i)JOHN@X\.COM",
        r"(?=john)[^\s]+",
        r"(?P<john_x_com>a)",
    ):
        leaky = _with_patterns(profile, "customers", "email", [pattern])
        score = score_privacy(real, _independent(spark), leaky, materialize("retail", "xs", 0).manifest)
        detail = {c.name: c for c in score.checks}
        assert detail["profile.sensitive_value_leaks"].detail["pattern_not_in_dialect"] >= 1.0, pattern
        assert detail["profile.k_threshold"].detail["pattern_not_in_dialect"] >= 1.0, pattern
        assert not score.passed


def test_sensitive_value_in_non_regex_profile_field_fails_raw_scan(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile
) -> None:
    email = real["customers"].orderBy("customer_id").collect()[0]["email"]
    data = profile.model_dump(mode="json")
    products = next(t for t in data["tables"] if t["name"] == "products")
    sku = next(c for c in products["columns"] if c["name"] == "sku")
    sku["string"] = {**(sku.get("string") or {}), "top_values": [{"value": email, "frequency": 0.1}]}
    leaky = Profile.model_validate(data)
    score = score_privacy(real, _independent(spark), leaky, materialize("retail", "xs", 0).manifest)
    detail = {c.name: c for c in score.checks}
    assert detail["profile.sensitive_value_leaks"].detail["sensitive_value_in_profile_text"] >= 1.0
    assert detail["profile.k_threshold"].detail["sensitive_value_in_profile_text"] >= 1.0
    assert not score.passed


def test_split_sensitive_value_across_source_types_fails_closed_strings(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile
) -> None:
    data = profile.model_dump(mode="json")
    email_col = next(
        c for t in data["tables"] if t["name"] == "customers" for c in t["columns"] if c["name"] == "email"
    )
    sku = next(c for t in data["tables"] if t["name"] == "products" for c in t["columns"] if c["name"] == "sku")
    email_col["source_type"] = "john"
    sku["source_type"] = "@x.com"
    leaky = Profile.model_validate(data)
    score = score_privacy(real, _independent(spark), leaky, materialize("retail", "xs", 0).manifest)
    detail = {c.name: c for c in score.checks}
    assert detail["profile.sensitive_value_leaks"].detail["unexpected_profile_string"] >= 2.0
    assert detail["profile.k_threshold"].detail["unexpected_profile_string"] >= 2.0
    assert not score.passed


def test_percent_encoded_sensitive_value_fails_raw_scan(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile
) -> None:
    email = real["customers"].orderBy("customer_id").collect()[0]["email"]
    encoded = email.replace("@", "%40")
    data = profile.model_dump(mode="json")
    sku = next(c for t in data["tables"] if t["name"] == "products" for c in t["columns"] if c["name"] == "sku")
    sku["string"] = {**(sku.get("string") or {}), "top_values": [{"value": encoded, "frequency": 0.1}]}
    leaky = Profile.model_validate(data)
    score = score_privacy(real, _independent(spark), leaky, materialize("retail", "xs", 0).manifest)
    detail = {c.name: c for c in score.checks}
    assert detail["profile.sensitive_value_leaks"].detail["sensitive_value_in_profile_text"] >= 1.0
    assert detail["profile.k_threshold"].detail["sensitive_value_in_profile_text"] >= 1.0
    assert not score.passed


def test_source_type_not_matching_schema_fails_closed_strings(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile
) -> None:
    data = profile.model_dump(mode="json")
    sku = next(c for t in data["tables"] if t["name"] == "products" for c in t["columns"] if c["name"] == "sku")
    sku["source_type"] = "varchar(32)"
    leaky = Profile.model_validate(data)
    score = score_privacy(real, _independent(spark), leaky, materialize("retail", "xs", 0).manifest)
    detail = {c.name: c for c in score.checks}
    assert detail["profile.sensitive_value_leaks"].detail["unexpected_profile_string"] >= 1.0
    assert any(
        key.startswith("unexpected_profile_string:$") and key.endswith(".source_type")
        for key in detail["profile.sensitive_value_leaks"].detail
    )
    assert not score.passed


def test_example_profiles_pass_closed_strings() -> None:
    for path in sorted(_EXAMPLES.glob("*.json")):
        raw = json.loads(path.read_text())
        profile = Profile.model_validate(raw)
        assert privacy._unexpected_profile_strings(profile, {}, data=raw) == {}, path.name


def test_baseline_profiles_of_every_xs_dataset_pass_closed_strings(spark: SparkSession) -> None:
    for name in DATASETS:
        real = load(name, "xs", 0, spark)
        profile = profile_tables(real)
        assert privacy._unexpected_profile_strings(profile, real) == {}, name


def test_partial_row_copying_is_detected(spark: SparkSession, real: dict[str, DataFrame], profile: Profile) -> None:
    synth = _independent(spark)
    copied = ["email", "signup_date", "country"]
    order = Window.orderBy("customer_id")
    kept = synth["customers"].drop(*copied).withColumn("__r", F.row_number().over(order))
    source = real["customers"].select(*copied, F.row_number().over(order).alias("__r"))
    synth["customers"] = kept.join(source, "__r").select(*real["customers"].columns)
    checks = {c.name: c for c in score_privacy(real, synth, profile, materialize("retail", "xs", 0).manifest).checks}
    assert checks["customers.copy_rate_excess"].passed
    partial = checks["customers.partial_copy_rate_excess"]
    assert not partial.passed
    assert partial.value > 0.9
    assert partial.detail["failed_pairs"] >= 3


def test_dcr_weights_each_column_at_most_about_one() -> None:
    reference = pd.DataFrame({"n": [0.0, 1.0, 2.0, 3.0], "c": ["a", "b", "a", "b"], "s": ["x1", "x2", "x3", "x4"]})
    kinds = {"n": "numeric", "c": "categorical", "s": "string"}
    encoder = stats.Encoder(reference, kinds, 20)
    row = reference.iloc[[0]]
    flipped = row.assign(c="b")
    distance = privacy._closest(encoder, flipped, row)
    assert distance[0] == pytest.approx(1.0)
    renamed = row.assign(s="zz")
    one_string = privacy._closest(encoder, renamed, row)[0] ** 2
    assert 1.0 <= one_string <= 2.0 + 1e-9
    shifted = row.assign(n=2.0)
    assert privacy._closest(encoder, shifted, row)[0] == pytest.approx(0.5)


def test_literal_split_across_groups_fails_structural_on_sensitive(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile
) -> None:
    email = real["customers"].orderBy("customer_id").collect()[0]["email"]
    local, domain = email.split("@")
    patterns = [
        "".join(f"({c})" for c in local[:4]) + r"[a-z.0-9]{0,30}@" + re.escape(domain),
        f"({email[0]}|{email[0].upper()})" + "".join(f"({c})" if c.isalnum() else re.escape(c) for c in email[1:]),
    ]
    for pattern in patterns:
        leaky = _with_patterns(profile, "customers", "email", [pattern])
        score = score_privacy(real, _independent(spark), leaky, materialize("retail", "xs", 0).manifest)
        detail = {c.name: c for c in score.checks}
        assert detail["profile.sensitive_value_leaks"].detail["sensitive_pattern_not_generalized"] >= 1.0, pattern
        assert not score.passed


def test_generalized_sensitive_pattern_and_shared_sku_literal_pass(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile
) -> None:
    broad = _with_patterns(profile, "customers", "email", [r"[^\s]+"])
    # Non-sensitive SKU: a short shared punctuation literal is below min_literal_len and the language is large.
    broad = _with_patterns(broad, "products", "sku", [r"[A-Z]{3}-\d{4}"])
    score = score_privacy(real, _independent(spark), broad, materialize("retail", "xs", 0).manifest)
    assert score.passed, [c for c in score.checks if not c.passed]
