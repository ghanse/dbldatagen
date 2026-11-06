import re

import pytest
from pyspark.sql import DataFrame, SparkSession

from dbldatagen_profiler.engines.spark import profile_tables
from dbldatagen_profiler.model import Profile
from profiler_bench.catalog import load, materialize
from profiler_bench.fidelity import PrivacyScore, privacy, score_privacy


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
    assert detail["profile.sensitive_value_leaks"].value == 1
    assert detail["profile.k_threshold"].detail == {"customers.email top_values": 1.0}
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
