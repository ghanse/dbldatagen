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
    score = score_privacy(real, load("retail", "xs", 2, spark), profile, materialize("retail", "xs", 0).manifest)
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
    score = score_privacy(real, load("retail", "xs", 2, spark), leaky, materialize("retail", "xs", 0).manifest)
    detail = {c.name: c for c in score.checks}
    assert detail["profile.sensitive_value_leaks"].value == 1
    assert detail["profile.k_threshold"].detail == {"customers.email top_values": 1.0}
    assert not score.passed


def test_regex_literals_leak_without_the_raw_value(real: dict[str, DataFrame]) -> None:
    email = real["customers"].collect()[0]["email"]
    assert email in privacy.regex.literals("x|" + re.escape(email))


def test_quasi_identifier_sets_are_checked(
    spark: SparkSession, real: dict[str, DataFrame], profile: Profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        privacy,
        "quasi_identifier_sets",
        lambda t: (("segment", "country", "signup_date"),) if t.name == "customers" else (),
    )
    truth = materialize("retail", "xs", 0).manifest
    name = "customers.quasi_identifiers[segment,country,signup_date].match_rate_excess"
    assert not _checks(score_privacy(real, real, profile, truth))[name]
    assert _checks(score_privacy(real, load("retail", "xs", 2, spark), profile, truth))[name]
