"""Tests for Cramér's V dependencies between low-cardinality categorical columns."""

from types import SimpleNamespace

import pytest
from pyspark.sql import DataFrame, SparkSession

from dbldatagen_profiler.engines.spark import engine as spark_engine
from dbldatagen_profiler.engines.spark import profile_dataframe
from dbldatagen_profiler.engines.spark.engine import _cramers_pairs, _cramers_v
from dbldatagen_profiler.model import DependenceMethod, PrivacyPolicy, Sensitivity
from dbldatagen_profiler.pii import PiiAction, PiiPolicy, PiiRule


pytestmark = [pytest.mark.spark, pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")]


def _categorical_frame(spark: SparkSession, rows: int = 400) -> DataFrame:
    countries = ["US", "CA", "MX", "FR", "DE", "JP"]
    regions = {"US": "NA", "CA": "NA", "MX": "NA", "FR": "EU", "DE": "EU", "JP": "APAC"}
    data = [
        (index, countries[index % 6], regions[countries[index % 6]], ("a", "b")[(index // 7) % 2], index % 5 == 0)
        for index in range(rows)
    ]
    return spark.createDataFrame(data, "customer_id int, country string, region string, noise string, flag boolean")


def _cramers(frame: DataFrame, **kwargs: object) -> dict[tuple[str, str], float]:
    table = profile_dataframe("t", frame, **kwargs).tables[0]  # type: ignore[arg-type]
    return {
        dependency.columns: dependency.value
        for dependency in table.dependencies or []
        if dependency.method == DependenceMethod.CRAMERS_V
    }


def test_profiles_value_free_cramers_v_for_dependent_categorical_pairs(spark: SparkSession) -> None:
    frame = _categorical_frame(spark)

    actual = _cramers(frame, detect_pii=False)

    assert list(actual) == [("country", "region")]
    assert actual[("country", "region")] == pytest.approx(1.0, abs=0.01)


def test_cramers_v_matches_bias_corrected_reference() -> None:
    assert _cramers_v({("a", "x"): 50, ("b", "y"): 50}) == pytest.approx(1.0)
    assert _cramers_v({("a", "x"): 25, ("a", "y"): 25, ("b", "x"): 25, ("b", "y"): 25}) == 0.0
    assert _cramers_v({("a", "x"): 50}) is None
    cells = {("a", "x"): 30, ("a", "y"): 10, ("b", "x"): 10, ("b", "y"): 30}
    chi2 = 80 * 0.25
    phi2 = chi2 / 80 - 1 / 79
    assert _cramers_v(cells) == pytest.approx((phi2 / (2 - 1 / 79 - 1)) ** 0.5)


def test_cramers_v_drops_cells_below_privacy_min_value_count(spark: SparkSession) -> None:
    rows = [(value, value) for value in ("a", "b", "c") for _ in range(4)]
    frame = spark.createDataFrame(rows, "left string, right string")

    assert _cramers(frame, detect_pii=False) == {}
    assert _cramers(frame, detect_pii=False, privacy=PrivacyPolicy(min_value_count=4)) == {
        ("left", "right"): pytest.approx(1.0)
    }


def test_cramers_v_skips_policy_covered_columns(spark: SparkSession) -> None:
    policy = PiiPolicy(columns={"t.region": PiiRule(action=PiiAction.SUBSTITUTE)})

    assert _cramers(_categorical_frame(spark), detect_pii=False, pii_policy=policy) == {}
    keep = PiiPolicy(columns={"t.region": PiiRule(action=PiiAction.KEEP)})
    assert list(_cramers(_categorical_frame(spark), detect_pii=False, pii_policy=keep)) == [("country", "region")]


def test_cramers_pairs_exclude_keys_sensitive_columns_and_quasi_identifier_pairs(spark: SparkSession) -> None:
    frame = spark.createDataFrame(
        [(f"o{index % 3}", f"s{index % 4}", f"e{index % 5}", f"q{index % 6}", f"r{index % 7}") for index in range(60)],
        "order_id string, secret string, email string, zip string, city string",
    )
    values = {f"c{index}_distinct": distinct for index, distinct in enumerate((3, 4, 5, 6, 7))}
    semantics = {
        "secret": SimpleNamespace(sensitivity=Sensitivity.SECRET),
        "email": SimpleNamespace(sensitivity=Sensitivity.DIRECT_IDENTIFIER),
        "zip": SimpleNamespace(sensitivity=Sensitivity.QUASI_IDENTIFIER),
        "city": SimpleNamespace(sensitivity=Sensitivity.QUASI_IDENTIFIER),
    }

    assert _cramers_pairs(frame, values, "t", semantics, None, []) == []
    del semantics["city"]
    assert _cramers_pairs(frame, values, "t", semantics, None, []) == [("city", "zip")]
    assert _cramers_pairs(frame, values, "t", semantics, None, [(["zip", "city"], 0.5, 2)]) == []


def test_cramers_pairs_are_capped_and_each_column_joins_one_dependency(
    spark: SparkSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = [f"c{index}" for index in range(10)]
    frame = spark.createDataFrame(
        [tuple(f"v{(row // 2) % 3}" for _ in names) for row in range(120)], ", ".join(f"{n} string" for n in names)
    )
    monkeypatch.setattr(spark_engine, "_MAX_CRAMERS_PAIRS", 7)
    calls: list[int] = []
    pair_counts = spark_engine._pair_counts

    def recording(df: DataFrame, pairs: list[tuple[str, str]]) -> list[dict[tuple[str, str], int]]:
        calls.append(len(pairs))
        return pair_counts(df, pairs)

    monkeypatch.setattr(spark_engine, "_pair_counts", recording)

    first = _cramers(frame, detect_pii=False)
    second = _cramers(frame, detect_pii=False)

    assert calls == [7, 7]
    assert first == second
    assert list(first) == [("c0", "c1"), ("c2", "c3"), ("c4", "c5"), ("c6", "c7")]
    assert all(value == pytest.approx(1.0) for value in first.values())


def test_cramers_v_ignores_integer_high_cardinality_and_weak_pairs(spark: SparkSession) -> None:
    rows = [(f"id{index}", f"g{index % 4}", f"h{(index // 4) % 4}", index % 4, index % 4) for index in range(400)]
    frame = spark.createDataFrame(rows, "label string, left string, right string, x int, y int")

    assert _cramers(frame, detect_pii=False) == {}
