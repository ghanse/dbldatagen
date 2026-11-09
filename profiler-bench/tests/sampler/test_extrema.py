"""Regression coverage for exact profiled extrema leaking into samples."""

from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from pyspark.sql import SparkSession
from scipy import stats

from dbldatagen_profiler.engines.spark import profile_tables
from dbldatagen_profiler.model import Histogram, LogicalType, NumericStats, Quantile, TableProfile, TemporalStats
from profiler_bench.catalog import load
from profiler_bench.runner.scorer import try_real_scorer
from profiler_bench.sampler import sample
from profiler_bench.sampler.core import (
    _calibrate_truncated_parameters,
    _standard_normal_cdf,
    _truncated_moments,
    _truncated_normal,
)

from .profiles import column, profile


def test_truncated_normal_continuous_avoids_exact_extrema(spark: SparkSession) -> None:
    """Skewed continuous columns must not pile mass on the exact profiled min/max."""
    stats = NumericStats(min=1.0, max=1000.0, mean=3.0, stddev=20.0)
    result = sample(
        profile(TableProfile(name="t", row_count=20_000, columns=[column("x", LogicalType.FLOAT64, numeric=stats)])),
        spark,
        seed=11,
    )
    values = np.asarray([row.x for row in result.tables["t"].collect()], dtype=float)

    assert float(np.mean((values == stats.min) | (values == stats.max))) < 0.001
    assert "numeric.truncated_normal" in result.report.columns["t.x"].approximated


def test_truncated_normal_integer_endpoint_mass_matches_model(spark: SparkSession) -> None:
    """Integer endpoints stay reachable, but only at truncated-normal rounded tail mass."""
    stats = NumericStats(min=0, max=12, mean=4.0, stddev=2.0)
    rows = (
        sample(
            profile(TableProfile(name="t", row_count=40_000, columns=[column("x", LogicalType.INT32, numeric=stats)])),
            spark,
            seed=13,
        )
        .tables["t"]
        .collect()
    )
    values = np.asarray([row.x for row in rows], dtype=int)
    mean = float(stats.mean)
    stddev = float(stats.stddev)
    fa = _standard_normal_cdf((0.0 - mean) / stddev)
    fb = _standard_normal_cdf((12.0 - mean) / stddev)
    expected_min = (_standard_normal_cdf((0.5 - mean) / stddev) - fa) / (fb - fa)
    expected_max = (fb - _standard_normal_cdf((11.5 - mean) / stddev)) / (fb - fa)

    assert float(np.mean(values == 0)) == pytest.approx(expected_min, abs=0.01)
    assert float(np.mean(values == 12)) == pytest.approx(expected_max, abs=0.005)
    assert float(np.mean(values == 0)) < 0.05


@pytest.mark.parametrize(("low", "high"), [(8.0, 9.0), (-40.0, -39.0)])
def test_far_tail_truncated_normal_is_stable_and_non_uniform(low: float, high: float) -> None:
    """Far-tail windows must not collapse to uniform or a single float from Φ saturation."""
    uniforms = np.linspace(np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0), 8_000)
    values = _truncated_normal(uniforms, 0.0, 1.0, low, high, open_bounds=True)
    again = _truncated_normal(uniforms, 0.0, 1.0, low, high, open_bounds=True)
    dist = stats.truncnorm(low, high, loc=0.0, scale=1.0)

    assert values.min() > low
    assert values.max() < high
    assert float(np.mean((values == low) | (values == high))) < 0.001
    assert np.allclose(values, again)
    assert stats.kstest(values, dist.cdf).pvalue > 0.05
    assert stats.kstest(values, "uniform", args=(low, high - low)).pvalue < 1e-6


def test_truncated_normal_zero_and_missing_stddev_fallbacks(spark: SparkSession) -> None:
    """stddev=0 is a point mass at the clipped mean; missing stddev falls back to uniform."""
    zero = _truncated_normal(np.asarray([0.1, 0.5, 0.9]), 5.0, 0.0, 0.0, 10.0, open_bounds=True)
    assert np.all(zero == 5.0)

    missing = NumericStats(min=2.0, max=8.0, mean=5.0)
    result = sample(
        profile(TableProfile(name="t", row_count=5_000, columns=[column("x", LogicalType.FLOAT64, numeric=missing)])),
        spark,
        seed=17,
    )
    values = np.asarray([row.x for row in result.tables["t"].collect()], dtype=float)

    assert "numeric.uniform" in result.report.columns["t.x"].approximated
    assert float(values.mean()) == pytest.approx(5.0, abs=0.08)
    assert float(np.mean((values == 2.0) | (values == 8.0))) < 0.001


def test_exact_int64_and_decimal_truncated_normal_preserve_domain(spark: SparkSession) -> None:
    """Exact int64/decimal draws keep the profiled mean/sd; float(mu) near 2^63 must not shift them."""
    row_count = 8_000
    int_mean = 2**63 - 10_000
    int_sd = 100
    int_low = int_mean - 1_000
    int_high = int_mean + 1_000
    int_stats = NumericStats(min=str(int_low), max=str(int_high), mean=str(int_mean), stddev=str(int_sd))
    int_values = np.asarray(
        [
            row.x
            for row in sample(
                profile(
                    TableProfile(
                        name="t",
                        row_count=row_count,
                        columns=[column("x", LogicalType.INT64, numeric=int_stats)],
                    )
                ),
                spark,
                seed=19,
            )
            .tables["t"]
            .collect()
        ],
        dtype=object,
    )
    int_as_float = np.asarray([float(value - int_mean) for value in int_values], dtype=float)
    int_se = int_sd / math.sqrt(row_count)
    assert all(isinstance(value, int) and int_low <= value <= int_high for value in int_values)
    assert float(np.mean([value in (int_low, int_high) for value in int_values])) < 0.05
    assert abs(float(np.mean(int_as_float))) <= 3 * int_se
    assert float(np.std(int_as_float, ddof=1)) == pytest.approx(int_sd, rel=0.10)

    # Near decimal(38,10)'s ~1e28 magnitude, float(low)==float(high)==float(mean).
    # mean+1000 == 1e28 needs 29 integer digits, so the upper bound is the type max (~mean+1000).
    dec_mean = Decimal("9999999999999999999999999000.0000000000")
    dec_sd = Decimal("100.0000000000")
    dec_low = dec_mean - Decimal("1000.0000000000")
    dec_high = Decimal("9999999999999999999999999999.9999999999")
    assert float(dec_low) == float(dec_high) == float(dec_mean)
    dec_stats = NumericStats(
        min=format(dec_low, "f"),
        max=format(dec_high, "f"),
        mean=format(dec_mean, "f"),
        stddev=format(dec_sd, "f"),
    )
    dec_values = [
        row.amount
        for row in sample(
            profile(
                TableProfile(
                    name="t",
                    row_count=row_count,
                    columns=[column("amount", LogicalType.DECIMAL, precision=38, scale=10, numeric=dec_stats)],
                )
            ),
            spark,
            seed=21,
        )
        .tables["t"]
        .collect()
    ]
    dec_offsets = np.asarray([float(value - dec_mean) for value in dec_values], dtype=float)
    dec_se = float(dec_sd) / math.sqrt(row_count)
    assert all(isinstance(value, Decimal) and dec_low <= value <= dec_high for value in dec_values)
    assert all(value.as_tuple().exponent == -10 for value in dec_values)
    assert float(np.mean([value in (dec_low, dec_high) for value in dec_values])) < 0.05
    assert abs(float(np.mean(dec_offsets))) <= 3 * dec_se
    assert float(np.std(dec_offsets, ddof=1)) == pytest.approx(float(dec_sd), rel=0.10)


def test_truncated_normal_mean_stddev_calibration_or_documented_shift(spark: SparkSession) -> None:
    """Mild truncation recovers moments; unattainable skew documents the truncated shift."""
    mild = NumericStats(min=-4.0, max=4.0, mean=0.5, stddev=0.8)
    mu, sigma = _calibrate_truncated_parameters(0.5, 0.8, -4.0, 4.0)
    matched_mean, matched_std = _truncated_moments(mu, sigma, -4.0, 4.0)
    mild_values = np.asarray(
        [
            row.x
            for row in sample(
                profile(
                    TableProfile(name="t", row_count=8_000, columns=[column("x", LogicalType.FLOAT64, numeric=mild)])
                ),
                spark,
                seed=12,
            )
            .tables["t"]
            .collect()
        ],
        dtype=float,
    )
    assert matched_mean == pytest.approx(0.5, abs=0.01)
    assert matched_std == pytest.approx(0.8, abs=0.01)
    assert float(mild_values.mean()) == pytest.approx(0.5, abs=0.05)
    assert float(mild_values.std(ddof=1)) == pytest.approx(0.8, abs=0.05)

    skew = NumericStats(min=1.0, max=1000.0, mean=3.0, stddev=20.0)
    expected_mean, expected_std = _truncated_moments(3.0, 20.0, 1.0, 1000.0)
    # No truncated-normal (mu, sigma) matches mean=3, std=20 on [1, 1000]; sampler keeps the
    # profiled parameters and the draws follow the shifted truncated moments above.
    assert _calibrate_truncated_parameters(3.0, 20.0, 1.0, 1000.0) == (3.0, 20.0)
    skew_values = np.asarray(
        [
            row.x
            for row in sample(
                profile(
                    TableProfile(name="t", row_count=12_000, columns=[column("x", LogicalType.FLOAT64, numeric=skew)])
                ),
                spark,
                seed=23,
            )
            .tables["t"]
            .collect()
        ],
        dtype=float,
    )
    assert float(skew_values.mean()) == pytest.approx(expected_mean, rel=0.08)
    assert float(skew_values.std(ddof=1)) == pytest.approx(expected_std, rel=0.12)


def test_histogram_path_avoids_exact_endpoints(spark: SparkSession) -> None:
    """Degenerate outer histogram bins must not emit the exact outer edge as a point mass."""
    stats = NumericStats(histogram=Histogram(edges=[0.0, 0.0, 10.0], counts=[1, 3]))
    values = np.asarray(
        [
            row.x
            for row in sample(
                profile(
                    TableProfile(name="t", row_count=8_000, columns=[column("x", LogicalType.FLOAT64, numeric=stats)])
                ),
                spark,
                seed=29,
            )
            .tables["t"]
            .collect()
        ],
        dtype=float,
    )
    assert float(np.mean((values == 0.0) | (values == 10.0))) < 0.001


def test_quantile_path_avoids_exact_endpoints(spark: SparkSession) -> None:
    """Quantile plateaus at q=0/1 must not pile continuous mass on the exact endpoint value."""
    stats = NumericStats(
        quantiles=[Quantile(q=0, value=2.0), Quantile(q=0.5, value=2.0), Quantile(q=1, value=8.0)],
        min=2.0,
        max=8.0,
    )
    values = np.asarray(
        [
            row.x
            for row in sample(
                profile(
                    TableProfile(name="t", row_count=8_000, columns=[column("x", LogicalType.FLOAT64, numeric=stats)])
                ),
                spark,
                seed=31,
            )
            .tables["t"]
            .collect()
        ],
        dtype=float,
    )
    assert float(np.mean((values == 2.0) | (values == 8.0))) < 0.001


def test_uniform_path_avoids_exact_endpoints(spark: SparkSession) -> None:
    """Min/max-only continuous uniforms stay inside the open interval."""
    stats = NumericStats(min=2.0, max=8.0)
    result = sample(
        profile(TableProfile(name="t", row_count=8_000, columns=[column("x", LogicalType.FLOAT64, numeric=stats)])),
        spark,
        seed=33,
    )
    values = np.asarray([row.x for row in result.tables["t"].collect()], dtype=float)
    assert "numeric.uniform" in result.report.columns["t.x"].approximated
    assert float(np.mean((values == 2.0) | (values == 8.0))) < 0.001


def test_temporal_uniform_path_avoids_exact_endpoints(spark: SparkSession) -> None:
    """Temporal min/max uniforms avoid exact endpoint timestamps."""
    stats = TemporalStats(min="2020-01-01T00:00:00", max="2020-01-03T00:00:00")
    values = np.asarray(
        [
            np.datetime64(row.x)
            for row in sample(
                profile(
                    TableProfile(
                        name="t",
                        row_count=8_000,
                        columns=[column("x", LogicalType.TIMESTAMP_NTZ, temporal=stats)],
                    )
                ),
                spark,
                seed=35,
            )
            .tables["t"]
            .collect()
        ]
    )
    assert (
        float(
            np.mean((values == np.datetime64("2020-01-01T00:00:00")) | (values == np.datetime64("2020-01-03T00:00:00")))
        )
        < 0.001
    )


@pytest.mark.parametrize("sampler_seed", [0, 1])
def test_numeric_xs_privacy_via_real_scorer(spark: SparkSession, tmp_path: Path, sampler_seed: int) -> None:
    """End-to-end via try_real_scorer: numeric@xs partial_copy_rate_excess is exactly 0."""
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    scorer = try_real_scorer(cache_dir=tmp_path, harness_hash="p1-4b", metrics_config_hash="p1-4b")
    assert scorer is not None
    real = load("numeric", "xs", seed=0, spark=spark, sample_seed=0)
    loaded = profile_tables(real, detect_pii=False)
    scores = scorer.score(
        loaded.model_dump(mode="json"),
        dataset="numeric",
        tier="xs",
        seed=sampler_seed,
        structure_seed=0,
        sample_seed=0,
    )
    checks = {check["name"]: check for check in scores.detail["privacy"]["checks"]}
    partial = checks["measurements.partial_copy_rate_excess"]

    assert scores.privacy_passed is True
    assert partial["passed"] is True
    assert partial["value"] == 0.0
