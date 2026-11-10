"""Regression coverage for exact profiled extrema leaking into samples."""

from __future__ import annotations

import ast
import inspect
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
from pyspark.sql import SparkSession
from scipy import stats

from dbldatagen_profiler.engines.spark import profile_tables
from dbldatagen_profiler.model import (
    ColumnProfile,
    DistinctCount,
    Histogram,
    LogicalType,
    NumericStats,
    Quantile,
    TableProfile,
    TemporalStats,
)
from profiler_bench.catalog import load
from profiler_bench.runner.scorer import try_real_scorer
from profiler_bench.sampler import core as sampler_core
from profiler_bench.sampler import sample
from profiler_bench.sampler.core import (
    _calibrate_truncated_parameters,
    _calibrate_truncated_parameters_exact,
    _exact_number,
    _exact_truncated_normal,
    _standard_normal_cdf,
    _standardized_bounds,
    _truncated_moments,
    _truncated_moments_exact,
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
    """Integer endpoints stay reachable, but only at half-bin truncated-normal tail mass."""
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
    # Continuous model on [min - 0.5, max + 0.5]; k gets P(k - 0.5 <= X < k + 0.5).
    fa = _standard_normal_cdf((-0.5 - mean) / stddev)
    fb = _standard_normal_cdf((12.5 - mean) / stddev)
    expected_min = (_standard_normal_cdf((0.5 - mean) / stddev) - fa) / (fb - fa)
    expected_max = (fb - _standard_normal_cdf((11.5 - mean) / stddev)) / (fb - fa)

    assert float(np.mean(values == 0)) == pytest.approx(expected_min, abs=0.01)
    assert float(np.mean(values == 12)) == pytest.approx(expected_max, abs=0.005)
    assert float(np.mean(values == 0)) < 0.05


def test_poisson_like_integer_max_mass_matches_half_bin_tail(spark: SparkSession) -> None:
    """Poisson-like ints (λ=4, observed max=14) keep exact-max share near modeled half-bin mass."""
    stats = NumericStats(min=0, max=14, mean=4.0, stddev=2.0)
    rows = (
        sample(
            profile(TableProfile(name="t", row_count=80_000, columns=[column("x", LogicalType.INT32, numeric=stats)])),
            spark,
            seed=19,
        )
        .tables["t"]
        .collect()
    )
    values = np.asarray([row.x for row in rows], dtype=int)
    mean = float(stats.mean)
    stddev = float(stats.stddev)
    fa = _standard_normal_cdf((-0.5 - mean) / stddev)
    fb = _standard_normal_cdf((14.5 - mean) / stddev)
    modeled_max = (fb - _standard_normal_cdf((13.5 - mean) / stddev)) / (fb - fa)
    observed_max = float(np.mean(values == 14))

    assert modeled_max < 0.001
    assert observed_max < 0.001
    assert observed_max == pytest.approx(modeled_max, rel=1.0, abs=1e-6)


def test_integer_distinct_count_does_not_inflate_endpoint_mass(spark: SparkSession) -> None:
    """Cardinality pooling must not promote a rare integer max to ~1/distinct_count mass."""
    stats = NumericStats(min=1, max=20, mean=8.6059, stddev=2.9086476605325218)
    rows = (
        sample(
            profile(
                TableProfile(
                    name="t",
                    row_count=20_000,
                    columns=[
                        column(
                            "x",
                            LogicalType.INT32,
                            numeric=stats,
                            distinct_count=DistinctCount(value=20, exact=True),
                        )
                    ],
                )
            ),
            spark,
            seed=2,
        )
        .tables["t"]
        .collect()
    )
    values = np.asarray([row.x for row in rows], dtype=int)
    mean = float(stats.mean)
    stddev = float(stats.stddev)
    fa = _standard_normal_cdf((-0.5 - mean) / stddev)
    fb = _standard_normal_cdf((20.5 - mean) / stddev)
    modeled_max = (fb - _standard_normal_cdf((19.5 - mean) / stddev)) / (fb - fa)
    observed_max = float(np.mean(values == 20))

    assert observed_max < 0.001
    assert observed_max == pytest.approx(modeled_max, rel=1.0, abs=1e-4)
    assert observed_max < 0.01  # well below the 1/20 ≈ 0.05 pool-slot inflation


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


def _exact_stats_column(
    kind: LogicalType,
    low: Decimal,
    high: Decimal,
    mean: Decimal,
    stddev: Decimal,
    *,
    precision: int | None = None,
    scale: int | None = None,
) -> ColumnProfile:
    """NumericStats with lossless Decimal mean/stddev (model_construct bypasses float coercion)."""
    stats = NumericStats.model_construct(
        min=format(low, "f"),
        max=format(high, "f"),
        mean=mean,
        stddev=stddev,
    )
    name = "amount" if kind == LogicalType.DECIMAL else "x"
    source = f"decimal({precision},{scale})" if kind == LogicalType.DECIMAL else kind.value
    return ColumnProfile.model_construct(
        name=name,
        logical_type=kind,
        source_type=source,
        nullable=False,
        null_fraction=0.0,
        precision=precision,
        scale=scale,
        numeric=stats,
    )


@pytest.mark.parametrize(
    ("label", "center", "kind", "precision", "scale"),
    [
        ("0", Decimal(0), LogicalType.INT64, None, None),
        ("1e10", Decimal("10000000000"), LogicalType.INT64, None, None),
        ("2^62", Decimal(2**62), LogicalType.INT64, None, None),
        ("1e28", Decimal("9999999999999999999999998500.0000000000"), LogicalType.DECIMAL, 38, 10),
    ],
)
@pytest.mark.parametrize(
    ("asymmetry", "fraction"),
    [("mid", Decimal("0.5")), ("low10", Decimal("0.1")), ("low90", Decimal("0.9"))],
)
def test_exact_truncated_normal_property_across_magnitudes_and_asymmetry(
    label: str,
    center: Decimal,
    kind: LogicalType,
    precision: int | None,
    scale: int | None,
    asymmetry: str,
    fraction: Decimal,
) -> None:
    """Exact-type truncated normal keeps location across magnitudes and asymmetric means."""
    del label, asymmetry
    range_width = Decimal(1000)
    stddev = range_width / Decimal(10)
    low = center - range_width / Decimal(2)
    high = center + range_width / Decimal(2)
    mean = low + fraction * range_width
    with localcontext() as ctx:
        ctx.prec = 50
        if kind != LogicalType.DECIMAL:
            low, high, mean, stddev = (value.to_integral_value() for value in (low, high, mean, stddev))
        else:
            quantum = Decimal("0.0000000001")
            low, high, mean, stddev = (value.quantize(quantum) for value in (low, high, mean, stddev))
            assert float(low) == float(high) == float(mean)
    column_profile = _exact_stats_column(kind, low, high, mean, stddev, precision=precision, scale=scale)
    stats = column_profile.numeric
    assert stats is not None
    assert isinstance(stats.mean, Decimal)
    assert isinstance(stats.stddev, Decimal)

    count = 8_000
    uniforms = np.linspace(np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0), count)
    values = _exact_truncated_normal(
        uniforms,
        stats.mean,
        stats.stddev,
        _exact_number(stats.min, column_profile),
        _exact_number(stats.max, column_profile),
        column_profile,
    )
    with localcontext() as ctx:
        ctx.prec = 50
        quantum = Decimal(1) if kind != LogicalType.DECIMAL else Decimal("0.0000000001")
        half = quantum / Decimal(2)
        cont_low, cont_high = low - half, high + half
        mu, sigma = _calibrate_truncated_parameters_exact(mean, stddev, cont_low, cont_high)
        expected_mean, expected_std = _truncated_moments_exact(mu, sigma, cont_low, cont_high)
        offsets = [value - expected_mean for value in values]
        se = expected_std / Decimal(str(math.sqrt(count)))
        mean_error = abs(sum(offsets, Decimal(0)) / Decimal(count))
        alpha, beta = _standardized_bounds(cont_low, cont_high, mu, sigma)
        low_z, high_z = _standardized_bounds(low + half, high - half, mu, sigma)
        span = _standard_normal_cdf(beta) - _standard_normal_cdf(alpha)
        modeled_low = (_standard_normal_cdf(low_z) - _standard_normal_cdf(alpha)) / span
        modeled_high = (_standard_normal_cdf(beta) - _standard_normal_cdf(high_z)) / span

    assert all(low <= value <= high for value in values)
    if kind == LogicalType.DECIMAL:
        assert all(isinstance(value, Decimal) and value.as_tuple().exponent == -10 for value in values)
    else:
        assert all(isinstance(value, int) for value in values)
    observed_low = sum(value == low for value in values) / count
    observed_high = sum(value == high for value in values) / count
    assert observed_low == pytest.approx(modeled_low, rel=0.2, abs=0.002)
    assert observed_high == pytest.approx(modeled_high, rel=0.2, abs=0.002)
    assert mean_error <= Decimal(3) * se
    sampled_std = float(np.std([float(offset) for offset in offsets], ddof=1))
    assert sampled_std == pytest.approx(float(expected_std), rel=0.15)


def test_exact_path_rejects_float_collapse_for_asymmetric_decimal_means() -> None:
    """At ~1e28, asymmetric Decimal means stay distinct even though float(mean) collides."""
    range_width = Decimal(1000)
    stddev = range_width / Decimal(10)
    center = Decimal("9999999999999999999999998500.0000000000")
    with localcontext() as ctx:
        ctx.prec = 50
        quantum = Decimal("0.0000000001")
        low = (center - range_width / Decimal(2)).quantize(quantum)
        high = (center + range_width / Decimal(2)).quantize(quantum)
        left = (low + Decimal("0.1") * range_width).quantize(quantum)
        right = (low + Decimal("0.9") * range_width).quantize(quantum)
    assert float(left) == float(right) == float(low) == float(high)

    column_profile = _exact_stats_column(LogicalType.DECIMAL, low, high, left, stddev, precision=38, scale=10)
    uniforms = np.linspace(np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0), 4_000)
    left_vals = _exact_truncated_normal(uniforms, left, stddev, low, high, column_profile)
    right_vals = _exact_truncated_normal(uniforms, right, stddev, low, high, column_profile)
    with localcontext() as ctx:
        ctx.prec = 50
        left_mean = sum(left_vals, Decimal(0)) / Decimal(len(left_vals))
        right_mean = sum(right_vals, Decimal(0)) / Decimal(len(right_vals))
        assert abs(left_mean - left) < abs(left_mean - right)
        assert abs(right_mean - right) < abs(right_mean - left)
        assert abs(left_mean - right_mean) > Decimal(100)


def test_exact_truncated_path_source_has_no_float_exact_quantities() -> None:
    """Static guard: exact-path helpers must not float()-cast min/max/mean/mid/offsets."""
    allowed_float_args = {"uniform", "uniforms", "value"}  # open-unit ranks / loop vars only

    for name in (
        "_exact_location",
        "_exact_scale",
        "_exact_truncated_normal",
        "_calibrate_truncated_parameters_exact",
        "_standardized_bounds",
        "_truncated_moments_exact",
        "_discrete_truncated_ppf",
    ):
        source = inspect.getsource(getattr(sampler_core, name))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "float":
                continue
            assert len(node.args) == 1
            arg = node.args[0]
            if isinstance(arg, ast.Name):
                assert arg.id in allowed_float_args or arg.id == "uniform", f"{name} float-casts exact name {arg.id!r}"
                continue
            # Only Decimal divisions for standardized a/b are allowed as float(...).
            assert isinstance(arg, ast.BinOp) and isinstance(
                arg.op, ast.Div
            ), f"{name} has a non-division float() argument: {ast.dump(arg)}"


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


@pytest.mark.slow
@pytest.mark.parametrize("sampler_seed", [0, 1, 2, 3, 4])
def test_nested_s_wide_privacy_via_real_scorer(spark: SparkSession, tmp_path: Path, sampler_seed: int) -> None:
    """End-to-end via try_real_scorer: nested@s wide.partial_copy_rate_excess passes for seeds 0-4."""
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    scorer = try_real_scorer(cache_dir=tmp_path, harness_hash="p1-4c", metrics_config_hash="p1-4c")
    assert scorer is not None
    real = load("nested", "s", seed=0, spark=spark, sample_seed=0)
    loaded = profile_tables(real, detect_pii=False)
    scores = scorer.score(
        loaded.model_dump(mode="json"),
        dataset="nested",
        tier="s",
        seed=sampler_seed,
        structure_seed=0,
        sample_seed=0,
    )
    checks = {check["name"]: check for check in scores.detail["privacy"]["checks"]}
    partial = checks["wide.partial_copy_rate_excess"]

    assert scores.privacy_passed is True
    assert partial["passed"] is True
    assert float(partial["value"]) < 0.015
