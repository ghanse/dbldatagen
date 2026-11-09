"""Regression coverage for exact profiled extrema leaking into samples."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pyspark.sql import SparkSession

from dbldatagen_profiler.engines.spark import profile_tables
from dbldatagen_profiler.model import LogicalType, NumericStats, TableProfile
from profiler_bench.catalog import load, materialize
from profiler_bench.fidelity import load_config, score_privacy
from profiler_bench.sampler import sample
from profiler_bench.sampler.core import _standard_normal_cdf

from .profiles import column, profile


def test_clipped_normal_continuous_avoids_exact_extrema(spark: SparkSession) -> None:
    """Skewed continuous columns must not pile mass on the exact profiled min/max."""
    stats = NumericStats(min=1.0, max=1000.0, mean=3.0, stddev=20.0)
    rows = (
        sample(
            profile(
                TableProfile(
                    name="t",
                    row_count=20_000,
                    columns=[column("x", LogicalType.FLOAT64, numeric=stats)],
                )
            ),
            spark,
            seed=11,
        )
        .tables["t"]
        .collect()
    )
    values = np.asarray([row.x for row in rows], dtype=float)
    share = float(np.mean((values == stats.min) | (values == stats.max)))

    assert share < 0.001


def test_clipped_normal_integer_endpoint_mass_matches_model(spark: SparkSession) -> None:
    """Integer endpoints stay reachable, but only at truncated-normal rounded tail mass."""
    stats = NumericStats(min=0, max=12, mean=4.0, stddev=2.0)
    rows = (
        sample(
            profile(
                TableProfile(
                    name="t",
                    row_count=40_000,
                    columns=[column("x", LogicalType.INT32, numeric=stats)],
                )
            ),
            spark,
            seed=13,
        )
        .tables["t"]
        .collect()
    )
    values = np.asarray([row.x for row in rows], dtype=int)
    low = float(stats.min)
    high = float(stats.max)
    open_low = np.nextafter(low, high)
    open_high = np.nextafter(high, low)
    mean = float(stats.mean)
    stddev = float(stats.stddev)
    fa = _standard_normal_cdf((open_low - mean) / stddev)
    fb = _standard_normal_cdf((open_high - mean) / stddev)
    # Continuity-corrected mass that rounds to each endpoint under the truncated normal.
    expected_min = (_standard_normal_cdf((0.5 - mean) / stddev) - fa) / (fb - fa)
    expected_max = (fb - _standard_normal_cdf((11.5 - mean) / stddev)) / (fb - fa)

    assert float(np.mean(values == 0)) == pytest.approx(expected_min, abs=0.01)
    assert float(np.mean(values == 12)) == pytest.approx(expected_max, abs=0.005)
    assert float(np.mean(values == 0)) < 0.05


@pytest.mark.parametrize("sampler_seed", [0, 1])
def test_numeric_xs_privacy_passes_partial_copy_guard(spark: SparkSession, sampler_seed: int) -> None:
    """End-to-end: baseline profile of catalog numeric@xs must pass the privacy guard."""
    structure_seed = 0
    sample_seed = 0
    config = load_config()
    real = load("numeric", "xs", seed=structure_seed, spark=spark, sample_seed=sample_seed)
    holdout = load(
        "numeric",
        "xs",
        seed=structure_seed,
        spark=spark,
        sample_seed=sample_seed + config.sampling.holdout_sample_offset,
    )
    manifest = materialize("numeric", "xs", structure_seed, sample_seed=sample_seed).manifest
    loaded = profile_tables(real, detect_pii=False)
    synth = sample(loaded, spark, seed=sampler_seed).tables
    score = score_privacy(
        real,
        synth,
        loaded,
        manifest,
        holdout=holdout,
        sample_seed=sample_seed,
        seed=sampler_seed,
    )
    checks = {check.name: check for check in score.checks}
    partial = checks["measurements.partial_copy_rate_excess"]

    assert math.isfinite(partial.value)
    assert partial.passed
    assert score.passed
