import math
from collections.abc import Callable

import numpy as np
import pytest

from profiler_bench.catalog.distributions import ALPHA, Z, check_distribution
from profiler_bench.catalog.manifest import Distribution, Family
from profiler_bench.catalog.stats import (
    binomial_logpmf,
    chi2_sf,
    chi2_test,
    kolmogorov_pvalue,
    ks_statistic,
    negative_binomial_logpmf,
    normal_cdf,
    normal_ppf,
    poisson_logpmf,
)


@pytest.mark.parametrize(
    ("x", "df", "p"), [(3.841459, 1, 0.05), (18.307038, 10, 0.05), (0.0, 3, 1.0), (100.0, 80, 0.06457)]
)
def test_chi2_sf_matches_reference_values(x: float, df: int, p: float) -> None:
    assert chi2_sf(x, df) == pytest.approx(p, abs=5e-5)


def test_normal_functions_match_reference_values() -> None:
    assert normal_ppf(0.975) == pytest.approx(1.959964, abs=1e-6)
    assert normal_cdf(np.asarray([0.0, 1.0]))[1] == pytest.approx(0.841345, abs=1e-6)
    assert pytest.approx(4.417, abs=1e-3) == Z
    assert ALPHA == 1e-5


def test_kolmogorov_pvalue_matches_the_asymptotic_critical_value() -> None:
    assert kolmogorov_pvalue(1.358 / math.sqrt(10**6), 10**6) == pytest.approx(0.05, abs=1e-3)
    assert kolmogorov_pvalue(0.0, 100) == 1.0


def test_log_pmfs_match_closed_forms() -> None:
    assert np.exp(poisson_logpmf(np.arange(3), 2.0)) == pytest.approx(
        [math.exp(-2), 2 * math.exp(-2), 2 * math.exp(-2)]
    )
    assert np.exp(binomial_logpmf(np.arange(3), 2, 0.5)) == pytest.approx([0.25, 0.5, 0.25])
    assert np.exp(negative_binomial_logpmf(np.arange(2), 2, 0.3)) == pytest.approx([0.09, 0.126])


def test_chi2_test_pools_sparse_bins() -> None:
    statistic, df, p = chi2_test(np.asarray([50.0, 1.0, 1.0, 48.0]), np.asarray([50.0, 1.0, 1.0, 48.0]))
    assert (statistic, df, p) == (0.0, 1, 1.0)


def test_ks_statistic_on_a_grid_is_bounded_by_the_grid_mass() -> None:
    values = (np.repeat(np.arange(1000.0), 10) + 0.5) / 1000
    assert ks_statistic(values, lambda x: np.clip(x, 0, 1)) <= 0.001 + 1e-12


@pytest.mark.parametrize(
    ("family", "params", "draw"),
    [
        (Family.NORMAL, {"mean": 1.0, "std": 2.0}, lambda r: r.normal(1, 2, 20_000)),
        (Family.PARETO, {"alpha": 1.5, "scale": 2.0}, lambda r: (r.pareto(1.5, 20_000) + 1) * 2),
        (Family.NEGATIVE_BINOMIAL, {"n": 2, "p": 0.3}, lambda r: r.negative_binomial(2, 0.3, 20_000)),
        (Family.BINOMIAL, {"n": 20, "p": 0.15}, lambda r: r.binomial(20, 0.15, 20_000)),
        (Family.DISCRETE_UNIFORM, {"low": -3, "high": 4}, lambda r: r.integers(-3, 5, 20_000)),
    ],
)
def test_check_distribution_accepts_true_samples(
    family: Family, params: dict[str, float], draw: Callable[[np.random.Generator], np.ndarray]
) -> None:
    values = draw(np.random.default_rng(1))
    assert check_distribution(np.asarray(values, dtype=np.float64), Distribution(family=family, params=params)) == []
