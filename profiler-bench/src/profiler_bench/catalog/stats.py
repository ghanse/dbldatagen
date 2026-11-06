"""Distribution functions and goodness-of-fit tests used to verify manifests, built on numpy and `math`.

The p-values are standard: the Kolmogorov distribution with Stephens' small-sample correction for
one-sample KS tests, and the regularized upper incomplete gamma function for chi-square tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


_SQRT2 = math.sqrt(2.0)
_erf = np.vectorize(math.erf, otypes=[np.float64])
_lgamma = np.vectorize(math.lgamma, otypes=[np.float64])
MIN_EXPECTED = 5.0


def normal_cdf(x: np.ndarray, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    """Returns the normal CDF at `x`."""
    return np.asarray(0.5 * (1.0 + _erf((np.asarray(x, dtype=np.float64) - mean) / (std * _SQRT2))))


def normal_ppf(q: float) -> float:
    """Returns the standard normal quantile of `q`, found by bisection to double precision."""
    low, high = -40.0, 40.0
    for _ in range(200):
        mid = (low + high) / 2
        if 0.5 * math.erfc(-mid / _SQRT2) < q:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def kolmogorov_pvalue(d: float, n: int) -> float:
    """Returns the p-value of a one-sample KS statistic `d` for sample size `n`."""
    if n == 0:
        return 1.0
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    if lam < 0.2:
        return 1.0
    total = float(sum((-1.0) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101)))
    return min(1.0, max(0.0, 2 * total))


def ks_statistic(values: np.ndarray, cdf: Callable[[np.ndarray], np.ndarray]) -> float:
    """Returns the KS distance between the empirical CDF of `values` and a continuous `cdf`.

    Ties are handled by comparing the empirical CDF just before and at each distinct value, so the
    statistic is also valid for data recorded on a grid such as dates.
    """
    unique, counts = np.unique(np.asarray(values, dtype=np.float64), return_counts=True)
    upper = np.cumsum(counts) / len(values)
    lower = upper - counts / len(values)
    theoretical = cdf(unique)
    return float(max(np.max(np.abs(upper - theoretical)), np.max(np.abs(theoretical - lower))))


def chi2_sf(x: float, df: int) -> float:
    """Returns the chi-square survival function `P(X > x)` with `df` degrees of freedom."""
    return _gammq(df / 2.0, x / 2.0) if x > 0 else 1.0


def _gammq(a: float, x: float) -> float:
    if x < a + 1:
        term = total = 1.0 / a
        ap = a
        for _ in range(10_000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - math.lgamma(a)) * h)


def chi2_test(observed: np.ndarray, expected: np.ndarray) -> tuple[float, int, float]:
    """Returns Pearson's chi-square statistic, degrees of freedom, and p-value for binned counts.

    Adjacent bins are merged from the left until every bin expects at least five counts, and the
    last bin absorbs any remainder.

    Args:
        observed: Observed count per bin.
        expected: Expected count per bin, in the same order, summing to the observed total.
    """
    merged_observed, merged_expected = [], []
    o_acc = e_acc = 0.0
    for o, e in zip(observed, expected, strict=True):
        o_acc, e_acc = o_acc + o, e_acc + e
        if e_acc >= MIN_EXPECTED:
            merged_observed.append(o_acc)
            merged_expected.append(e_acc)
            o_acc = e_acc = 0.0
    if merged_expected:
        merged_observed[-1] += o_acc
        merged_expected[-1] += e_acc
    if len(merged_expected) < 2:
        return 0.0, 0, 1.0
    o_arr, e_arr = np.asarray(merged_observed), np.asarray(merged_expected)
    statistic = float(((o_arr - e_arr) ** 2 / e_arr).sum())
    df = len(e_arr) - 1
    return statistic, df, chi2_sf(statistic, df)


def poisson_logpmf(k: np.ndarray, lam: float) -> np.ndarray:
    """Returns the Poisson log-pmf at non-negative integers `k`."""
    k = np.asarray(k, dtype=np.float64)
    return np.asarray(k * math.log(lam) - lam - _lgamma(k + 1))


def binomial_logpmf(k: np.ndarray, n: int, p: float) -> np.ndarray:
    """Returns the binomial log-pmf at integers `0 <= k <= n`."""
    k = np.asarray(k, dtype=np.float64)
    return np.asarray(
        math.lgamma(n + 1) - _lgamma(k + 1) - _lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)
    )


def negative_binomial_logpmf(k: np.ndarray, n: float, p: float) -> np.ndarray:
    """Returns the log-pmf of the number of failures before `n` successes with success probability `p`."""
    k = np.asarray(k, dtype=np.float64)
    return np.asarray(_lgamma(k + n) - math.lgamma(n) - _lgamma(k + 1) + n * math.log(p) + k * math.log1p(-p))
