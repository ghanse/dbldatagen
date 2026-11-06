"""Checks that a sample was drawn from a declared distribution.

Every family gets a goodness-of-fit test at significance `ALPHA` against its declared parameters:
a one-sample Kolmogorov-Smirnov test for continuous families (normal, lognormal, normal mixture,
uniform, exponential, Pareto) and a chi-square test on binned counts for discrete ones (Poisson,
binomial, negative binomial, discrete uniform, categorical, the hour and weekday profile of
`seasonal`, and the count-occupancy profile of `zipf`). Bounds hold exactly, and means, variances,
and shares must also lie within `Z` standard errors, `Z` being the two-sided normal quantile of
`ALPHA`. `ALPHA` is small because one verification runs a few hundred tests and a frozen benchmark
must not fail by chance; at the `xs` tier it still rejects, for example, a uniform sample declared
normal with the same mean and variance.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from profiler_bench.catalog.manifest import Bound, Distribution, Family
from profiler_bench.catalog.stats import (
    binomial_logpmf,
    chi2_test,
    kolmogorov_pvalue,
    ks_statistic,
    negative_binomial_logpmf,
    normal_cdf,
    normal_ppf,
    poisson_logpmf,
)


ALPHA = 1e-5
Z = normal_ppf(1 - ALPHA / 2)
NUMBER, DATE, TIMESTAMP, RAW = "number", "date", "timestamp", "raw"
MICROS_PER_HOUR = 3_600_000_000
MICROS_PER_DAY = 24 * MICROS_PER_HOUR
EPOCH_WEEKDAY = 3
MAX_DISCRETE_SUPPORT = 1_000_000
ZIPF_OCCUPANCY_BINS = 30


def near(label: str, observed: float, expected: float, tolerance: float) -> list[str]:
    """Returns a failure message when `observed` is farther than `tolerance` from `expected`."""
    if abs(observed - expected) <= tolerance:
        return []
    return [f"{label} {observed:.6g}, expected {expected:.6g} ± {tolerance:.3g}"]


def share(label: str, observed: float, expected: float, n: int) -> list[str]:
    """Returns a failure message when a binomial share is more than `Z` standard errors off."""
    tolerance = 1e-9 + Z * math.sqrt(expected * (1 - expected) / max(n, 1))
    return near(label, observed, expected, tolerance)


def to_number(value: Bound, kind: str) -> float:
    """Returns an ISO date or timestamp parameter as days or microseconds since the epoch, or a number as is."""
    if kind == DATE:
        return float(np.datetime64(str(value), "D").astype(np.int64))
    if kind == TIMESTAMP:
        return float(np.datetime64(str(value).removesuffix("Z"), "us").astype(np.int64))
    return float(value)


def check_distribution(
    values: np.ndarray, dist: Distribution, kind: str = NUMBER, counts_input: bool = False
) -> list[str]:
    """Returns the ways `values` departs from `dist`, as failure messages.

    Args:
        values: Non-null values: numbers (days for dates, microseconds for timestamps) or raw
            values for strings and booleans.
        dist: Declared distribution.
        kind: `number`, `date`, `timestamp`, or `raw`; selects how ISO parameters are read.
        counts_input: Whether `values` are counts per category including zeros (as for foreign-key
            fan-out) rather than draws. Only changes the meaning of `zipf`.
    """
    if len(values) == 0:
        return ["no values to check"]
    if dist.family == Family.ZIPF:
        counts = values if counts_input else pd.Series(values).value_counts().to_numpy()
        return _zipf(np.asarray(counts, dtype=np.float64), dist.params["s"], dist.params["n_values"])
    if kind == RAW:
        if dist.family == Family.CONSTANT:
            return _constant_raw(values, dist.params["value"])
        if dist.family == Family.CATEGORICAL:
            return _categorical(values, dist.params)
        return [f"{dist.family.value} needs numeric values"]
    params = {
        k: to_number(v, kind) if k in ("low", "high", "start", "end", "value") else v for k, v in dist.params.items()
    }
    return _CHECKS[dist.family](np.asarray(values, dtype=np.float64), params)


def _ks(values: np.ndarray, cdf: Callable[[np.ndarray], np.ndarray]) -> list[str]:
    d = ks_statistic(values, cdf)
    p = kolmogorov_pvalue(d, len(values))
    return [] if p >= ALPHA else [f"KS D={d:.4g}, p={p:.2g} < alpha={ALPHA:g}"]


def _chi2(label: str, observed: np.ndarray, expected: np.ndarray) -> list[str]:
    statistic, df, p = chi2_test(observed, expected)
    return [] if p >= ALPHA else [f"{label} chi-square {statistic:.4g} on {df} df, p={p:.2g} < alpha={ALPHA:g}"]


def _discrete(values: np.ndarray, logpmf: Callable[[np.ndarray], np.ndarray], low: int) -> list[str]:
    """Chi-square test of integer `values >= low`; the last bin holds the whole upper tail."""
    ints = values.astype(np.int64)
    support = np.arange(low, int(ints.max()) + 1)
    probabilities = np.exp(logpmf(support - low))
    probabilities[-1] += max(0.0, 1.0 - probabilities.sum())
    observed = np.bincount(ints - low, minlength=len(support))
    return _chi2("goodness of fit", observed.astype(np.float64), probabilities * len(values))


def _mean_std(values: np.ndarray, mean: float, std: float) -> list[str]:
    n = len(values)
    return near("mean", float(values.mean()), mean, Z * std / math.sqrt(n) + 1e-9) + near(
        "std", float(values.std()), std, Z * std / math.sqrt(2 * n) + 1e-9
    )


def _bounds(values: np.ndarray, low: float | None, high: float | None) -> list[str]:
    failures = []
    if low is not None and values.min() < low:
        failures.append(f"min {values.min():.6g} < {low:.6g}")
    if high is not None and values.max() > high:
        failures.append(f"max {values.max():.6g} > {high:.6g}")
    return failures


def _integral(values: np.ndarray) -> list[str]:
    return [] if np.array_equal(values, np.round(values)) else ["non-integer values"]


def _normal(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    return _mean_std(v, p["mean"], p["std"]) + _ks(v, lambda x: normal_cdf(x, p["mean"], p["std"]))


def _lognormal(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    failures = _bounds(v, 1e-300, None)
    if failures:
        return failures
    logs = np.log(v)
    return _mean_std(logs, p["mu"], p["sigma"]) + _ks(logs, lambda x: normal_cdf(x, p["mu"], p["sigma"]))


def _mixture(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    components = p["components"]
    mean = sum(c["weight"] * c["mean"] for c in components)
    variance = sum(c["weight"] * (c["std"] ** 2 + c["mean"] ** 2) for c in components) - mean**2
    centered = v - v.mean()
    fourth = float((centered**4).mean())
    n = len(v)

    def cdf(x: np.ndarray) -> np.ndarray:
        return np.asarray(sum(c["weight"] * normal_cdf(x, c["mean"], c["std"]) for c in components))

    return (
        near("mean", float(v.mean()), mean, Z * math.sqrt(variance / n))
        + near("variance", float(centered.var()), variance, Z * math.sqrt(max(fourth - variance**2, 1e-12) / n))
        + _ks(v, cdf)
    )


def _uniform(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    low, high = p["low"], p["high"]
    tolerance = Z * (high - low) / math.sqrt(12 * len(v)) + 1e-9
    failures = _bounds(v, low, high) + near("mean", float(v.mean()), (low + high) / 2, tolerance)
    return failures + _ks(v, lambda x: np.clip((x - low) / (high - low), 0.0, 1.0))


def _discrete_uniform(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    low, high = int(p["low"]), int(p["high"])
    failures = _integral(v) + _bounds(v, low, high)
    if failures:
        return failures
    width = high - low + 1
    tolerance = Z * math.sqrt((width**2 - 1) / 12 / len(v)) + 1e-9
    failures = near("mean", float(v.mean()), (low + high) / 2, tolerance)
    if width > MAX_DISCRETE_SUPPORT:
        return failures + _ks(v, lambda x: np.clip((np.floor(x) - low + 1) / width, 0.0, 1.0))
    observed = np.bincount(v.astype(np.int64) - low, minlength=width).astype(np.float64)
    return failures + _chi2("goodness of fit", observed, np.full(width, len(v) / width))


def _exponential(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    scale = p["scale"]
    failures = _bounds(v, 0.0, None) + near("mean", float(v.mean()), scale, Z * scale / math.sqrt(len(v)))
    return failures + _ks(v, lambda x: 1.0 - np.exp(-np.maximum(x, 0.0) / scale))


def _pareto(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    alpha, scale = p["alpha"], p["scale"]
    failures = _bounds(v, scale * (1 - 1e-9), None)
    return failures + _ks(v, lambda x: 1.0 - (scale / np.maximum(x, scale)) ** alpha)


def _poisson(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    lam, shift, n = p["lam"], int(p.get("shift", 0)), len(v)
    failures = _integral(v) + _bounds(v, shift, None)
    if failures:
        return failures
    return (
        near("mean", float(v.mean()), lam + shift, Z * math.sqrt(lam / n))
        + near("variance", float(v.var()), lam, Z * math.sqrt((lam + 2 * lam**2) / n))
        + _discrete(v, lambda k: poisson_logpmf(k, lam), shift)
    )


def _binomial(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    trials, prob = int(p["n"]), p["p"]
    failures = _integral(v) + _bounds(v, 0, trials)
    if failures:
        return failures
    sd = math.sqrt(trials * prob * (1 - prob))
    return near("mean", float(v.mean()), trials * prob, Z * sd / math.sqrt(len(v))) + _discrete(
        v, lambda k: binomial_logpmf(k, trials, prob), 0
    )


def _negative_binomial(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    size, prob = p["n"], p["p"]
    failures = _integral(v) + _bounds(v, 0, None)
    if failures:
        return failures
    mean, variance = size * (1 - prob) / prob, size * (1 - prob) / prob**2
    return near("mean", float(v.mean()), mean, Z * math.sqrt(variance / len(v))) + _discrete(
        v, lambda k: negative_binomial_logpmf(k, size, prob), 0
    )


def _constant(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    others = int((v != p["value"]).sum())
    return [f"{others} values differ from the constant"] if others else []


def _constant_raw(values: np.ndarray, value: object) -> list[str]:
    others = int((pd.Series(values) != value).sum())
    return [f"{others} values differ from the constant"] if others else []


def _categorical(values: np.ndarray, p: dict[str, Any]) -> list[str]:
    counts = pd.Series(values).value_counts()
    unknown = sorted(map(str, set(counts.index) - set(p["values"])))
    failures = [f"unexpected values {unknown[:5]}"] if unknown else []
    observed = np.asarray([float(counts.get(v, 0)) for v in p["values"]])
    expected = np.asarray(p["probabilities"], dtype=np.float64) * len(values)
    for value, o, e in zip(p["values"], observed, expected, strict=True):
        failures += share(f"share of {value!r}", o / len(values), e / len(values), len(values))
    return failures + _chi2("goodness of fit", observed, expected)


def _sequence(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    expected = p["start"] + p["step"] * np.arange(len(v))
    return [] if np.array_equal(np.sort(v), expected) else [f"values are not the sequence {p['start']} + {p['step']}*i"]


def _seasonal(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    failures = _bounds(v, p["start"], p["end"])
    whole = v.astype(np.int64)
    for label, index, weights in (
        ("hour", (whole // MICROS_PER_HOUR) % 24, p["hour_weights"]),
        ("weekday", (whole // MICROS_PER_DAY + EPOCH_WEEKDAY) % 7, p["weekday_weights"]),
    ):
        observed = np.bincount(index, minlength=len(weights)).astype(np.float64)
        expected = np.asarray(weights, dtype=np.float64) / sum(weights) * len(v)
        failures += _chi2(f"{label} profile", observed, expected)
    return failures


def _zipf(counts: np.ndarray, s: float, n_values: int) -> list[str]:
    """Checks counts per category against a Zipf law whose rank order is unknown.

    The top three shares are compared with the top three probabilities, and the occupancy profile
    (how many categories hold 0, 1, 2, ... draws) with its expectation under Poisson counts.
    """
    if len(counts) > n_values:
        return [f"{len(counts)} distinct values > n_values {n_values}"]
    total = counts.sum()
    p = np.arange(1, n_values + 1, dtype=np.float64) ** -s
    p /= p.sum()
    padded = np.pad(counts, (0, n_values - len(counts)))
    observed_shares = np.sort(padded)[::-1] / total
    failures = [
        m for k in range(min(3, n_values)) for m in share(f"rank {k + 1} share", observed_shares[k], p[k], int(total))
    ]
    lam = total * p
    j = np.arange(ZIPF_OCCUPANCY_BINS, dtype=np.float64)
    occupancy = np.exp(j[None, :] * np.log(lam)[:, None] - lam[:, None] - np.asarray([math.lgamma(k + 1) for k in j]))
    expected = np.append(occupancy.sum(axis=0), n_values - occupancy.sum())
    observed = np.bincount(np.minimum(padded.astype(np.int64), ZIPF_OCCUPANCY_BINS), minlength=ZIPF_OCCUPANCY_BINS + 1)
    return failures + _chi2("occupancy profile", observed.astype(np.float64), expected)


_CHECKS: dict[Family, Callable[[np.ndarray, dict[str, Any]], list[str]]] = {
    Family.CONSTANT: _constant,
    Family.NORMAL: _normal,
    Family.LOGNORMAL: _lognormal,
    Family.MIXTURE: _mixture,
    Family.UNIFORM: _uniform,
    Family.DISCRETE_UNIFORM: _discrete_uniform,
    Family.EXPONENTIAL: _exponential,
    Family.PARETO: _pareto,
    Family.POISSON: _poisson,
    Family.BINOMIAL: _binomial,
    Family.NEGATIVE_BINOMIAL: _negative_binomial,
    Family.CATEGORICAL: _categorical,
    Family.SEQUENCE: _sequence,
    Family.SEASONAL: _seasonal,
}
