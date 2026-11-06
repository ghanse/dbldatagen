"""Seeded paired bootstrap confidence intervals.

Gates use two-sided percentile intervals at the policy confidence level (default 95%)
for each gate separately. There is no family-wise or simultaneous coverage guarantee
across gates or datasets.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np


StatisticName = Literal["mean", "median"]
BootstrapMethod = Literal["pooled", "stratified_dataset_mean"]


@dataclass(frozen=True)
class BootstrapCI:
    """One paired-bootstrap interval.

    Attributes:
        point: Statistic on the observed paired differences.
        low: Lower confidence bound.
        high: Upper confidence bound.
        n: Number of paired differences (pooled) or number of datasets (stratified).
        n_bootstrap: Bootstrap resample count.
        statistic: `mean` or `median`.
        seed: RNG seed.
        confidence: Nominal confidence level, e.g. 0.95.
        method: `pooled` or `stratified_dataset_mean`.
    """

    point: float
    low: float
    high: float
    n: int
    n_bootstrap: int
    statistic: StatisticName
    seed: int
    confidence: float
    method: BootstrapMethod = "pooled"

    def crosses_zero(self) -> bool:
        """Returns true when the interval contains both signs (or zero)."""
        return self.low < 0.0 < self.high or self.low == 0.0 or self.high == 0.0

    def to_dict(self) -> dict[str, float | int | str | bool]:
        """Returns a JSON-serializable mapping."""
        return {
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "n": self.n,
            "n_bootstrap": self.n_bootstrap,
            "statistic": self.statistic,
            "seed": self.seed,
            "confidence": self.confidence,
            "method": self.method,
            "crosses_zero": self.crosses_zero(),
        }


def paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    statistic: StatisticName = "mean",
    n_bootstrap: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Returns a paired bootstrap CI over `differences` (candidate - baseline effects).

    Args:
        differences: One paired difference per seed or sample.
        statistic: Aggregate applied to each resample.
        n_bootstrap: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.
        confidence: Confidence level in (0, 1).

    Returns:
        Point estimate and two-sided percentile confidence interval.

    Raises:
        ValueError: If `differences` is empty or `confidence` is invalid.
    """
    if not differences:
        raise ValueError("differences must be non-empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    values = np.asarray(list(differences), dtype=float)
    agg = _statistic(statistic)
    point = float(agg(values))
    rng = np.random.default_rng(seed)
    n = values.size
    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        draw = values[rng.integers(0, n, size=n)]
        samples[index] = agg(draw)
    return _percentile_ci(
        point=point,
        samples=samples,
        n=int(n),
        n_bootstrap=n_bootstrap,
        statistic=statistic,
        seed=seed,
        confidence=confidence,
        method="pooled",
    )


def stratified_dataset_mean_ci(
    per_dataset_diffs: Mapping[str, Sequence[float]],
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Returns a CI for the equally weighted mean of per-dataset means.

    The estimand is the mean across datasets of each dataset's mean paired-seed
    difference. The bootstrap resamples seeds within each dataset, averages within
    the dataset, then averages across datasets. Datasets are fixed strata and are
    not themselves resampled. A hierarchical bootstrap that also resamples datasets
    is a documented alternative and is not used here.
    """
    if not per_dataset_diffs:
        raise ValueError("per_dataset_diffs must be non-empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    datasets = sorted(per_dataset_diffs)
    arrays = {name: np.asarray(list(per_dataset_diffs[name]), dtype=float) for name in datasets}
    for name, values in arrays.items():
        if values.size == 0:
            raise ValueError(f"dataset {name!r} has no paired differences")
    point = float(np.mean([float(np.mean(arrays[name])) for name in datasets]))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        stratum_means = []
        for name in datasets:
            values = arrays[name]
            draw = values[rng.integers(0, values.size, size=values.size)]
            stratum_means.append(float(np.mean(draw)))
        samples[index] = float(np.mean(stratum_means))
    return _percentile_ci(
        point=point,
        samples=samples,
        n=len(datasets),
        n_bootstrap=n_bootstrap,
        statistic="mean",
        seed=seed,
        confidence=confidence,
        method="stratified_dataset_mean",
    )


def _percentile_ci(
    *,
    point: float,
    samples: np.ndarray,
    n: int,
    n_bootstrap: int,
    statistic: StatisticName,
    seed: int,
    confidence: float,
    method: BootstrapMethod,
) -> BootstrapCI:
    alpha = 1.0 - confidence
    low = float(np.quantile(samples, alpha / 2.0))
    high = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return BootstrapCI(
        point=point,
        low=low,
        high=high,
        n=n,
        n_bootstrap=n_bootstrap,
        statistic=statistic,
        seed=seed,
        confidence=confidence,
        method=method,
    )


def _statistic(name: StatisticName) -> Callable[[np.ndarray], float]:
    if name == "mean":
        return lambda values: float(np.mean(values))
    if name == "median":
        return lambda values: float(np.median(values))
    raise ValueError(f"unknown statistic: {name}")
