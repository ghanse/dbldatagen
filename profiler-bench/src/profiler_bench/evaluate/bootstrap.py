"""Seeded paired bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np


StatisticName = Literal["mean", "median"]


@dataclass(frozen=True)
class BootstrapCI:
    """One paired-bootstrap interval.

    Attributes:
        point: Statistic on the observed paired differences.
        low: Lower confidence bound.
        high: Upper confidence bound.
        n: Number of paired differences.
        n_bootstrap: Bootstrap resample count.
        statistic: `mean` or `median`.
        seed: RNG seed.
        confidence: Nominal confidence level, e.g. 0.95.
    """

    point: float
    low: float
    high: float
    n: int
    n_bootstrap: int
    statistic: StatisticName
    seed: int
    confidence: float

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
        Point estimate and percentile confidence interval.

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
    alpha = 1.0 - confidence
    low = float(np.quantile(samples, alpha / 2.0))
    high = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return BootstrapCI(
        point=point,
        low=low,
        high=high,
        n=int(n),
        n_bootstrap=n_bootstrap,
        statistic=statistic,
        seed=seed,
        confidence=confidence,
    )


def _statistic(name: StatisticName) -> Callable[[np.ndarray], float]:
    if name == "mean":
        return lambda values: float(np.mean(values))
    if name == "median":
        return lambda values: float(np.median(values))
    raise ValueError(f"unknown statistic: {name}")
