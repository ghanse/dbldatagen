"""Numeric-heavy table: parametric marginals, a Gaussian copula, and a nonlinear dependency."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa

from profiler_bench.catalog import gen
from profiler_bench.catalog.gen import Rng
from profiler_bench.catalog.manifest import (
    Column,
    ComparisonCheck,
    Dependency,
    Distribution,
    EnumerationCheck,
    Family,
    RangeCheck,
    Table,
)
from profiler_bench.catalog.registry import Build, Dataset


NORMAL = {"mean": 50.0, "std": 10.0}
NORMAL_NULL_RATE = 0.05
LOGNORMAL = {"mu": 3.0, "sigma": 0.75}
MIXTURE = [{"weight": 0.6, "mean": -2.0, "std": 0.5}, {"weight": 0.4, "mean": 3.0, "std": 1.0}]
PARETO = {"alpha": 1.5, "scale": 1.0}
POISSON = {"lam": 4.0}
NEGATIVE_BINOMIAL = {"n": 2, "p": 0.3}
NEGATIVE_BINOMIAL_NULL_RATE = 0.02
RATINGS = [1, 2, 3, 4, 5]
RATING_PROBABILITIES = [0.05, 0.1, 0.2, 0.4, 0.25]
COPULA = np.array([[1.0, 0.7, -0.4], [0.7, 1.0, -0.2], [-0.4, -0.2, 1.0]])
COPULA_X = {"mean": 10.0, "std": 2.0}
LINEAR_RHO = 0.8
NONLINEAR_NOISE = 0.5
NONLINEAR_RANGE = 3.0
GAP_SCALE = 5.0


def spearman_of_gaussian(rho: float) -> float:
    """Returns the Spearman correlation of a bivariate Gaussian copula with correlation `rho`."""
    return 6 / math.pi * math.asin(rho / 2)


def nonlinear_eta(bins: int = 50) -> float:
    """Returns the population correlation ratio of `nl_y` given `nl_x` cut into `bins` quantile bins.

    `nl_x` is uniform, so quantile bins are equal-width; the between-bin variance of `nl_x ** 2` is
    exact, and the total variance is `Var(nl_x ** 2) + noise**2`.
    """
    edges = np.linspace(-NONLINEAR_RANGE, NONLINEAR_RANGE, bins + 1)
    bin_means = (edges[1:] ** 3 - edges[:-1] ** 3) / (3 * (edges[1:] - edges[:-1]))
    between = float(((bin_means - NONLINEAR_RANGE**2 / 3) ** 2).mean())
    total = 4 * NONLINEAR_RANGE**4 / 45 + NONLINEAR_NOISE**2
    return math.sqrt(between / total)


def sizes(n: int) -> dict[str, float]:
    """Returns the row count of each table for base row count `n`."""
    return {"measurements": n}


def build(n: int, rng: Rng) -> Build:
    """Generates the `measurements` table with `n` rows."""
    mixture = rng("mixture")
    component = mixture.random(n) < MIXTURE[0]["weight"]
    bimodal = np.where(
        component,
        mixture.normal(MIXTURE[0]["mean"], MIXTURE[0]["std"], n),
        mixture.normal(MIXTURE[1]["mean"], MIXTURE[1]["std"], n),
    )
    z = rng("copula").multivariate_normal(np.zeros(3), COPULA, n, method="cholesky")
    linear_a = rng("linear_a").standard_normal(n)
    nl_x = rng("nl_x").uniform(-NONLINEAR_RANGE, NONLINEAR_RANGE, n)
    range_low = rng("range_low").uniform(0, 100, n)
    table = pa.table(
        {
            "id": pa.array(np.arange(1, n + 1), pa.int64()),
            "normal": pa.array(
                rng("normal").normal(NORMAL["mean"], NORMAL["std"], n),
                pa.float64(),
                mask=gen.null_mask(rng("normal.nulls"), n, NORMAL_NULL_RATE),
            ),
            "lognormal": pa.array(rng("lognormal").lognormal(LOGNORMAL["mu"], LOGNORMAL["sigma"], n), pa.float64()),
            "bimodal": pa.array(bimodal, pa.float64()),
            "pareto": pa.array((rng("pareto").pareto(PARETO["alpha"], n) + 1) * PARETO["scale"], pa.float64()),
            "count_poisson": pa.array(rng("count_poisson").poisson(POISSON["lam"], n), pa.int32()),
            "count_negbin": pa.array(
                rng("count_negbin").negative_binomial(NEGATIVE_BINOMIAL["n"], NEGATIVE_BINOMIAL["p"], n),
                pa.int64(),
                mask=gen.null_mask(rng("count_negbin.nulls"), n, NEGATIVE_BINOMIAL_NULL_RATE),
            ),
            "rating": pa.array(np.asarray(RATINGS)[gen.choice(rng("rating"), n, RATING_PROBABILITIES)], pa.int8()),
            "ratio": pa.array(rng("ratio").uniform(0, 1, n), pa.float32()),
            "copula_x": pa.array(COPULA_X["mean"] + COPULA_X["std"] * z[:, 0], pa.float64()),
            "copula_y": pa.array(np.exp(z[:, 1]), pa.float64()),
            "copula_z": pa.array(z[:, 2] ** 3, pa.float64()),
            "linear_a": pa.array(linear_a, pa.float64()),
            "linear_b": pa.array(
                LINEAR_RHO * linear_a + math.sqrt(1 - LINEAR_RHO**2) * rng("linear_b").standard_normal(n), pa.float64()
            ),
            "nl_x": pa.array(nl_x, pa.float64()),
            "nl_y": pa.array(nl_x**2 + rng("nl_y").normal(0, NONLINEAR_NOISE, n), pa.float64()),
            "range_low": pa.array(range_low, pa.float64()),
            "range_high": pa.array(range_low + rng("range_high").exponential(GAP_SCALE, n), pa.float64()),
        }
    )
    return Build({"measurements": table}, [_truth(n)], [])


def _truth(n: int) -> Table:
    def dist(family: Family, **params: object) -> Distribution:
        return Distribution(family=family, params=params)

    def copula(a: str, b: str, i: int, j: int) -> Dependency:
        rho = float(COPULA[i, j])
        return Dependency(
            columns=(a, b),
            measure="spearman",
            value=spearman_of_gaussian(rho),
            mechanism=f"gaussian copula rho={rho}",
        )

    return Table(
        name="measurements",
        description="One row per measurement; every column has a known marginal or a known dependency.",
        primary_key=("id",),
        columns=(
            Column(name="id", unique=True, distinct=n, distribution=dist(Family.SEQUENCE, start=1, step=1)),
            Column(name="normal", null_rate=NORMAL_NULL_RATE, distribution=dist(Family.NORMAL, **NORMAL)),
            Column(name="lognormal", distribution=dist(Family.LOGNORMAL, **LOGNORMAL)),
            Column(name="bimodal", distribution=dist(Family.MIXTURE, components=MIXTURE)),
            Column(name="pareto", distribution=dist(Family.PARETO, **PARETO), note="Heavy tail; infinite variance."),
            Column(name="count_poisson", distribution=dist(Family.POISSON, **POISSON)),
            Column(
                name="count_negbin",
                null_rate=NEGATIVE_BINOMIAL_NULL_RATE,
                distribution=dist(Family.NEGATIVE_BINOMIAL, **NEGATIVE_BINOMIAL),
                note="Overdispersed counts.",
            ),
            Column(
                name="rating",
                max_distinct=len(RATINGS),
                distribution=dist(Family.CATEGORICAL, values=RATINGS, probabilities=RATING_PROBABILITIES),
            ),
            Column(name="ratio", note="Stored as float.", distribution=dist(Family.UNIFORM, low=0.0, high=1.0)),
            Column(name="copula_x", distribution=dist(Family.NORMAL, **COPULA_X), note="10 + 2 * z1."),
            Column(name="copula_y", distribution=dist(Family.LOGNORMAL, mu=0.0, sigma=1.0), note="exp(z2)."),
            Column(name="copula_z", note="z3 ** 3; (z1, z2, z3) is a Gaussian copula."),
            Column(name="linear_a", distribution=dist(Family.NORMAL, mean=0.0, std=1.0)),
            Column(name="linear_b", distribution=dist(Family.NORMAL, mean=0.0, std=1.0)),
            Column(name="nl_x", distribution=dist(Family.UNIFORM, low=-NONLINEAR_RANGE, high=NONLINEAR_RANGE)),
            Column(name="nl_y", note=f"nl_x ** 2 + N(0, {NONLINEAR_NOISE})."),
            Column(name="range_low", distribution=dist(Family.UNIFORM, low=0.0, high=100.0)),
            Column(name="range_high", note=f"range_low + Exponential({GAP_SCALE})."),
        ),
        checks=(
            RangeCheck(column="ratio", min=0.0, max=1.0),
            RangeCheck(column="count_poisson", min=0),
            RangeCheck(column="lognormal", min=0.0),
            RangeCheck(column="pareto", min=PARETO["scale"]),
            EnumerationCheck(column="rating", values=tuple(RATINGS)),
            ComparisonCheck(left="range_low", op="<=", right="range_high"),
        ),
        dependencies=(
            copula("copula_x", "copula_y", 0, 1),
            copula("copula_x", "copula_z", 0, 2),
            copula("copula_y", "copula_z", 1, 2),
            Dependency(
                columns=("linear_a", "linear_b"),
                measure="pearson",
                value=LINEAR_RHO,
                mechanism=f"linear_b = {LINEAR_RHO} * linear_a + noise",
            ),
            Dependency(columns=("nl_x", "nl_y"), measure="pearson", value=0.0, mechanism="nl_y = nl_x ** 2 + noise"),
            Dependency(
                columns=("nl_x", "nl_y"),
                measure="correlation_ratio",
                value=nonlinear_eta(),
                mechanism="nl_y = nl_x ** 2 + noise",
            ),
        ),
    )


DATASET = Dataset(
    name="numeric",
    description="Parametric marginals (normal, lognormal, mixture, Pareto, counts), a copula, and a nonlinear pair.",
    build=build,
    sizes=sizes,
)
