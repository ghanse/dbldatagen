"""Checks that a manifest is true of the data it describes.

Structural claims (row counts, types, keys, foreign keys, check constraints, functional
dependencies, null rates, exact distinct counts) must hold exactly. Statistical claims
(distribution parameters, format shares, dependency strengths, fan-out) must hold within a
tolerance of a few standard errors for the observed sample size. Verification collects
column values to the driver, so it is meant for the `xs` and `s` tiers.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import Column as SparkColumn
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from profiler_bench.catalog.manifest import (
    Bound,
    Check,
    Column,
    ComparisonCheck,
    Dependency,
    Distribution,
    EnumerationCheck,
    Family,
    ForeignKey,
    Manifest,
    RangeCheck,
    Table,
)


Z = 5.0
MICROS_PER_SECOND = 1_000_000
MICROS_PER_HOUR = 3_600 * MICROS_PER_SECOND
MICROS_PER_DAY = 24 * MICROS_PER_HOUR
EPOCH_WEEKDAY = 3
CORRELATION_BINS = 50
EPOCH_ORDINAL = date(1970, 1, 1).toordinal()


def verify(truth: Manifest, frames: Mapping[str, DataFrame]) -> list[str]:
    """Returns every claim of `truth` that the data in `frames` does not satisfy.

    Args:
        truth: Ground-truth manifest.
        frames: DataFrames by table name, as returned by `load`.

    Returns:
        One message per failed claim; empty when the manifest is true of the data.

    Raises:
        ValueError: If the Spark session time zone is not UTC, which the timestamp claims assume.
    """
    session = next(iter(frames.values())).sparkSession
    if session.conf.get("spark.sql.session.timeZone") not in ("UTC", "Etc/UTC", "Z"):
        raise ValueError("verify needs spark.sql.session.timeZone=UTC")
    failures = [f"{t.name}: {m}" for t in truth.tables for m in _verify_table(t, frames[t.name])]
    for fk in truth.foreign_keys:
        label = f"{fk.child_table}{list(fk.child_columns)} -> {fk.parent_table}{list(fk.parent_columns)}"
        failures += [f"{label}: {m}" for m in _verify_foreign_key(fk, frames)]
    return failures


def _verify_table(table: Table, df: DataFrame) -> list[str]:
    rows = df.count()
    failures = [] if rows == table.row_count else [f"row count {rows} != {table.row_count}"]
    actual = [(f.name, f.dataType.simpleString()) for f in df.schema.fields]
    expected = [(c.name, c.type) for c in table.columns]
    if actual != expected:
        return [*failures, f"schema {actual} != {expected}"]
    failures += _verify_columns(df, table.columns, "")
    failures += _verify_keys(table, df, rows)
    failures += [m for check in table.checks for m in _verify_check(check, df)]
    for fd in table.functional_dependencies:
        cols = [*fd.determinant, *fd.dependent]
        worst = (
            df.dropna(subset=cols)
            .groupBy(*fd.determinant)
            .agg(F.countDistinct(*fd.dependent).alias("n"))
            .agg(F.max("n"))
            .collect()[0][0]
        )
        if worst is not None and worst > 1:
            failures.append(f"functional dependency {list(fd.determinant)} -> {list(fd.dependent)} violated")
    if table.dependencies:
        names = sorted({c for dep in table.dependencies for c in dep.columns})
        raw = _collect(df, names)
        pairs = pd.DataFrame({c: _values(raw[c], df.schema[c].dataType) for c in names})
        failures += [m for dep in table.dependencies for m in _verify_dependency(dep, pairs)]
    return failures


def _verify_keys(table: Table, df: DataFrame, rows: int) -> list[str]:
    failures = []
    if table.primary_key:
        key = list(table.primary_key)
        if df.dropna(subset=key).select(*key).distinct().count() != rows:
            failures.append(f"primary key {key} has nulls or duplicates")
    for unique in table.unique_keys:
        present = df.dropna(subset=list(unique)).select(*unique)
        if present.distinct().count() != present.count():
            failures.append(f"unique key {list(unique)} has duplicates")
    return failures


def _verify_check(check: Check, df: DataFrame) -> list[str]:
    if isinstance(check, RangeCheck):
        column, dtype = F.col(f"`{check.column}`"), df.schema[check.column].dataType
        bad = F.lit(False)
        if check.min is not None:
            bad = bad | (column < F.lit(check.min).cast(dtype))
        if check.max is not None:
            bad = bad | (column > F.lit(check.max).cast(dtype))
        condition, label = bad, f"range {check.column} in [{check.min}, {check.max}]"
    elif isinstance(check, EnumerationCheck):
        condition, label = ~F.col(f"`{check.column}`").isin(list(check.values)), f"enumeration {check.column}"
    elif isinstance(check, ComparisonCheck):
        sql = f"`{check.left}` {check.op} `{check.right}`"
        condition, label = ~F.expr(sql), sql
    else:
        condition, label = ~F.expr(check.sql), check.sql
    violations = df.where(condition).count()
    return [f"check {label} violated by {violations} rows"] if violations else []


def _verify_foreign_key(fk: ForeignKey, frames: Mapping[str, DataFrame]) -> list[str]:
    child = frames[fk.child_table].select(*fk.child_columns).dropna()
    parent = frames[fk.parent_table].select(
        *(F.col(p).alias(c) for p, c in zip(fk.parent_columns, fk.child_columns, strict=True))
    )
    total = child.count()
    missing = child.join(parent, list(fk.child_columns), "left_anti").count()
    coverage = 1.0 - missing / total if total else 1.0
    failures = _share("coverage", coverage, fk.coverage, total)
    if fk.fanout is not None:
        joined: pd.DataFrame = (
            parent.join(child.groupBy(*fk.child_columns).count(), list(fk.child_columns), "left")
            .fillna(0, subset=["count"])
            .toPandas()
        )
        counts = joined["count"].to_numpy(dtype=np.float64)
        failures += [f"fanout {m}" for m in check_distribution(counts, fk.fanout, _NUMBER, counts_input=True)]
    return failures


def _verify_dependency(dep: Dependency, values: pd.DataFrame) -> list[str]:
    pdf = values[list(dep.columns)].dropna()
    x, y = pdf[dep.columns[0]], pdf[dep.columns[1]]
    if dep.measure == "pearson":
        observed = float(np.corrcoef(x.astype(float), y.astype(float))[0, 1])
    elif dep.measure == "spearman":
        observed = float(np.corrcoef(x.rank(), y.rank())[0, 1])
    elif dep.measure == "cramers_v":
        observed = _cramers_v(x, y)
    else:
        observed = _correlation_ratio(x, y)
    tolerance = 0.02 + 4 * (1 - dep.value**2) / math.sqrt(len(pdf))
    return _near(f"{dep.measure}{list(dep.columns)}", observed, dep.value, tolerance)


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y).to_numpy(dtype=np.float64)
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
    chi2 = float(((table - expected) ** 2 / expected).sum())
    return math.sqrt(chi2 / (table.sum() * (min(table.shape) - 1)))


def _correlation_ratio(x: pd.Series, y: pd.Series) -> float:
    groups = pd.qcut(x, CORRELATION_BINS, labels=False, duplicates="drop") if pd.api.types.is_numeric_dtype(x) else x
    y = y.astype(float)
    means = y.groupby(groups).transform("mean")
    return math.sqrt(float(((means - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum()))


_NUMBER, _DATE, _TIMESTAMP, _RAW = "number", "date", "timestamp", "raw"


def _kind(dtype: T.DataType) -> str:
    if isinstance(dtype, T.DateType):
        return _DATE
    if isinstance(dtype, T.TimestampType | T.TimestampNTZType):
        return _TIMESTAMP
    if isinstance(dtype, T.NumericType):
        return _NUMBER
    return _RAW


def _collect(frame: DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """Collects columns to pandas, with timestamps as epoch microseconds computed by Spark.

    Converting in Spark keeps the session time zone; `toPandas` would apply the local one.
    """
    schema = {f.name: f.dataType for f in frame.schema.fields}

    def expr(name: str) -> SparkColumn:
        column = F.col(f"`{name}`")
        if isinstance(schema[name], T.TimestampNTZType):
            column = column.cast(T.TimestampType())
        return F.unix_micros(column).alias(name) if _kind(schema[name]) == _TIMESTAMP else column

    return frame.select(*(expr(n) for n in names)).toPandas()


def _values(series: pd.Series, dtype: T.DataType) -> pd.Series:
    """Returns the non-null, finite values of a collected column in storage order.

    Arrays, maps, and binary values become their lengths, dates days since the epoch, and numbers
    (including timestamps already collected as microseconds) floats.
    """
    present = series.dropna()
    if isinstance(dtype, T.ArrayType | T.MapType | T.BinaryType):
        return present.map(len).astype(np.float64)
    if isinstance(dtype, T.DateType):
        return present.map(lambda d: d.toordinal() - EPOCH_ORDINAL).astype(np.float64)
    if isinstance(dtype, T.NumericType):
        numbers = present.astype(np.float64)
        return numbers[np.isfinite(numbers)]
    return present


def _verify_columns(frame: DataFrame, columns: Sequence[Column], path: str) -> list[str]:
    schema = {f.name: f.dataType for f in frame.schema.fields}
    aggregates = [F.count(F.lit(1)).alias("__rows")]
    for c in columns:
        col = F.col(f"`{c.name}`")
        aggregates.append(F.count(col).alias(f"{c.name}__present"))
        if c.unique or c.distinct is not None or c.max_distinct is not None:
            aggregates.append(F.countDistinct(col).alias(f"{c.name}__distinct"))
        for name, bound, aggregate in (("min", c.min_value, F.min), ("max", c.max_value, F.max)):
            if bound is not None:
                aggregates.append((aggregate(col) == F.lit(bound).cast(schema[c.name])).alias(f"{c.name}__{name}"))
        if isinstance(schema[c.name], T.FloatType | T.DoubleType):
            aggregates.append(F.sum(F.isnan(col).cast("int")).alias(f"{c.name}__nan"))
            aggregates.append(F.sum((col == float("inf")).cast("int")).alias(f"{c.name}__inf"))
            aggregates.append(F.sum((col == float("-inf")).cast("int")).alias(f"{c.name}__-inf"))
    stats = frame.agg(*aggregates).collect()[0].asDict()
    values = _collect(frame, [c.name for c in columns])
    failures = []
    for c in columns:
        label = f"{path}{c.name}"
        series = _values(values[c.name], schema[c.name])
        failures += [f"{label}: {m}" for m in _verify_column(c, schema[c.name], stats, series)]
        failures += _verify_children(frame, c, schema[c.name], f"{label}.")
    return failures


def _verify_column(column: Column, dtype: T.DataType, stats: dict[str, Any], series: pd.Series) -> list[str]:
    rows, present = stats["__rows"], stats[f"{column.name}__present"]
    failures = _rate("null rate", 1 - present / rows if rows else 0.0, column.null_rate, rows)
    distinct = stats.get(f"{column.name}__distinct")
    if column.unique and distinct != present:
        failures.append(f"{present - (distinct or 0)} duplicate values in a unique column")
    if column.distinct is not None and distinct != column.distinct:
        failures.append(f"distinct {distinct} != {column.distinct}")
    if column.max_distinct is not None and distinct is not None and distinct > column.max_distinct:
        failures.append(f"distinct {distinct} > max {column.max_distinct}")
    for name, bound in (("min", column.min_value), ("max", column.max_value)):
        if bound is not None and not stats[f"{column.name}__{name}"]:
            failures.append(f"{name} is not {bound}")
    for special, share in column.special_values.items():
        failures += _share(f"{special} share", stats[f"{column.name}__{special}"] / max(present, 1), share, present)
    if isinstance(dtype, T.ArrayType | T.MapType | T.BinaryType):
        if column.length is not None:
            failures += [f"length {m}" for m in check_distribution(series.to_numpy(), column.length, _NUMBER)]
        return failures
    kind = _kind(dtype)
    if column.formats:
        failures += _verify_formats(column, series.astype(str))
    if column.distribution is not None:
        failures += check_distribution(series.to_numpy(), column.distribution, kind)
    if column.monotonic is not None or column.increments is not None:
        failures += _verify_order(column, series.to_numpy(dtype=np.float64), kind)
    return failures


def _verify_formats(column: Column, strings: pd.Series) -> list[str]:
    failures, matched = [], np.zeros(len(strings), dtype=bool)
    for fmt in column.formats:
        pattern = re.compile(fmt.regex)
        hits = strings.map(lambda s, p=pattern: p.fullmatch(s) is not None).to_numpy(dtype=bool)
        matched |= hits
        failures += _share(f"format {fmt.regex!r} share", hits.mean() if len(hits) else 0.0, fmt.share, len(hits))
    unmatched = 1.0 - matched.mean() if len(matched) else 0.0
    declared = max(0.0, 1.0 - sum(f.share for f in column.formats))
    return failures + _share("unmatched share", unmatched, declared, len(strings))


def _verify_order(column: Column, values: np.ndarray, kind: str) -> list[str]:
    steps = np.diff(values)
    failures = []
    if column.monotonic == "increasing" and (steps < 0).any():
        failures.append("values decrease in storage order")
    if column.monotonic == "strictly_increasing" and (steps <= 0).any():
        failures.append("values do not strictly increase in storage order")
    if column.increments is not None:
        scaled = steps / MICROS_PER_SECOND if kind == _TIMESTAMP else steps
        failures += [f"increments {m}" for m in check_distribution(scaled, column.increments, _NUMBER)]
    return failures


def _verify_children(frame: DataFrame, column: Column, dtype: T.DataType, path: str) -> list[str]:
    col = F.col(f"`{column.name}`")
    if isinstance(dtype, T.StructType):
        nested = frame.where(col.isNotNull()).select(*(col[f.name].alias(f.name) for f in column.fields))
        return _verify_columns(nested, column.fields, path)
    if isinstance(dtype, T.ArrayType) and column.element is not None:
        return _verify_columns(frame.select(F.explode(col).alias("element")), (column.element,), path)
    if isinstance(dtype, T.MapType) and column.key is not None and column.value is not None:
        return _verify_columns(frame.select(F.explode(col).alias("key", "value")), (column.key, column.value), path)
    return []


def _near(label: str, observed: float, expected: float, tolerance: float) -> list[str]:
    if abs(observed - expected) <= tolerance:
        return []
    return [f"{label} {observed:.6g}, expected {expected:.6g} ± {tolerance:.3g}"]


def _share(label: str, observed: float, expected: float, n: int) -> list[str]:
    tolerance = 0.002 + Z * math.sqrt(expected * (1 - expected) / max(n, 1))
    return _near(label, observed, expected, tolerance)


def _rate(label: str, observed: float, expected: float, n: int) -> list[str]:
    if expected in (0.0, 1.0):
        return [] if observed == expected else [f"{label} {observed:.6g} != {expected}"]
    return _share(label, observed, expected, n)


def _to_number(value: Bound, kind: str) -> float:
    if kind == _DATE:
        return float(np.datetime64(str(value), "D").astype(np.int64))
    if kind == _TIMESTAMP:
        return float(np.datetime64(str(value).rstrip("Z"), "us").astype(np.int64))
    return float(value)


def check_distribution(
    values: np.ndarray, dist: Distribution, kind: str = _NUMBER, counts_input: bool = False
) -> list[str]:
    """Returns the ways `values` departs from `dist` by more than the sampling tolerance.

    Args:
        values: Non-null values in storage order: numbers (days for dates, microseconds for
            timestamps) or raw values for strings and booleans.
        dist: Declared distribution.
        kind: `number`, `date`, `timestamp`, or `raw`; selects how ISO parameters are read.
        counts_input: Whether `values` are counts per category (as for foreign-key fan-out)
            rather than draws. Only changes the meaning of `zipf`.
    """
    if len(values) == 0:
        return ["no values to check"]
    if dist.family == Family.ZIPF:
        counts = values if counts_input else pd.Series(values).value_counts().to_numpy()
        return _check_zipf(np.asarray(counts, dtype=np.float64), dist.params["s"], dist.params["n_values"])
    if dist.family in (Family.CATEGORICAL, Family.CONSTANT) and kind == _RAW:
        return _check_categorical(values, dist)
    params = {
        k: _to_number(v, kind) if k in ("low", "high", "start", "end", "value") else v for k, v in dist.params.items()
    }
    return _CHECKS[dist.family](np.asarray(values, dtype=np.float64), params)


def _check_categorical(values: np.ndarray, dist: Distribution) -> list[str]:
    if dist.family == Family.CONSTANT:
        others = int((pd.Series(values) != dist.params["value"]).sum())
        return [f"{others} values differ from the constant"] if others else []
    shares = pd.Series(values).value_counts(normalize=True)
    unknown = sorted(map(str, set(shares.index) - set(dist.params["values"])))
    failures = [f"unexpected values {unknown[:5]}"] if unknown else []
    for value, p in zip(dist.params["values"], dist.params["probabilities"], strict=True):
        failures += _share(f"share of {value!r}", float(shares.get(value, 0.0)), p, len(values))
    return failures


def _check_zipf(counts: np.ndarray, s: float, n_values: int) -> list[str]:
    total = counts.sum()
    if len(counts) > n_values:
        return [f"{len(counts)} distinct values > n_values {n_values}"]
    p = np.sort(np.arange(1, n_values + 1, dtype=np.float64) ** -s)[::-1]
    p /= p.sum()
    observed = np.sort(np.pad(counts, (0, n_values - len(counts))))[::-1] / total
    failures = [m for k in range(min(3, n_values)) for m in _share(f"rank {k + 1} share", observed[k], p[k], total)]
    hit = 1.0 - (1.0 - p) ** total
    used = float((counts > 0).sum())
    tolerance = 1 + Z * math.sqrt(float((hit * (1 - hit)).sum()))
    return failures + _near("distinct used", used, float(hit.sum()), tolerance)


def _mean_std(values: np.ndarray, mean: float, std: float) -> list[str]:
    n = len(values)
    return _near("mean", float(values.mean()), mean, Z * std / math.sqrt(n) + 1e-9) + _near(
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
    return _mean_std(v, p["mean"], p["std"])


def _lognormal(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    return _bounds(v, 1e-300, None) or _mean_std(np.log(v), p["mu"], p["sigma"])


def _mixture(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    components = p["components"]
    mean = sum(c["weight"] * c["mean"] for c in components)
    variance = sum(c["weight"] * (c["std"] ** 2 + c["mean"] ** 2) for c in components) - mean**2
    centered = v - v.mean()
    fourth = float((centered**4).mean())
    n = len(v)
    return _near("mean", float(v.mean()), mean, Z * math.sqrt(variance / n)) + _near(
        "variance", float(centered.var()), variance, Z * math.sqrt(max(fourth - variance**2, 1e-12) / n)
    )


def _uniform(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    low, high = p["low"], p["high"]
    tolerance = Z * (high - low) / math.sqrt(12 * len(v)) + 1e-9
    return _bounds(v, low, high) + _near("mean", float(v.mean()), (low + high) / 2, tolerance)


def _discrete_uniform(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    return _integral(v) + _uniform(v, p)


def _exponential(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    scale = p["scale"]
    return _bounds(v, 0.0, None) + _near("mean", float(v.mean()), scale, Z * scale / math.sqrt(len(v)))


def _pareto(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    alpha, scale = p["alpha"], p["scale"]
    failures = _bounds(v, scale * (1 - 1e-9), None)
    for factor in (2.0, 10.0):
        failures += _share(f"share above {factor}x scale", float((v > factor * scale).mean()), factor**-alpha, len(v))
    return failures


def _poisson(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    lam, shift, n = p["lam"], p.get("shift", 0), len(v)
    return (
        _integral(v)
        + _bounds(v, shift, None)
        + _near("mean", float(v.mean()), lam + shift, Z * math.sqrt(lam / n))
        + _near("variance", float(v.var()), lam, Z * math.sqrt((lam + 2 * lam**2) / n))
    )


def _binomial(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    trials, prob = p["n"], p["p"]
    sd = math.sqrt(trials * prob * (1 - prob))
    return (
        _integral(v) + _bounds(v, 0, trials) + _near("mean", float(v.mean()), trials * prob, Z * sd / math.sqrt(len(v)))
    )


def _negative_binomial(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    size, prob = p["n"], p["p"]
    mean, variance = size * (1 - prob) / prob, size * (1 - prob) / prob**2
    return _integral(v) + _bounds(v, 0, None) + _near("mean", float(v.mean()), mean, Z * math.sqrt(variance / len(v)))


def _constant(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    others = int((v != p["value"]).sum())
    return [f"{others} values differ from the constant"] if others else []


def _categorical_numbers(v: np.ndarray, p: dict[str, Any]) -> list[str]:
    return _check_categorical(v, Distribution(family=Family.CATEGORICAL, params=p))


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
        shares = np.bincount(index, minlength=len(weights)) / len(v)
        expected = np.asarray(weights, dtype=np.float64) / sum(weights)
        failures += [
            m for i in range(len(weights)) for m in _share(f"{label} {i} share", shares[i], expected[i], len(v))
        ]
    return failures


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
    Family.CATEGORICAL: _categorical_numbers,
    Family.SEQUENCE: _sequence,
    Family.SEASONAL: _seasonal,
}
