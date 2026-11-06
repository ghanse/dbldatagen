"""Checks that a manifest is true of the data it describes.

Structural claims (row counts, types, keys, foreign keys, check constraints, functional
dependencies, exact distinct counts, planted extremes, null rates of 0 or 1) must hold exactly.
Statistical claims are tested at significance `ALPHA` (see `profiler_bench.catalog.distributions`):
distributions by goodness of fit, shares and rates within `Z` binomial standard errors, and
dependency strengths with a measure-specific standard error:

- Pearson: Fisher z, `var(atanh r) = 1 / (n - 3)`. This assumes bivariate normality; for
  non-normal pairs the true variance can be larger, which makes the test stricter.
- Spearman: Fisher z with the Fieller-Hartley-Pearson correction, `var = 1.06 / (n - 3)`.
- Cramér's V: seeded parametric bootstrap, resampling the observed contingency table from its
  multinomial distribution `BOOTSTRAP_SAMPLES` times.
- Correlation ratio: seeded nonparametric bootstrap over rows with the bins held fixed.

Verification collects column values to the driver, so it is meant for the `xs` and `s` tiers.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from pyspark.errors import AnalysisException
from pyspark.sql import Column as SparkColumn
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from profiler_bench.catalog.distributions import ALPHA, DATE, NUMBER, RAW, TIMESTAMP, Z, check_distribution, near, share
from profiler_bench.catalog.manifest import (
    Check,
    Column,
    ComparisonCheck,
    Dependency,
    EnumerationCheck,
    ForeignKey,
    Manifest,
    RangeCheck,
    Table,
)


__all__ = ["ALPHA", "Z", "check_distribution", "verify"]

MICROS_PER_SECOND = 1_000_000
CORRELATION_BINS = 50
EPOCH_ORDINAL = date(1970, 1, 1).toordinal()
SPEARMAN_VARIANCE = 1.06
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_SEED = 20_261_106


def verify(truth: Manifest, frames: Mapping[str, DataFrame], tables: Collection[str] | None = None) -> list[str]:
    """Returns every claim of `truth` that the data in `frames` does not satisfy.

    Args:
        truth: Ground-truth manifest.
        frames: DataFrames by table name, as returned by `load`.
        tables: Tables whose claims to check, with the foreign keys between them; all when None.

    Returns:
        One message per failed claim; empty when the manifest is true of the data.

    Raises:
        ValueError: If the Spark session time zone is not UTC, which the timestamp claims assume.
    """
    session = next(iter(frames.values())).sparkSession
    if session.conf.get("spark.sql.session.timeZone") not in ("UTC", "Etc/UTC", "Z"):
        raise ValueError("verify needs spark.sql.session.timeZone=UTC")
    selected = {t.name for t in truth.tables} if tables is None else set(tables)
    failures = [f"{t.name}: {m}" for t in truth.tables if t.name in selected for m in _verify_table(t, frames[t.name])]
    for fk in (fk for fk in truth.foreign_keys if {fk.child_table, fk.parent_table} <= selected):
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
    ordered = any(c.monotonic is not None or c.increments is not None for c in table.columns)
    failures += _verify_columns(storage_ordered(df) if ordered else df, table.columns, "")
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
        raw = collect(df, names)
        pairs = pd.DataFrame({c: values_of(raw[c], df.schema[c].dataType) for c in names})
        failures += [m for dep in table.dependencies for m in _verify_dependency(dep, pairs)]
    for qi in table.quasi_identifiers:
        unique_rows = df.groupBy(*qi.columns).count().where("count = 1").count()
        failures += share(f"unique share of {list(qi.columns)}", unique_rows / max(rows, 1), qi.unique_fraction, rows)
    return failures


def storage_ordered(df: DataFrame) -> DataFrame:
    """Returns `df` sorted into storage order: part files by path, then rows within each file.

    Spark does not read multi-file tables in file order, so storage-order claims need this. A
    DataFrame that is not a plain file scan (for example a union) is returned unchanged.
    """
    try:
        with_file = df.withColumn("__file", F.col("_metadata.file_path"))
    except AnalysisException:
        return df
    return (
        with_file.withColumn("__row", F.monotonically_increasing_id())
        .orderBy("__file", "__row")
        .drop("__file", "__row")
    )


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
    failures = _rate("coverage", coverage, fk.coverage, total)
    if fk.fanout is not None:
        joined: pd.DataFrame = (
            parent.join(child.groupBy(*fk.child_columns).count(), list(fk.child_columns), "left")
            .fillna(0, subset=["count"])
            .toPandas()
        )
        counts = joined["count"].to_numpy(dtype=np.float64)
        failures += [f"fanout {m}" for m in check_distribution(counts, fk.fanout, NUMBER, counts_input=True)]
    return failures


def dependency_value(measure: str, x: pd.Series, y: pd.Series) -> float:
    """Returns the realized value of a dependency measure over aligned non-null pairs."""
    if measure == "pearson":
        return float(np.corrcoef(x.astype(float), y.astype(float))[0, 1])
    if measure == "spearman":
        return float(np.corrcoef(x.rank(), y.rank())[0, 1])
    if measure == "cramers_v":
        return _cramers_v(pd.crosstab(x, y).to_numpy(dtype=np.float64))
    codes, values = _bins(x), y.to_numpy(dtype=np.float64)
    return _correlation_ratio(codes, values)


def _verify_dependency(dep: Dependency, values: pd.DataFrame) -> list[str]:
    pdf = values[list(dep.columns)].dropna()
    x, y, n = pdf[dep.columns[0]], pdf[dep.columns[1]], len(pdf)
    label = f"{dep.measure}{list(dep.columns)}"
    observed = dependency_value(dep.measure, x, y)
    if dep.measure in ("pearson", "spearman"):
        factor = 1.0 if dep.measure == "pearson" else SPEARMAN_VARIANCE
        return _fisher(label, observed, dep.value, n, factor)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    if dep.measure == "cramers_v":
        table = pd.crosstab(x, y).to_numpy(dtype=np.float64)
        draws = rng.multinomial(n, (table / n).ravel(), BOOTSTRAP_SAMPLES).reshape(-1, *table.shape)
        replicates = np.asarray([_cramers_v(t.astype(np.float64)) for t in draws])
    else:
        codes, ys = _bins(x), y.to_numpy(dtype=np.float64)
        resamples = (rng.integers(0, n, n) for _ in range(BOOTSTRAP_SAMPLES))
        replicates = np.asarray([_correlation_ratio(codes[r], ys[r]) for r in resamples])
    return near(f"{label} (bootstrap)", observed, dep.value, Z * float(replicates.std(ddof=1)) + 1e-9)


def _fisher(label: str, observed: float, expected: float, n: int, factor: float) -> list[str]:
    if n <= 3:
        return [f"{label}: too few pairs ({n})"]
    se = math.sqrt(factor / (n - 3))
    z_observed, z_expected = (math.atanh(min(max(r, -0.999999), 0.999999)) for r in (observed, expected))
    if abs(z_observed - z_expected) <= Z * se:
        return []
    return [
        f"{label} {observed:.4g}, expected {expected:.4g} (Fisher z off by {abs(z_observed - z_expected) / se:.3g} se)"
    ]


def _cramers_v(table: np.ndarray) -> float:
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
    chi2 = float(((table - expected) ** 2 / expected).sum())
    return math.sqrt(chi2 / (table.sum() * (min(table.shape) - 1)))


def _bins(x: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(x):
        return np.asarray(pd.qcut(x, CORRELATION_BINS, labels=False, duplicates="drop"), dtype=np.int64)
    return np.asarray(pd.factorize(x)[0], dtype=np.int64)


def _correlation_ratio(codes: np.ndarray, y: np.ndarray) -> float:
    counts = np.bincount(codes)
    sums = np.bincount(codes, weights=y)
    present = counts > 0
    means = sums[present] / counts[present]
    total = ((y - y.mean()) ** 2).sum()
    return math.sqrt(float((counts[present] * (means - y.mean()) ** 2).sum() / total))


def _kind(dtype: T.DataType) -> str:
    if isinstance(dtype, T.DateType):
        return DATE
    if isinstance(dtype, T.TimestampType | T.TimestampNTZType):
        return TIMESTAMP
    if isinstance(dtype, T.NumericType):
        return NUMBER
    return RAW


def collect(frame: DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """Collects columns to pandas, with timestamps as epoch microseconds computed by Spark.

    Converting in Spark keeps the session time zone; `toPandas` would apply the local one.
    """
    schema = {f.name: f.dataType for f in frame.schema.fields}

    def expr(name: str) -> SparkColumn:
        column = F.col(f"`{name}`")
        if isinstance(schema[name], T.TimestampNTZType):
            column = column.cast(T.TimestampType())
        return F.unix_micros(column).alias(name) if _kind(schema[name]) == TIMESTAMP else column

    return frame.select(*(expr(n) for n in names)).toPandas()


def values_of(series: pd.Series, dtype: T.DataType) -> pd.Series:
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
            if c.special_values:
                aggregates.append(F.sum(F.isnan(col).cast("int")).alias(f"{c.name}__nan"))
                aggregates.append(F.sum((col == float("inf")).cast("int")).alias(f"{c.name}__inf"))
                aggregates.append(F.sum((col == float("-inf")).cast("int")).alias(f"{c.name}__-inf"))
            else:
                nonfinite = F.isnan(col) | (F.abs(col) == float("inf"))
                aggregates.append(F.sum(nonfinite.cast("int")).alias(f"{c.name}__nonfinite"))
    stats = frame.agg(*aggregates).collect()[0].asDict()
    values = collect(frame, [c.name for c in columns])
    failures = []
    for c in columns:
        label = f"{path}{c.name}"
        series = values_of(values[c.name], schema[c.name])
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
    if stats.get(f"{column.name}__nonfinite"):
        failures.append(f"{stats[f'{column.name}__nonfinite']} undeclared NaN or infinite values")
    for special, expected in column.special_values.items():
        observed = stats[f"{column.name}__{special}"] / max(present, 1)
        failures += share(f"{special} share", observed, expected, present)
    if isinstance(dtype, T.ArrayType | T.MapType | T.BinaryType):
        if column.length is not None:
            failures += [f"length {m}" for m in check_distribution(series.to_numpy(), column.length, NUMBER)]
        return failures
    kind = _kind(dtype)
    if column.formats:
        failures += _verify_formats(column, series.astype(str))
    if column.distribution is not None:
        failures += check_distribution(series.to_numpy(), column.distribution, kind)
    if column.monotonic is not None or column.increments is not None:
        failures += _verify_order(column, series.to_numpy(dtype=np.float64), kind)
    return failures


def _rate(label: str, observed: float, expected: float, n: int) -> list[str]:
    if expected in (0.0, 1.0):
        return [] if observed == expected else [f"{label} {observed:.6g} != {expected}"]
    return share(label, observed, expected, n)


def _verify_formats(column: Column, strings: pd.Series) -> list[str]:
    failures, matched = [], np.zeros(len(strings), dtype=bool)
    for fmt in column.formats:
        pattern = re.compile(fmt.regex)
        hits = strings.map(lambda s, p=pattern: p.fullmatch(s) is not None).to_numpy(dtype=bool)
        matched |= hits
        failures += share(f"format {fmt.regex!r} share", hits.mean() if len(hits) else 0.0, fmt.share, len(hits))
    unmatched = 1.0 - matched.mean() if len(matched) else 0.0
    declared = max(0.0, 1.0 - sum(f.share for f in column.formats))
    return failures + share("unmatched share", unmatched, declared, len(strings))


def _verify_order(column: Column, values: np.ndarray, kind: str) -> list[str]:
    steps = np.diff(values)
    failures = []
    if column.monotonic == "increasing" and (steps < 0).any():
        failures.append("values decrease in storage order")
    if column.monotonic == "strictly_increasing" and (steps <= 0).any():
        failures.append("values do not strictly increase in storage order")
    if column.increments is not None:
        scaled = steps / MICROS_PER_SECOND if kind == TIMESTAMP else steps
        failures += [f"increments {m}" for m in check_distribution(scaled, column.increments, NUMBER)]
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
