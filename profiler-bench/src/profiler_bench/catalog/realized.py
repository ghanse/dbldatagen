"""Realized statistics of a materialized dataset instance.

Manifests declare generating parameters. Metrics that need what one instance actually contains
(for example, to separate profiler error from sampling noise) use these statistics instead. They
cover top-level columns; nested fields are not summarized.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from profiler_bench.catalog.manifest import Family, Manifest, Table
from profiler_bench.catalog.registry import DEFAULT_TIER
from profiler_bench.catalog.store import load, materialize
from profiler_bench.catalog.verification import collect, dependency_value, values_of


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RealizedColumn(_Model):
    """Realized statistics of one top-level column.

    Attributes:
        name: Column name.
        null_rate: Fraction of null values.
        distinct: Number of distinct non-null values, or None for map columns.
        min: Minimum non-null value as Spark casts it to a string, or None.
        max: Maximum non-null value as Spark casts it to a string, or None.
        mean: Mean of the finite non-null values in the column's unit (days for dates,
            microseconds for timestamps, element count for arrays and maps), or None.
        std: Population standard deviation in the same unit, or None.
        format_shares: Share of non-null values matching each declared format regex.
        category_shares: Share of non-null values equal to each declared categorical value.
    """

    name: str
    null_rate: float
    distinct: int | None
    min: str | None
    max: str | None
    mean: float | None
    std: float | None
    format_shares: dict[str, float] = {}
    category_shares: dict[str, float] = {}


class RealizedDependency(_Model):
    """Realized value of a declared pairwise dependency."""

    columns: tuple[str, str]
    measure: str
    value: float


class RealizedQuasiIdentifier(_Model):
    """Realized share of rows whose quasi-identifier combination is unique."""

    columns: tuple[str, ...]
    unique_fraction: float


class RealizedTable(_Model):
    """Realized statistics of one table."""

    name: str
    row_count: int
    columns: tuple[RealizedColumn, ...]
    dependencies: tuple[RealizedDependency, ...] = ()
    quasi_identifiers: tuple[RealizedQuasiIdentifier, ...] = ()


class RealizedForeignKey(_Model):
    """Realized coverage and fan-out of a declared foreign key.

    Attributes:
        child_table: Referencing table.
        child_columns: Referencing columns.
        parent_table: Referenced table.
        coverage: Fraction of non-null child keys that match a parent key.
        fanout_mean: Mean child rows per parent key, counting keys with none.
        fanout_max: Largest number of child rows of one parent key.
        fanout_zero_share: Share of parent keys with no child rows.
    """

    child_table: str
    child_columns: tuple[str, ...]
    parent_table: str
    coverage: float
    fanout_mean: float
    fanout_max: int
    fanout_zero_share: float


class Realized(_Model):
    """Realized statistics of one dataset instance."""

    dataset: str
    tier: str
    seed: int
    tables: tuple[RealizedTable, ...]
    foreign_keys: tuple[RealizedForeignKey, ...] = ()


def realized(truth: Manifest, frames: Mapping[str, DataFrame]) -> Realized:
    """Returns the realized statistics of the data in `frames` for the claims declared in `truth`.

    Args:
        truth: Ground-truth manifest of the instance.
        frames: DataFrames by table name, as returned by `load`.
    """
    return Realized(
        dataset=truth.dataset,
        tier=truth.tier,
        seed=truth.seed,
        tables=tuple(_table(t, frames[t.name]) for t in truth.tables),
        foreign_keys=tuple(
            _foreign_key(fk.child_table, fk.child_columns, fk.parent_table, fk.parent_columns, frames)
            for fk in truth.foreign_keys
        ),
    )


def realized_statistics(
    name: str, tier: str = DEFAULT_TIER, seed: int = 0, spark: SparkSession | None = None
) -> Realized:
    """Returns the realized statistics of a catalog dataset instance, materializing it if needed."""
    return realized(materialize(name, tier, seed).manifest, load(name, tier, seed, spark))


def _table(table: Table, df: DataFrame) -> RealizedTable:
    rows = df.count()
    schema = {f.name: f.dataType for f in df.schema.fields}
    aggregates = []
    for c in table.columns:
        col = F.col(f"`{c.name}`")
        aggregates.append(F.count(col).alias(f"{c.name}__present"))
        if not isinstance(schema[c.name], T.MapType):
            aggregates.append(F.countDistinct(col).alias(f"{c.name}__distinct"))
        if not isinstance(schema[c.name], T.MapType | T.ArrayType | T.StructType):
            aggregates.append(F.min(col).cast("string").alias(f"{c.name}__min"))
            aggregates.append(F.max(col).cast("string").alias(f"{c.name}__max"))
    stats = df.agg(*aggregates).collect()[0].asDict() if aggregates else {}
    raw = collect(df, [c.name for c in table.columns])
    columns = []
    for c in table.columns:
        series = values_of(raw[c.name], schema[c.name])
        numeric = series.to_numpy(dtype=np.float64) if pd.api.types.is_numeric_dtype(series) and len(series) else None
        strings = series.astype(str) if c.formats else None
        categories = (
            c.distribution.params["values"] if c.distribution and c.distribution.family == Family.CATEGORICAL else []
        )
        counts = series.value_counts() if categories else pd.Series(dtype=np.float64)
        present = stats[f"{c.name}__present"]
        columns.append(
            RealizedColumn(
                name=c.name,
                null_rate=1 - present / rows if rows else 0.0,
                distinct=stats.get(f"{c.name}__distinct"),
                min=stats.get(f"{c.name}__min"),
                max=stats.get(f"{c.name}__max"),
                mean=None if numeric is None else float(numeric.mean()),
                std=None if numeric is None else float(numeric.std()),
                format_shares=(
                    {
                        f.regex: float(strings.map(lambda v, p=re.compile(f.regex): p.fullmatch(v) is not None).mean())
                        for f in c.formats
                    }
                    if strings is not None and len(strings)
                    else {}
                ),
                category_shares={str(v): float(counts.get(v, 0)) / max(len(series), 1) for v in categories},
            )
        )
    dependencies = []
    if table.dependencies:
        names = sorted({n for d in table.dependencies for n in d.columns})
        pairs_raw = collect(df, names)
        pairs = pd.DataFrame({n: values_of(pairs_raw[n], schema[n]) for n in names})
        for d in table.dependencies:
            aligned = pairs[list(d.columns)].dropna()
            value = dependency_value(d.measure, aligned[d.columns[0]], aligned[d.columns[1]])
            dependencies.append(RealizedDependency(columns=d.columns, measure=d.measure, value=value))
    quasi = [
        RealizedQuasiIdentifier(
            columns=qi.columns,
            unique_fraction=df.groupBy(*qi.columns).count().where("count = 1").count() / max(rows, 1),
        )
        for qi in table.quasi_identifiers
    ]
    return RealizedTable(
        name=table.name,
        row_count=rows,
        columns=tuple(columns),
        dependencies=tuple(dependencies),
        quasi_identifiers=tuple(quasi),
    )


def _foreign_key(
    child_table: str,
    child_columns: tuple[str, ...],
    parent_table: str,
    parent_columns: tuple[str, ...],
    frames: Mapping[str, DataFrame],
) -> RealizedForeignKey:
    child = frames[child_table].select(*child_columns).dropna()
    parent = frames[parent_table].select(
        *(F.col(p).alias(c) for p, c in zip(parent_columns, child_columns, strict=True))
    )
    total = child.count()
    missing = child.join(parent, list(child_columns), "left_anti").count()
    joined: pd.DataFrame = (
        parent.join(child.groupBy(*child_columns).count(), list(child_columns), "left")
        .fillna(0, subset=["count"])
        .toPandas()
    )
    counts = joined["count"].to_numpy(dtype=np.float64)
    return RealizedForeignKey(
        child_table=child_table,
        child_columns=child_columns,
        parent_table=parent_table,
        coverage=1.0 - missing / total if total else 1.0,
        fanout_mean=float(counts.mean()) if len(counts) else 0.0,
        fanout_max=int(counts.max()) if len(counts) else 0,
        fanout_zero_share=float((counts == 0).mean()) if len(counts) else 0.0,
    )
