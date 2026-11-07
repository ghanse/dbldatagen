"""Checks whether a profile's claims hold on the real data.

Structure recovery credits claims about planted structure against the manifest. Claims about structure the manifest
does not plant are checked here instead: a true claim is neutral and a false one counts as 0 in precision, so spraying
claims cannot raise precision and false claims lower it. Fractions are compared with the catalog's share tolerance,
keys must hold exactly, and dependencies use the structure dependency tolerance on a bounded sample.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pyspark.errors import AnalysisException
from pyspark.sql import Column as SparkColumn
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from dbldatagen_profiler.model import (
    CheckConstraint,
    ComparisonConstraint,
    Dependency,
    EnumerationConstraint,
    RangeConstraint,
    Relationship,
)
from profiler_bench.catalog.distributions import share
from profiler_bench.fidelity.config import MetricsConfig
from profiler_bench.fidelity.frames import ROW_FRAME, collect, identifier, leaf_kind, quoted
from profiler_bench.fidelity.reconstruction import pair_measure


class Verifier:
    """Evaluates claims against real tables.

    Args:
        real: Real DataFrames by table name.
        seed: Seed of the dependency samples.
        config: Metrics configuration.
    """

    def __init__(self, real: Mapping[str, DataFrame], seed: int, config: MetricsConfig) -> None:
        self.real = real
        self.seed = seed
        self.config = config
        self._rows: dict[str, int] = {}

    def rows(self, table: str) -> int:
        """Returns the row count of a real table."""
        if table not in self._rows:
            self._rows[table] = self.real[table].count()
        return self._rows[table]

    def _columns(self, table: str, columns: Sequence[str]) -> bool:
        return table in self.real and set(columns) <= set(self.real[table].columns)

    def key(self, table: str, columns: Sequence[str], primary: bool) -> bool:
        """Returns whether `columns` are distinct over the rows where they are all non-null (and none is null, for a
        primary key)."""
        if not self._columns(table, columns):
            return False
        df = self.real[table]
        complete = df.dropna(subset=list(columns))
        present = complete.count()
        if primary and present != self.rows(table):
            return False
        return complete.select(*(quoted(c) for c in columns)).distinct().count() == present

    def foreign_key(self, relationship: Relationship) -> bool:
        """Returns whether the real coverage of a relationship matches its claimed coverage."""
        child, parent = relationship.child_table, relationship.parent_table
        if not (
            self._columns(child, relationship.child_columns) and self._columns(parent, relationship.parent_columns)
        ):
            return False
        keys = list(relationship.child_columns)
        values = self.real[child].select(*(quoted(c) for c in keys)).dropna()
        targets = self.real[parent].select(
            *(quoted(p).alias(c) for p, c in zip(relationship.parent_columns, keys, strict=True))
        )
        total = values.count()
        coverage = 1.0 - values.join(targets, keys, "left_anti").count() / total if total else 1.0
        return not share("coverage", coverage, relationship.coverage, total)

    def check(self, table: str, claim: CheckConstraint) -> bool:
        """Returns whether the real share of rows satisfying a check matches its claimed `satisfied_fraction`."""
        if table not in self.real:
            return False
        df = self.real[table]
        try:
            violated = _violated(claim, df)
            row = df.agg(F.count(F.lit(1)).alias("n"), F.sum(violated.cast("long")).alias("bad")).collect()[0]
        except (AnalysisException, KeyError, ValueError):
            return False
        rows = row["n"]
        satisfied = 1.0 - (row["bad"] or 0) / rows if rows else 1.0
        return not share("satisfied", satisfied, claim.satisfied_fraction, rows)

    def dependency(self, table: str, dependency: Dependency) -> bool:
        """Returns whether a claimed dependency strength matches the real one on a bounded sample."""
        if not self._columns(table, dependency.columns):
            return False
        df = self.real[table].select(*(quoted(c) for c in dependency.columns))
        leaves, frames = collect(df, None, self.rows(table), self.config.sampling.max_rows, self.seed)
        frame = frames[ROW_FRAME]
        limit = self.config.marginal.categorical_max_distinct
        kinds = {leaf.path: leaf_kind(leaf, frame[leaf.path], limit) for leaf in leaves}
        if set(kinds) != set(dependency.columns):
            return False
        value = pair_measure(frame, dependency.columns, dependency.method.value, kinds, self.config)
        n = max(len(frame), 1)
        allowed = self.config.structure_tolerances.dependency + 4 * (1 - dependency.value**2) / math.sqrt(n)
        return bool(abs(value - dependency.value) <= allowed)


def _violated(claim: CheckConstraint, df: DataFrame) -> SparkColumn:
    if isinstance(claim, RangeConstraint):
        column, dtype = quoted(claim.column), df.schema[claim.column].dataType
        bad = F.lit(False)
        if claim.min is not None:
            bad = bad | (column < F.lit(claim.min).cast(dtype))
        if claim.max is not None:
            bad = bad | (column > F.lit(claim.max).cast(dtype))
        condition = bad
    elif isinstance(claim, EnumerationConstraint):
        dtype = df.schema[claim.column].dataType
        condition = ~quoted(claim.column).isin([F.lit(v).cast(dtype) for v in claim.values])
    elif isinstance(claim, ComparisonConstraint):
        condition = ~F.expr(f"{identifier(claim.left)} {claim.operator} {identifier(claim.right)}")
    else:
        condition = ~F.expr(claim.expression)
    return F.coalesce(condition, F.lit(False))
