"""Adapter between the ground-truth catalog and the fidelity metrics.

The metrics read catalog concepts only through this module, so a catalog schema change is absorbed here:

- Each column has a privacy class (`direct_identifier`, `quasi_identifier`, `secret`, `free_text`) or none.
- Tables declare quasi-identifier sets: column combinations whose joint values can single out a row.
- A dataset instance has a structure seed (generating parameters and vocabularies, which the manifest depends on)
  and a sample seed (rows). The ceiling and the privacy holdout are other sample seeds of the same structure.
- Manifest values are generating parameters. Structure recovery compares a profile against them; reconstruction
  compares synthetic data against the realized real data and never reads distribution parameters.
"""

from __future__ import annotations

from collections.abc import Iterator

from pyspark.sql import DataFrame, SparkSession

from profiler_bench.catalog import load, materialize
from profiler_bench.catalog.manifest import Column, Manifest, Table


VALUE_LEAK_CATEGORIES = frozenset({"direct_identifier", "secret", "free_text"})


def load_instance(
    name: str, tier: str, *, structure_seed: int, sample_seed: int, spark: SparkSession
) -> dict[str, DataFrame]:
    """Returns the tables of one catalog instance as Spark DataFrames."""
    return load(name, tier, structure_seed, spark, sample_seed=sample_seed)


def manifest_of(name: str, tier: str, *, structure_seed: int) -> Manifest:
    """Returns the ground-truth manifest of a catalog dataset structure; it does not depend on the sample seed."""
    return materialize(name, tier, structure_seed).manifest


def sensitive_columns(table: Table) -> dict[str, str]:
    """Returns the privacy class of each sensitive top-level column or struct field, by dotted path."""
    return {path: column.privacy for path, column in _walk(table.columns, "") if column.privacy is not None}


def quasi_identifier_sets(table: Table) -> tuple[tuple[str, ...], ...]:
    """Returns the declared quasi-identifier column sets of `table`."""
    return tuple(qi.columns for qi in table.quasi_identifiers)


def _walk(columns: tuple[Column, ...], prefix: str) -> Iterator[tuple[str, Column]]:
    for column in columns:
        yield f"{prefix}{column.name}", column
        yield from _walk(column.fields, f"{prefix}{column.name}.")
