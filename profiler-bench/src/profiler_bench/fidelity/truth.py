"""Adapter between the ground-truth catalog and the fidelity metrics.

The metrics read catalog concepts only through this module, so a catalog schema change is absorbed here:

- Sensitivity is a category per column (`direct_identifier`, `quasi_identifier`, `secret`, `free_text`). The manifest
  `sensitive` flag maps to `direct_identifier`.
- Quasi-identifier sets are column combinations whose joint values can single out a row.
- A dataset instance has a structure seed (distribution parameters and vocabularies) and a sample seed (rows). The
  ceiling and the privacy holdout are other sample seeds of the same structure. While the catalog has a single
  `seed`, the sample seed is used as that seed, so other samples also change the structure.
- Manifest values are generating parameters. Structure recovery compares a profile against them; reconstruction
  compares synthetic data against the realized real data and never reads distribution parameters.
"""

from __future__ import annotations

from collections.abc import Iterator

from pyspark.sql import DataFrame, SparkSession

from profiler_bench.catalog import load, materialize
from profiler_bench.catalog.manifest import Column, Manifest, Table


SENSITIVE_CATEGORIES = ("direct_identifier", "quasi_identifier", "secret", "free_text")
VALUE_LEAK_CATEGORIES = frozenset({"direct_identifier", "secret", "free_text"})


def structure_seed(manifest: Manifest) -> int:
    """Returns the seed of the manifest's distribution parameters and vocabularies."""
    return int(getattr(manifest, "structure_seed", manifest.seed))


def sample_seed(manifest: Manifest) -> int:
    """Returns the seed of the manifest's rows."""
    return int(getattr(manifest, "sample_seed", manifest.seed))


def load_instance(
    name: str, tier: str, *, structure_seed: int, sample_seed: int, spark: SparkSession
) -> dict[str, DataFrame]:
    """Returns the tables of one catalog instance as Spark DataFrames."""
    return load(name, tier, sample_seed, spark)


def manifest_of(name: str, tier: str, *, structure_seed: int, sample_seed: int) -> Manifest:
    """Returns the ground-truth manifest of one catalog instance."""
    return materialize(name, tier, sample_seed).manifest


def sensitive_columns(table: Table) -> dict[str, str]:
    """Returns the sensitivity category of each sensitive top-level column or struct field, by dotted path."""
    return {path: category for path, column in _walk(table.columns, "") if (category := _category(column))}


def quasi_identifier_sets(table: Table) -> tuple[tuple[str, ...], ...]:
    """Returns the declared quasi-identifier column sets of `table`; empty when the manifest declares none."""
    return tuple(tuple(s) for s in getattr(table, "quasi_identifiers", ()))


def _category(column: Column) -> str | None:
    declared = getattr(column, "sensitivity", None)
    if declared is not None:
        return str(getattr(declared, "value", declared))
    return "direct_identifier" if column.sensitive else None


def _walk(columns: tuple[Column, ...], prefix: str) -> Iterator[tuple[str, Column]]:
    for column in columns:
        yield f"{prefix}{column.name}", column
        yield from _walk(column.fields, f"{prefix}{column.name}.")
