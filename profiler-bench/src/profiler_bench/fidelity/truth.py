"""Adapter between the ground-truth manifest and the fidelity metrics.

The metrics read manifest concepts only through this module, so a manifest schema change is absorbed here:

- Sensitivity is a category per column (`direct_identifier`, `quasi_identifier`, `secret`, `free_text`). The manifest
  `sensitive` flag maps to `direct_identifier`.
- Quasi-identifier sets are column combinations whose joint values can single out a row.
- Manifest values are generating parameters. Structure recovery compares a profile against them; reconstruction
  compares synthetic data against the realized real data and never reads distribution parameters.
"""

from __future__ import annotations

from collections.abc import Iterator

from profiler_bench.catalog.manifest import Column, Table


SENSITIVE_CATEGORIES = ("direct_identifier", "quasi_identifier", "secret", "free_text")
VALUE_LEAK_CATEGORIES = frozenset({"direct_identifier", "secret", "free_text"})


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
