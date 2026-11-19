"""Deterministic, FK-preserving input subsets for the data-volume RSI objective.

The harness, not the profiler, chooses which rows the profiler sees. Every table is
sampled independently at fraction ``f`` with a seeded hash of its primary key (or of
the full row content when there is no PK). The subset is then closed upward under
foreign keys: repeatedly add every parent row referenced by a selected child
(composite keys included) until a fixpoint. Self-references and cycles are handled
by the same closure; children are never thinned toward ``f**N``. The method name
recorded on every sample is ``fk_preserving_uniform``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from profiler_bench.catalog.manifest import ForeignKey, Manifest
from profiler_bench.fidelity.frames import HASH_RANGE, identifier


METHOD = "fk_preserving_uniform"
DEFAULT_FRACTIONS: tuple[float, ...] = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


@dataclass(frozen=True)
class SubsetResult:
    """One harness-controlled input subset.

    Attributes:
        tables: Subset frames keyed by table name.
        full_row_counts: Row counts of the full tables before subsetting.
        subset_row_counts: Row counts after subsetting.
        scales: ``full_rows / subset_rows`` per non-empty table (omitted when full is empty).
        fraction: Requested grid fraction ``f`` (indexes ``n*``).
        actual_fractions: Realized ``subset_rows / full_rows`` per table (0.0 when full is empty).
        overall_actual_fraction: ``sum(subset_rows) / sum(full_rows)`` across tables (0.0 if all empty).
        subset_seed: Seed that fixes the hash draws.
        method: Sampling method name recorded for the experiment.
    """

    tables: dict[str, DataFrame]
    full_row_counts: dict[str, int]
    subset_row_counts: dict[str, int]
    scales: dict[str, float]
    fraction: float
    actual_fractions: dict[str, float]
    overall_actual_fraction: float
    subset_seed: int
    method: str = METHOD

    def to_meta(self) -> dict[str, Any]:
        """Returns JSON-serializable subset metadata (no DataFrames)."""
        return {
            "fraction": self.fraction,
            "actual_fractions": dict(self.actual_fractions),
            "overall_actual_fraction": self.overall_actual_fraction,
            "subset_seed": self.subset_seed,
            "method": self.method,
            "full_row_counts": dict(self.full_row_counts),
            "subset_row_counts": dict(self.subset_row_counts),
            "scales": dict(self.scales),
        }


def subset_tables(
    frames: Mapping[str, DataFrame],
    manifest: Manifest,
    *,
    fraction: float,
    subset_seed: int,
) -> SubsetResult:
    """Returns a deterministic FK-preserving subset of ``frames`` at ``fraction``.

    Args:
        frames: Full real tables.
        manifest: Ground-truth manifest providing foreign keys.
        fraction: Target row fraction in ``(0, 1]``. ``1.0`` returns the inputs unchanged.
        subset_seed: Seed mixed into the per-row content/key hash.

    Returns:
        Subset frames plus per-table scales and realized input fractions.

    Raises:
        ValueError: If ``fraction`` is outside ``(0, 1]`` or a frame is missing.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    missing = [table.name for table in manifest.tables if table.name not in frames]
    if missing:
        raise ValueError(f"missing frames for tables: {missing}")

    full_counts = {name: int(frame.count()) for name, frame in frames.items()}
    if fraction == 1.0:
        return _result(
            tables=dict(frames),
            full_counts=full_counts,
            subset_counts=dict(full_counts),
            fraction=fraction,
            subset_seed=subset_seed,
        )

    tables_by_name = {table.name: table for table in manifest.tables}
    selected: dict[str, DataFrame] = {}
    for name, frame in frames.items():
        table = tables_by_name.get(name)
        key_columns = list(table.primary_key) if table is not None and table.primary_key else []
        selected[name] = _hash_sample(
            frame,
            fraction=fraction,
            seed=_table_seed(subset_seed, name),
            key_columns=key_columns,
        )

    selected = _upward_closure(frames, selected, manifest.foreign_keys)
    subset_counts = {name: int(frame.count()) for name, frame in selected.items()}
    return _result(
        tables=selected,
        full_counts=full_counts,
        subset_counts=subset_counts,
        fraction=fraction,
        subset_seed=subset_seed,
    )


def apply_table_scales(profile: dict[str, Any], scales: Mapping[str, float]) -> dict[str, Any]:
    """Returns a profile copy whose per-table ``row_count`` absorbs ``scales``.

    The reference sampler accepts a single scalar ``scale``. Multiplying each
    profiled ``row_count`` by that table's ``full_rows / profile_rows`` and then
    calling ``sample(..., scale=1.0)`` is equivalent to a per-table scale, so
    reconstructed row counts match the full real tables.
    """
    out = copy.deepcopy(profile)
    for table in out.get("tables", []):
        name = str(table.get("name"))
        if name not in scales:
            continue
        scale = float(scales[name])
        if not np.isfinite(scale) or scale < 0:
            raise ValueError(f"invalid scale for table {name}: {scale}")
        table["row_count"] = int(max(0, round(int(table["row_count"]) * scale)))
    return out


def _result(
    *,
    tables: dict[str, DataFrame],
    full_counts: Mapping[str, int],
    subset_counts: Mapping[str, int],
    fraction: float,
    subset_seed: int,
) -> SubsetResult:
    scales: dict[str, float] = {}
    actual: dict[str, float] = {}
    for name, full in full_counts.items():
        kept = int(subset_counts[name])
        if full > 0:
            if kept <= 0:
                raise RuntimeError(f"non-empty table {name} produced an empty subset")
            scales[name] = float(full) / float(kept)
            actual[name] = float(kept) / float(full)
        else:
            actual[name] = 0.0
    total_full = sum(int(value) for value in full_counts.values())
    total_kept = sum(int(value) for value in subset_counts.values())
    overall = (float(total_kept) / float(total_full)) if total_full > 0 else 0.0
    return SubsetResult(
        tables=tables,
        full_row_counts=dict(full_counts),
        subset_row_counts=dict(subset_counts),
        scales=scales,
        fraction=fraction,
        actual_fractions=actual,
        overall_actual_fraction=overall,
        subset_seed=subset_seed,
    )


def _hash_sample(
    frame: DataFrame,
    *,
    fraction: float,
    seed: int,
    key_columns: Sequence[str],
) -> DataFrame:
    """Keeps rows whose seeded key/content hash falls below ``fraction``, at least one if non-empty."""
    n = int(frame.count())
    if n == 0 or fraction >= 1.0:
        return frame
    hashed = frame.withColumn("__dv_h", _row_hash_column(frame, seed=seed, key_columns=key_columns))
    threshold = max(1, int(HASH_RANGE * fraction))
    sampled = hashed.where(F.pmod(F.col("__dv_h"), F.lit(HASH_RANGE)) < F.lit(threshold))
    if sampled.take(1):
        return sampled.drop("__dv_h")
    # Deterministic fallback: the row with the smallest hash (stable under repartition).
    return hashed.orderBy(F.col("__dv_h").asc()).limit(1).drop("__dv_h")


def _row_hash_column(frame: DataFrame, *, seed: int, key_columns: Sequence[str]) -> Column:
    """Returns a seeded 64-bit hash of the PK columns, or of the full row when there is no PK."""
    if key_columns:
        parts = ", ".join(identifier(column) for column in key_columns)
        return F.expr(f"xxhash64({int(seed)}, {parts})")
    columns = ", ".join(identifier(column) for column in frame.columns)
    return F.expr(f"xxhash64({int(seed)}, to_json(struct({columns})))")


def _upward_closure(
    full: Mapping[str, DataFrame],
    selected: Mapping[str, DataFrame],
    foreign_keys: Sequence[ForeignKey],
) -> dict[str, DataFrame]:
    """Adds every parent row referenced by a selected child until a fixpoint (bounded by sizes)."""
    current = dict(selected)
    if not foreign_keys:
        return current
    max_rows = sum(int(frame.count()) for frame in full.values())
    for _ in range(max(1, max_rows)):
        changed = False
        for fk in foreign_keys:
            if fk.child_table not in current or fk.parent_table not in full:
                continue
            child = current[fk.child_table]
            parent_full = full[fk.parent_table]
            parent_selected = current[fk.parent_table]
            child_keys = child.select(*list(fk.child_columns)).distinct()
            needed = _semi_join(
                parent_full,
                child_keys,
                list(fk.parent_columns),
                list(fk.child_columns),
            )
            new_rows = _anti_join(
                needed,
                parent_selected,
                list(fk.parent_columns),
                list(fk.parent_columns),
            )
            if not new_rows.take(1):
                continue
            current[fk.parent_table] = parent_selected.unionByName(new_rows)
            changed = True
        if not changed:
            return current
    raise RuntimeError("upward FK closure exceeded table-size bound")


def _semi_join(
    left: DataFrame,
    right_keys: DataFrame,
    left_columns: Sequence[str],
    right_columns: Sequence[str],
) -> DataFrame:
    """Keeps ``left`` rows whose key columns match a key in ``right_keys``."""
    renamed = right_keys
    aliases: list[str] = []
    for index, column in enumerate(right_columns):
        alias = f"__dv_k{index}"
        renamed = renamed.withColumnRenamed(column, alias)
        aliases.append(alias)
    cond: Column | None = None
    for left_col, alias in zip(left_columns, aliases, strict=True):
        piece = F.col(left_col).eqNullSafe(F.col(alias))
        cond = piece if cond is None else (cond & piece)
    assert cond is not None
    return left.join(renamed, on=cond, how="left_semi")


def _anti_join(
    left: DataFrame,
    right: DataFrame,
    left_columns: Sequence[str],
    right_columns: Sequence[str],
) -> DataFrame:
    """Keeps ``left`` rows whose key columns are absent from ``right``."""
    renamed = right.select(*list(right_columns))
    aliases: list[str] = []
    for index, column in enumerate(right_columns):
        alias = f"__dv_a{index}"
        renamed = renamed.withColumnRenamed(column, alias)
        aliases.append(alias)
    cond: Column | None = None
    for left_col, alias in zip(left_columns, aliases, strict=True):
        piece = F.col(left_col).eqNullSafe(F.col(alias))
        cond = piece if cond is None else (cond & piece)
    assert cond is not None
    return left.join(renamed, on=cond, how="left_anti")


def _table_seed(subset_seed: int, table_name: str) -> int:
    """Derives a stable per-table hash seed from the subset seed and table name."""
    material = f"{subset_seed}:{table_name}".encode()
    # Keep in signed 32-bit range for Spark integer literals in xxhash64.
    return int.from_bytes(material, "little") % (2**31 - 1)
