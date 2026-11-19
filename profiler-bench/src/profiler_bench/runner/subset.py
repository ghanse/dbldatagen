"""Deterministic, FK-preserving input subsets for the data-volume RSI objective.

The harness, not the profiler, chooses which rows the profiler sees. Root tables
(and tables with no foreign keys) take a seeded uniform row sample at fraction
``f``. Child tables keep only rows whose foreign keys land in the already-sampled
parent key sets, so referential integrity holds in the subset. The method name
recorded on every sample is ``fk_preserving_uniform``.
"""

from __future__ import annotations

import copy
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from profiler_bench.catalog.manifest import ForeignKey, Manifest


METHOD = "fk_preserving_uniform"
DEFAULT_FRACTIONS: tuple[float, ...] = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


@dataclass(frozen=True)
class SubsetResult:
    """One harness-controlled input subset.

    Attributes:
        tables: Subset frames keyed by table name.
        full_row_counts: Row counts of the full tables before subsetting.
        subset_row_counts: Row counts after subsetting.
        scales: ``full_rows / subset_rows`` per table (1.0 when the subset is empty).
        fraction: Requested input fraction.
        subset_seed: Seed that fixes the uniform draws.
        method: Sampling method name recorded for the experiment.
    """

    tables: dict[str, DataFrame]
    full_row_counts: dict[str, int]
    subset_row_counts: dict[str, int]
    scales: dict[str, float]
    fraction: float
    subset_seed: int
    method: str = METHOD

    def to_meta(self) -> dict[str, Any]:
        """Returns JSON-serializable subset metadata (no DataFrames)."""
        return {
            "fraction": self.fraction,
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
        subset_seed: Seed for the uniform parent draws.

    Returns:
        Subset frames plus per-table scales for the reference sampler.

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
        scales = dict.fromkeys(full_counts, 1.0)
        return SubsetResult(
            tables=dict(frames),
            full_row_counts=full_counts,
            subset_row_counts=dict(full_counts),
            scales=scales,
            fraction=fraction,
            subset_seed=subset_seed,
        )

    order = _table_order([table.name for table in manifest.tables], manifest.foreign_keys)
    parents_of = _parents_of(manifest.foreign_keys)
    selected_keys: dict[tuple[str, tuple[str, ...]], DataFrame] = {}
    subset: dict[str, DataFrame] = {}

    for table_name in order:
        frame = frames[table_name]
        parent_fks = parents_of.get(table_name, [])
        kept = frame
        for fk in parent_fks:
            parent_key = selected_keys.get((fk.parent_table, tuple(fk.parent_columns)))
            if parent_key is None:
                raise RuntimeError(f"parent keys missing for {fk.parent_table} before child {table_name}")
            kept = _semi_join(kept, parent_key, list(fk.child_columns), list(fk.parent_columns))
        if not parent_fks:
            kept = _uniform_sample(kept, fraction=fraction, seed=_table_seed(subset_seed, table_name))
        subset[table_name] = kept
        table_meta = next(table for table in manifest.tables if table.name == table_name)
        if table_meta.primary_key:
            selected_keys[(table_name, tuple(table_meta.primary_key))] = kept.select(
                *list(table_meta.primary_key)
            ).distinct()
        for unique in table_meta.unique_keys:
            selected_keys[(table_name, tuple(unique))] = kept.select(*list(unique)).distinct()

    for name in frames:
        if name not in subset:
            subset[name] = _uniform_sample(frames[name], fraction=fraction, seed=_table_seed(subset_seed, name))

    subset_counts = {name: int(frame.count()) for name, frame in subset.items()}
    scales = {
        name: (float(full_counts[name]) / float(subset_counts[name]) if subset_counts[name] > 0 else 1.0)
        for name in full_counts
    }
    return SubsetResult(
        tables=subset,
        full_row_counts=full_counts,
        subset_row_counts=subset_counts,
        scales=scales,
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


def _uniform_sample(frame: DataFrame, *, fraction: float, seed: int) -> DataFrame:
    """Takes a deterministic without-replacement sample of ``round(n * fraction)`` rows."""
    n = int(frame.count())
    if n == 0 or fraction >= 1.0:
        return frame
    k = max(1, round(n * fraction)) if fraction > 0 else 0
    if k >= n:
        return frame
    # Stable order + seeded permutation of row positions.
    ordered = frame.withColumn("__dv_row", F.monotonically_increasing_id())
    ids = [int(row[0]) for row in ordered.select("__dv_row").collect()]
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(ids, size=k, replace=False).tolist())
    return ordered.filter(F.col("__dv_row").isin(list(chosen))).drop("__dv_row")


def _semi_join(
    child: DataFrame,
    parent_keys: DataFrame,
    child_columns: Sequence[str],
    parent_columns: Sequence[str],
) -> DataFrame:
    """Keeps child rows whose key columns match a selected parent key."""
    renamed = parent_keys
    aliases: list[str] = []
    for index, column in enumerate(parent_columns):
        alias = f"__dv_p{index}"
        renamed = renamed.withColumnRenamed(column, alias)
        aliases.append(alias)
    cond: Column | None = None
    for child_col, alias in zip(child_columns, aliases, strict=True):
        piece = F.col(child_col).eqNullSafe(F.col(alias))
        cond = piece if cond is None else (cond & piece)
    assert cond is not None
    joined = child.join(renamed, on=cond, how="left_semi")
    return joined


def _parents_of(foreign_keys: Sequence[ForeignKey]) -> dict[str, list[ForeignKey]]:
    out: dict[str, list[ForeignKey]] = defaultdict(list)
    for fk in foreign_keys:
        out[fk.child_table].append(fk)
    return dict(out)


def _table_order(tables: Sequence[str], foreign_keys: Sequence[ForeignKey]) -> list[str]:
    """Topological order: parents before children. Isolated tables keep input order."""
    nodes = list(tables)
    indegree = dict.fromkeys(nodes, 0)
    edges: dict[str, list[str]] = defaultdict(list)
    for fk in foreign_keys:
        if fk.parent_table not in indegree or fk.child_table not in indegree:
            continue
        if fk.parent_table == fk.child_table:
            continue
        edges[fk.parent_table].append(fk.child_table)
        indegree[fk.child_table] += 1
    queue = deque([name for name in nodes if indegree[name] == 0])
    order: list[str] = []
    while queue:
        name = queue.popleft()
        order.append(name)
        for child in edges.get(name, []):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(nodes):
        # Cycle or missing edge bookkeeping: fall back to declaration order.
        return list(nodes)
    return order


def _table_seed(subset_seed: int, table_name: str) -> int:
    """Derives a stable per-table RNG seed from the subset seed and table name."""
    material = f"{subset_seed}:{table_name}".encode()
    # 31-bit positive seed for numpy Generator.
    return int.from_bytes(material, "little") % (2**31 - 1)
