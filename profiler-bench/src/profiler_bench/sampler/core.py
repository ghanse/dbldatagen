"""Reference reconstruction of PySpark DataFrames from profile information."""

from __future__ import annotations

import hashlib
import math
import re
import string
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, localcontext
from itertools import pairwise
from typing import Any

import numpy as np
import pyarrow as pa
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as st

from dbldatagen_profiler.model import (
    Cardinality,
    ColumnProfile,
    ComparisonConstraint,
    DependenceMethod,
    EnumerationConstraint,
    ExpressionConstraint,
    LengthStats,
    LogicalType,
    Profile,
    RangeConstraint,
    Relationship,
    TableProfile,
    parse_temporal,
)

from .regex import UnsupportedRegex, generate_regex
from .report import ReportBuilder, SampleReport


@dataclass(frozen=True)
class SampleResult:
    """Contains reconstructed Spark tables and their sampler coverage report."""

    tables: dict[str, DataFrame]
    report: SampleReport


def sample(profile: Profile, spark: SparkSession, *, seed: int, scale: float = 1.0) -> SampleResult:
    """Reconstructs deterministic Spark DataFrames from a profile."""
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("scale must be a finite non-negative number")
    report = ReportBuilder()
    tables_by_name = {table.name: table for table in profile.tables}
    relationships = profile.relationships or []
    frames: dict[str, DataFrame] = {}
    values: dict[str, dict[str, list[Any]]] = {}
    for table_name in _table_order(profile.tables, relationships):
        table = tables_by_name[table_name]
        count = round(table.row_count * scale)
        candidate_count = count * 4 if _expression_constraints(table) else count
        columns = _generate_table(table, candidate_count, seed, report)
        _apply_constraints(table, columns, report)
        _apply_relationships(table, columns, values, relationships, seed, report)
        protected = {
            name
            for relationship in relationships
            if relationship.child_table == table.name
            for name in relationship.child_columns
        }
        _apply_keys(table, columns, report, protected)
        _audit_constraints(table, columns, report)
        rows = [
            {"__row_id": index, **{column.name: columns[column.name][index] for column in table.columns}}
            for index in range(candidate_count)
        ]
        arrow_schema = pa.schema(
            [pa.field("__row_id", pa.int64(), nullable=False), *[_arrow_field(c) for c in table.columns]]
        )
        arrow = pa.Table.from_pylist(rows, schema=arrow_schema)
        construction_schema = st.StructType(
            [st.StructField("__row_id", st.LongType(), False), *[_construction_field(c) for c in table.columns]]
        )
        frame = spark.createDataFrame(arrow, schema=construction_schema)
        frame = frame.select(
            F.col("__row_id"),
            *[
                (
                    F.parse_json(F.col(column.name)).alias(column.name)
                    if column.logical_type == LogicalType.VARIANT
                    else F.col(column.name)
                )
                for column in table.columns
            ],
        )
        frame = _apply_expressions(table, frame, count, report)
        frames[table_name] = frame
        if candidate_count == count:
            values[table_name] = columns
        else:
            sampled_rows = frame.collect()
            values[table_name] = {column.name: [row[column.name] for row in sampled_rows] for column in table.columns}
    return SampleResult(tables=frames, report=report.build())


def _rng(seed: int, *parts: str) -> np.random.Generator:
    digest = hashlib.sha256("\0".join((str(seed), *parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _table_order(tables: list[TableProfile], relationships: list[Relationship]) -> list[str]:
    names = [table.name for table in tables]
    edges: dict[str, set[str]] = defaultdict(set)
    incoming = dict.fromkeys(names, 0)
    for relationship in relationships:
        if relationship.child_table not in edges[relationship.parent_table]:
            edges[relationship.parent_table].add(relationship.child_table)
            incoming[relationship.child_table] += 1
    queue = deque(name for name in names if incoming[name] == 0)
    result: list[str] = []
    while queue:
        parent = queue.popleft()
        result.append(parent)
        for child in sorted(edges[parent]):
            incoming[child] -= 1
            if incoming[child] == 0:
                queue.append(child)
    return result + [name for name in names if name not in result]


def _generate_table(table: TableProfile, count: int, seed: int, report: ReportBuilder) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {}
    uniforms: dict[str, np.ndarray] = {}
    numeric_names = {column.name for column in table.columns if _is_ordered(column)}
    dependencies = [
        dependency
        for dependency in table.dependencies or []
        if dependency.method in (DependenceMethod.PEARSON, DependenceMethod.SPEARMAN)
        and set(dependency.columns) <= numeric_names
    ]
    if dependencies and count:
        involved = sorted({name for dependency in dependencies for name in dependency.columns})
        matrix = np.eye(len(involved))
        for dependency in dependencies:
            left, right = (involved.index(name) for name in dependency.columns)
            rho = dependency.value
            if dependency.method == DependenceMethod.SPEARMAN:
                rho = 2 * math.sin(math.pi * rho / 6)
            matrix[left, right] = matrix[right, left] = rho
        matrix, repaired = _nearest_psd(matrix)
        normals = _rng(seed, table.name, "copula").multivariate_normal(np.zeros(len(involved)), matrix, count)
        for index, name in enumerate(involved):
            uniforms[name] = _normal_cdf(normals[:, index])
            report.add(f"{table.name}.{name}", "used", "dependencies")
            report.add(f"{table.name}.{name}", "approximated", "dependencies.gaussian_copula")
            if repaired:
                report.add(f"{table.name}.{name}", "approximated", "dependencies.nearest_psd")
    for column in table.columns:
        path = f"{table.name}.{column.name}"
        columns[column.name] = _generate_column(
            column, count, _rng(seed, table.name, column.name), report, path, uniforms.get(column.name)
        )
    for dependency in table.dependencies or []:
        if dependency.method == DependenceMethod.CORRELATION_RATIO:
            for name in dependency.columns:
                report.add(f"{table.name}.{name}", "ignored", "dependencies.correlation_ratio")
    _apply_categorical_dependencies(table, columns, seed, report)
    return columns


def _generate_column(
    column: ColumnProfile,
    count: int,
    rng: np.random.Generator,
    report: ReportBuilder,
    path: str,
    uniforms: np.ndarray | None = None,
) -> list[Any]:
    report.add(path, "used", "logical_type", "nullable", "null_fraction")
    report.add(path, "used" if column.logical_type == LogicalType.INTERVAL else "ignored", "source_type")
    if column.value_count is not None:
        report.add(path, "ignored", "value_count")
    values: list[Any]
    if column.logical_type == LogicalType.STRUCT:
        children = {
            child.name: _generate_column(child, count, rng, report, f"{path}.{child.name}")
            for child in column.children or []
        }
        values = [{name: child_values[i] for name, child_values in children.items()} for i in range(count)]
    elif column.logical_type in (LogicalType.ARRAY, LogicalType.MAP):
        values = _generate_collection(column, count, rng, report, path)
    else:
        null_count = min(count, round(column.null_fraction * count))
        indexes = {int(index) for index in rng.permutation(count)[:null_count]}
        present = [index for index in range(count) if index not in indexes]
        present_uniforms = uniforms[present] if uniforms is not None else None
        generated = _generate_scalar(column, len(present), rng, report, path, present_uniforms)
        iterator = iter(generated)
        values = [None if index in indexes else next(iterator) for index in range(count)]
        return values
    if column.null_fraction and count:
        null_count = min(count, round(column.null_fraction * count))
        for index in rng.permutation(count)[:null_count]:
            values[int(index)] = None
    return values


def _generate_scalar(
    column: ColumnProfile,
    count: int,
    rng: np.random.Generator,
    report: ReportBuilder,
    path: str,
    uniforms: np.ndarray | None,
) -> list[Any]:
    kind = column.logical_type
    if kind == LogicalType.NULL:
        return [None] * count
    if kind == LogicalType.BOOLEAN:
        fraction = column.boolean.true_fraction if column.boolean else 0.5
        report.add(path, "used" if column.boolean else "approximated", "boolean.true_fraction")
        return [bool(value) for value in rng.random(count) < fraction]
    if kind in _ORDERED_TYPES:
        values = _ordered_values(column, count, rng, report, path, uniforms)
    elif kind == LogicalType.STRING:
        values = _string_values(column, count, rng, report, path)
    elif kind == LogicalType.BINARY:
        values = [rng.bytes(8) for _ in range(count)]
        report.add(path, "approximated", "type_default")
    elif kind == LogicalType.INTERVAL:
        year_month = "year" in column.source_type.lower() or "month" in column.source_type.lower()
        values = [0 if year_month else timedelta(0)] * count
        report.add(path, "approximated", "type_default")
    elif kind == LogicalType.VARIANT:
        values = ["{}"] * count
        report.add(path, "approximated", "type_default")
    else:
        values = [None] * count
        report.add(path, "unmet", "logical_type")
    if kind == LogicalType.STRING and column.string and column.string.top_values:
        return values
    return _apply_pool(column, values, rng, report, path, uniforms)


def _ordered_values(  # noqa: PLR0912, PLR0915
    column: ColumnProfile,
    count: int,
    rng: np.random.Generator,
    report: ReportBuilder,
    path: str,
    uniforms: np.ndarray | None,
) -> list[Any]:
    stats = column.numeric or column.temporal
    u = _open_unit(rng.random(count) if uniforms is None else uniforms)
    if column.logical_type in _INTEGER_TYPES | {LogicalType.DECIMAL}:
        return _exact_ordered_values(column, count, rng, report, path, u)
    endpoints: tuple[float, float] | None = None
    if column.numeric and column.numeric.histogram:
        histogram = column.numeric.histogram
        weights = np.asarray(histogram.counts, dtype=float)
        bins = np.minimum(np.searchsorted(np.cumsum(weights / weights.sum()), u, side="right"), len(weights) - 1)
        low = np.asarray([_number(histogram.edges[index], column) for index in bins], dtype=float)
        high = np.asarray([_number(histogram.edges[index + 1], column) for index in bins], dtype=float)
        raw = _open_interval_between(low, high, _open_unit(rng.random(count)))
        if histogram.edges:
            endpoints = (_number(histogram.edges[0], column), _number(histogram.edges[-1], column))
        report.add(path, "used", "numeric.histogram")
        for field in ("quantiles", "min", "max", "mean", "stddev"):
            if getattr(column.numeric, field) is not None:
                report.add(path, "ignored", f"numeric.{field}")
    elif stats and stats.quantiles:
        quantiles = sorted(stats.quantiles, key=lambda item: item.q)
        xp = np.asarray([item.q for item in quantiles], dtype=float)
        fp = np.asarray([_number(item.value, column) for item in quantiles], dtype=float)
        raw = np.interp(u, xp, fp)
        endpoints = (float(fp[0]), float(fp[-1])) if len(fp) else None
        report.add(path, "used", f"{'numeric' if column.numeric else 'temporal'}.quantiles")
        prefix = "numeric" if column.numeric else "temporal"
        for field in ("min", "max", "mean", "stddev") if column.numeric else ("min", "max"):
            if getattr(stats, field) is not None:
                report.add(path, "ignored", f"{prefix}.{field}")
    elif stats and stats.min is not None and stats.max is not None:
        bound_low, bound_high = _number(stats.min, column), _number(stats.max, column)
        endpoints = (bound_low, bound_high)
        if column.numeric and column.numeric.mean is not None and column.numeric.stddev is not None:
            mu, sigma = _calibrate_truncated_parameters(
                float(column.numeric.mean), float(column.numeric.stddev), bound_low, bound_high
            )
            raw = _truncated_normal(u, mu, sigma, bound_low, bound_high, open_bounds=True)
            report.add(path, "used", "numeric.min", "numeric.max", "numeric.mean", "numeric.stddev")
            report.add(path, "approximated", "numeric.truncated_normal")
        else:
            # Missing stddev (or mean): fall back to open uniform on [min, max].
            raw = _open_interval_between(bound_low, bound_high, u)
            prefix = "numeric" if column.numeric else "temporal"
            report.add(path, "used", f"{prefix}.min", f"{prefix}.max")
            report.add(path, "approximated", f"{prefix}.uniform")
    else:
        raw = np.arange(count, dtype=float) if column.cardinality == Cardinality.UNIQUE else np.zeros(count)
        report.add(path, "approximated", "type_default")
    if column.logical_type in (LogicalType.FLOAT32, LogicalType.FLOAT64) and column.numeric:
        special = [
            float(value)
            for value in (column.numeric.min, column.numeric.max)
            if value is not None and not math.isfinite(float(value))
        ]
        raw = np.nan_to_num(raw, nan=column.numeric.mean or 0.0, posinf=0.0, neginf=0.0)
        for index, value in enumerate(special[:count]):
            raw[index] = value
        if special:
            report.add(path, "used", "numeric.non_finite")
        elif endpoints is not None:
            raw = _away_from_endpoints(raw, endpoints[0], endpoints[1])
    elif column.logical_type in _TEMPORAL_TYPES and endpoints is not None:
        raw = _away_from_endpoints(raw, endpoints[0], endpoints[1])
    return [_convert_ordered(value, column) for value in raw]


def _exact_ordered_values(
    column: ColumnProfile,
    count: int,
    rng: np.random.Generator,
    report: ReportBuilder,
    path: str,
    uniforms: np.ndarray,
) -> list[Any]:
    stats = column.numeric
    if stats and stats.histogram:
        histogram = stats.histogram
        weights = np.asarray(histogram.counts, dtype=float)
        bins = np.minimum(np.searchsorted(np.cumsum(weights / weights.sum()), uniforms, side="right"), len(weights) - 1)
        values = [
            _exact_between(
                _exact_number(histogram.edges[index], column),
                _exact_number(histogram.edges[index + 1], column),
                float(rng.random()),
                column,
                closed=index == len(histogram.counts) - 1,
            )
            for index in bins
        ]
        report.add(path, "used", "numeric.histogram")
        for field in ("quantiles", "min", "max", "mean", "stddev"):
            if getattr(stats, field) is not None:
                report.add(path, "ignored", f"numeric.{field}")
        return values
    if stats and stats.quantiles:
        quantiles = sorted(stats.quantiles, key=lambda item: item.q)
        values = [_exact_quantile(value, quantiles, column) for value in uniforms]
        report.add(path, "used", "numeric.quantiles")
        for field in ("min", "max", "mean", "stddev"):
            if getattr(stats, field) is not None:
                report.add(path, "ignored", f"numeric.{field}")
        return values
    if stats and stats.min is not None and stats.max is not None:
        low = _exact_number(stats.min, column)
        high = _exact_number(stats.max, column)
        if stats.mean is not None and stats.stddev is not None:
            # Truncated shape in float/z-space; map onto the exact int/Decimal domain so ranges
            # above 2^53 and high-precision decimals do not collapse into endpoint atoms.
            values = _exact_truncated_normal(uniforms, stats.mean, stats.stddev, low, high, column)
            report.add(path, "used", "numeric.min", "numeric.max", "numeric.mean", "numeric.stddev")
            report.add(path, "approximated", "numeric.truncated_normal")
            return values
        # Missing stddev: fall back to exact discrete/quantum-uniform on [min, max].
        report.add(path, "used", "numeric.min", "numeric.max")
        report.add(path, "approximated", "numeric.uniform")
        return [_exact_between(low, high, float(value), column, closed=True) for value in uniforms]
    report.add(path, "approximated", "type_default")
    if column.cardinality == Cardinality.UNIQUE:
        return [_convert_exact(Decimal(index), column) for index in range(count)]
    return [_convert_exact(Decimal(0), column)] * count


def _exact_quantile(value: float, quantiles: list[Any], column: ColumnProfile) -> Any:  # noqa: ANN401
    if value <= quantiles[0].q:
        return _convert_exact(_exact_number(quantiles[0].value, column), column)
    if value >= quantiles[-1].q:
        return _convert_exact(_exact_number(quantiles[-1].value, column), column)
    for left, right in pairwise(quantiles):
        if left.q <= value <= right.q:
            fraction = Decimal(str((value - left.q) / (right.q - left.q)))
            low = _exact_number(left.value, column)
            high = _exact_number(right.value, column)
            return _convert_exact(low + fraction * (high - low), column)
    raise AssertionError("quantile interval not found")


def _exact_number(value: Any, column: ColumnProfile) -> Decimal:  # noqa: ANN401
    literal = _literal(value, column)
    return literal if isinstance(literal, Decimal) else Decimal(literal)


def _exact_between(
    low: Decimal,
    high: Decimal,
    uniform: float,
    column: ColumnProfile,
    *,
    closed: bool,
) -> Any:  # noqa: ANN401
    if column.logical_type in _INTEGER_TYPES:
        lower, upper = int(low), int(high)
        width = upper - lower + int(closed)
        return lower if width <= 0 else lower + min(width - 1, int(uniform * width))
    quantum = Decimal(1).scaleb(-(column.scale or 0))
    lower, upper = int(low / quantum), int(high / quantum)
    width = upper - lower + int(closed)
    units = lower if width <= 0 else lower + min(width - 1, int(uniform * width))
    return _convert_exact(Decimal(units) * quantum, column)


def _convert_exact(value: Decimal, column: ColumnProfile) -> int | Decimal:
    if column.logical_type in _INTEGER_TYPES:
        return round(value)
    quantum = Decimal(1).scaleb(-(column.scale or 0))
    limit = Decimal(10) ** ((column.precision or 38) - (column.scale or 0)) - quantum
    return min(limit, max(-limit, value)).quantize(quantum)


def _string_values(
    column: ColumnProfile, count: int, rng: np.random.Generator, report: ReportBuilder, path: str
) -> list[str]:
    stats = column.string
    result: list[str] = []
    top_values = stats.top_values if stats and stats.top_values else []
    top_counts = [round(count * item.frequency) for item in top_values]
    for item, item_count in zip(top_values, top_counts, strict=True):
        result.extend([item.value] * item_count)
    if top_values:
        report.add(path, "used", "string.top_values")
    remaining = max(0, count - len(result))
    patterns = stats.patterns if stats and stats.patterns else []
    weights = np.asarray([pattern.coverage for pattern in patterns], dtype=float)
    if weights.size and weights.sum() > 0:
        weights /= weights.sum()
        report.add(path, "used", "string.patterns")
    length = stats.length if stats else None
    if length:
        report.add(path, "used", "string.length")
    for _index in range(remaining):
        target_length = _draw_length(length, rng)
        if patterns:
            pattern = patterns[int(rng.choice(len(patterns), p=weights))].regex
            try:
                max_length = length.max if length and length.max is not None else target_length
                result.append(generate_regex(pattern, rng, max_length=max(max_length, 1)))
                continue
            except (UnsupportedRegex, re.error):
                report.add(path, "unmet", f"string.patterns:{pattern}")
        result.append("".join(rng.choice(list(string.ascii_lowercase), max(target_length, 1))))
    if remaining and not patterns:
        report.add(path, "approximated", "string.random_fallback")
    if column.distinct_count and remaining:
        target_other = max(1, min(remaining, column.distinct_count.value - len(top_values)))
        generated = result[len(result) - remaining :]
        pool = list(dict.fromkeys(generated))[:target_other]
        while len(pool) < target_other:
            pool.append(f"synthetic_{len(pool)}")
        result[len(result) - remaining :] = [pool[int(index)] for index in rng.integers(0, len(pool), remaining)]
        report.add(path, "used", "distinct_count")
    rng.shuffle(result)
    return result[:count]


def _draw_length(stats: LengthStats | None, rng: np.random.Generator) -> int:
    if stats is None:
        return 12
    low = stats.min if stats.min is not None else 0
    high = stats.max if stats.max is not None else max(low, round((stats.mean or 12) * 2))
    if stats.mean is not None and stats.stddev is not None:
        return int(np.clip(round(rng.normal(stats.mean, stats.stddev)), low, high))
    return int(rng.integers(low, high + 1)) if high > low else int(low)


def _generate_collection(
    column: ColumnProfile, count: int, rng: np.random.Generator, report: ReportBuilder, path: str
) -> list[Any]:
    lengths = [_draw_length(column.collection_length, rng) for _ in range(count)]
    report.add(path, "used" if column.collection_length else "approximated", "collection_length")
    total = sum(lengths)
    children = column.children or []
    if column.logical_type == LogicalType.ARRAY:
        elements = _generate_column(children[0], total, rng, report, f"{path}.element")
        offsets = np.cumsum([0, *lengths])
        return [elements[offsets[i] : offsets[i + 1]] for i in range(count)]
    keys = _generate_column(children[0], total, rng, report, f"{path}.key")
    values = _generate_column(children[1], total, rng, report, f"{path}.value")
    for index, key in enumerate(keys):
        if key is None:
            keys[index] = _map_key(index, children[0], keys)
            report.add(f"{path}.key", "approximated", "map.null_key_redraw")
    offsets = np.cumsum([0, *lengths])
    result = []
    for index in range(count):
        pairs = zip(
            keys[offsets[index] : offsets[index + 1]],
            values[offsets[index] : offsets[index + 1]],
            strict=True,
        )
        result.append({key: value for key, value in pairs if key is not None})
    return result


def _map_key(index: int, column: ColumnProfile, existing: list[Any]) -> Any:  # noqa: ANN401
    if column.logical_type == LogicalType.BOOLEAN:
        return bool(index % 2)
    if column.logical_type == LogicalType.BINARY:
        return index.to_bytes(8, "little")
    return _unique_value(None, index + len(existing), column, existing)


def _apply_pool(
    column: ColumnProfile,
    values: list[Any],
    rng: np.random.Generator,
    report: ReportBuilder,
    path: str,
    uniforms: np.ndarray | None,
) -> list[Any]:
    if column.cardinality == Cardinality.UNIQUE:
        return values
    size = column.distinct_count.value if column.distinct_count else _bucket_size(column.cardinality, len(values))
    if size is None or size <= 0 or not values:
        return values
    ordered = sorted(values)
    pool_size = min(size, len(ordered))
    positions = np.linspace(0, len(ordered) - 1, pool_size, dtype=int)
    pool = [ordered[index] for index in positions]
    report.add(path, "used", "distinct_count" if column.distinct_count else "cardinality")
    if column.distinct_count and not column.distinct_count.exact:
        report.add(path, "approximated", "distinct_count.approximate")
    draws = rng.random(len(values)) if uniforms is None else uniforms
    indexes = np.minimum((np.clip(draws, 0, np.nextafter(1.0, 0.0)) * len(pool)).astype(int), len(pool) - 1)
    return [pool[index] for index in indexes]


def _bucket_size(cardinality: Cardinality | None, count: int) -> int | None:
    return {
        Cardinality.CONSTANT: 1,
        Cardinality.LOW: min(100, max(1, count // 20)),
        Cardinality.MEDIUM: min(max(101, count // 20), max(1, count)),
        Cardinality.HIGH: min(max(1, count // 2), count),
    }.get(cardinality)


def _apply_relationships(
    table: TableProfile,
    columns: dict[str, list[Any]],
    parent_values: dict[str, dict[str, list[Any]]],
    relationships: list[Relationship],
    seed: int,
    report: ReportBuilder,
) -> None:
    for relationship in relationships:
        if relationship.child_table != table.name or relationship.parent_table not in parent_values:
            continue
        rng = _rng(seed, table.name, "fk", ",".join(relationship.child_columns))
        parents = list(
            dict.fromkeys(
                zip(
                    *(parent_values[relationship.parent_table][name] for name in relationship.parent_columns),
                    strict=True,
                )
            )
        )
        non_null = [
            index
            for index, values in enumerate(zip(*(columns[name] for name in relationship.child_columns), strict=True))
            if all(value is not None for value in values)
        ]
        covered = min(len(non_null), round(len(non_null) * relationship.coverage))
        positions = rng.permutation(non_null).tolist()
        covered_positions = positions[:covered]
        uncovered_positions = positions[covered:]
        indexes, fanout_used = _fanout_indexes(covered, len(parents), relationship, rng)
        if not parents:
            for child_name in relationship.child_columns:
                report.add(f"{table.name}.{child_name}", "unmet", "relationship.parent_key")
            continue
        for child_index, child_name in enumerate(relationship.child_columns):
            for position, parent in zip(covered_positions, indexes, strict=True):
                columns[child_name][position] = parents[parent][child_index]
            path = f"{table.name}.{child_name}"
            report.add(path, "used", "relationship.coverage", "relationship.parent_key")
            if relationship.fan_out:
                report.add(path, "used" if fanout_used else "ignored", "relationship.fan_out")
                if fanout_used:
                    report.add(path, "approximated", "relationship.fan_out.discretized")
        _exclude_parent_keys(table, columns, relationship, parents, uncovered_positions)


def _exclude_parent_keys(
    table: TableProfile,
    columns: dict[str, list[Any]],
    relationship: Relationship,
    parents: list[tuple[Any, ...]],
    uncovered: list[int],
) -> None:
    parent_set = set(parents)
    count = len(next(iter(columns.values()), []))
    last_name = relationship.child_columns[-1]
    last_column = next(column for column in table.columns if column.name == last_name)
    for index in uncovered:
        key = tuple(columns[name][index] for name in relationship.child_columns)
        if key in parent_set:
            columns[last_name][index] = _unique_value(
                columns[last_name][index], index + count, last_column, columns[last_name]
            )


def _fanout_indexes(
    count: int, parent_count: int, relationship: Relationship, rng: np.random.Generator
) -> tuple[np.ndarray, bool]:
    if parent_count == 0:
        return np.zeros(count, dtype=int), False
    stats = relationship.fan_out
    if stats is None:
        return rng.integers(0, parent_count, count), False
    if not stats.min * parent_count <= count <= stats.max * parent_count:
        return rng.integers(0, parent_count, count), False
    if stats.quantiles:
        quantiles = sorted(stats.quantiles, key=lambda item: item.q)
        raw = np.interp(
            rng.random(parent_count),
            [item.q for item in quantiles],
            [int(item.value) for item in quantiles],
        )
    elif stats.stddev is not None:
        raw = rng.normal(stats.mean, stats.stddev, parent_count)
    else:
        raw = np.full(parent_count, stats.mean)
    counts = np.clip(np.rint(raw), stats.min, stats.max).astype(int)
    _adjust_counts(counts, count, stats.min, stats.max, rng)
    indexes = np.repeat(np.arange(parent_count), counts)
    rng.shuffle(indexes)
    return indexes, True


def _adjust_counts(counts: np.ndarray, target: int, minimum: int, maximum: int, rng: np.random.Generator) -> None:
    delta = target - int(counts.sum())
    direction = 1 if delta > 0 else -1
    candidates = np.flatnonzero(counts < maximum if direction > 0 else counts > minimum)
    cursor = 0
    while delta and len(candidates):
        index = int(candidates[cursor % len(candidates)])
        counts[index] += direction
        delta -= direction
        cursor += 1
        if cursor % len(candidates) == 0:
            candidates = np.flatnonzero(counts < maximum if direction > 0 else counts > minimum)
            rng.shuffle(candidates)


def _apply_keys(table: TableProfile, columns: dict[str, list[Any]], report: ReportBuilder, protected: set[str]) -> None:
    keys = ([table.primary_key] if table.primary_key else []) + (table.unique_keys or [])
    count = len(next(iter(columns.values()), []))
    for key in keys:
        seen: set[tuple[Any, ...]] = set()
        unmet = False
        for index in range(count):
            value = tuple(columns[name][index] for name in key.columns)
            needs_value = key is table.primary_key and any(item is None for item in value)
            duplicate = value in seen and not (key is not table.primary_key and any(item is None for item in value))
            if needs_value or duplicate:
                mutable = [name for name in key.columns if name not in protected]
                if not mutable:
                    unmet = True
                    for name in key.columns:
                        report.add(
                            f"{table.name}.{name}",
                            "unmet",
                            "primary_key" if key is table.primary_key else "unique_keys",
                        )
                    continue
                last = mutable[-1]
                column = next(column for column in table.columns if column.name == last)
                columns[last][index] = _unique_value(columns[last][index], index, column, columns[last])
                value = tuple(columns[name][index] for name in key.columns)
            seen.add(value)
        for name in key.columns:
            if not unmet:
                report.add(f"{table.name}.{name}", "used", "primary_key" if key is table.primary_key else "unique_keys")


def _unique_value(value: Any, index: int, column: ColumnProfile, existing: list[Any]) -> Any:  # noqa: ANN401, PLR0911
    kind = column.logical_type
    if kind == LogicalType.STRING:
        return f"{value or ''}_{index}"
    if kind in _INTEGER_TYPES:
        present = [item for item in existing if item is not None]
        return (max(present) if present else 0) + 1
    if kind in (LogicalType.FLOAT32, LogicalType.FLOAT64):
        present = [item for item in existing if item is not None]
        return float(np.nextafter(max(present) if present else 0.0, math.inf))
    if kind == LogicalType.DECIMAL:
        present = [item for item in existing if item is not None]
        base = max(present) if present else Decimal(0)
        return (base + Decimal(1).scaleb(-(column.scale or 0))).quantize(Decimal(1).scaleb(-(column.scale or 0)))
    if kind == LogicalType.DATE:
        present = [item for item in existing if item is not None]
        return (max(present) if present else date(1970, 1, 1)) + timedelta(days=1)
    if kind in (LogicalType.TIMESTAMP, LogicalType.TIMESTAMP_NTZ):
        present = [item for item in existing if item is not None]
        return (max(present) if present else datetime(1970, 1, 1)) + timedelta(microseconds=1)
    return f"{value}_{index}"


def _apply_constraints(table: TableProfile, columns: dict[str, list[Any]], report: ReportBuilder) -> None:
    for constraint in table.check_constraints or []:
        if isinstance(constraint, RangeConstraint):
            column = next(column for column in table.columns if column.name == constraint.column)
            low = _literal(constraint.min, column) if constraint.min is not None else None
            high = _literal(constraint.max, column) if constraint.max is not None else None
            columns[constraint.column] = [
                (
                    value
                    if value is None
                    else (
                        max(low, value)
                        if high is None
                        else min(high, value) if low is None else min(high, max(low, value))
                    )
                )
                for value in columns[constraint.column]
            ]
            report.add(f"{table.name}.{constraint.column}", "used", "check_constraints.range")
        elif isinstance(constraint, EnumerationConstraint):
            column = next(column for column in table.columns if column.name == constraint.column)
            allowed = [_literal(value, column) for value in constraint.values]
            columns[constraint.column] = [
                value if value is None else allowed[index % len(allowed)]
                for index, value in enumerate(columns[constraint.column])
            ]
            report.add(f"{table.name}.{constraint.column}", "used", "check_constraints.enumeration")
        elif isinstance(constraint, ComparisonConstraint):
            _enforce_comparison(table, columns, constraint)
            report.add(f"{table.name}.{constraint.left}", "used", "check_constraints.comparison")
            report.add(f"{table.name}.{constraint.right}", "used", "check_constraints.comparison")


def _audit_constraints(table: TableProfile, columns: dict[str, list[Any]], report: ReportBuilder) -> None:
    operators = {
        "<": lambda left, right: left < right,
        "<=": lambda left, right: left <= right,
        "=": lambda left, right: left == right,
        "!=": lambda left, right: left != right,
        ">=": lambda left, right: left >= right,
        ">": lambda left, right: left > right,
    }
    for constraint in table.check_constraints or []:
        failed = False
        names: list[str] = []
        if isinstance(constraint, RangeConstraint):
            names = [constraint.column]
            column = next(column for column in table.columns if column.name == constraint.column)
            low = _literal(constraint.min, column) if constraint.min is not None else None
            high = _literal(constraint.max, column) if constraint.max is not None else None
            failed = any(
                value is not None and ((low is not None and value < low) or (high is not None and value > high))
                for value in columns[constraint.column]
            )
        elif isinstance(constraint, EnumerationConstraint):
            names = [constraint.column]
            column = next(column for column in table.columns if column.name == constraint.column)
            allowed = {_literal(value, column) for value in constraint.values}
            failed = any(value is not None and value not in allowed for value in columns[constraint.column])
        elif isinstance(constraint, ComparisonConstraint):
            names = [constraint.left, constraint.right]
            pairs = zip(columns[constraint.left], columns[constraint.right], strict=True)
            failed = any(
                left is not None and right is not None and not operators[constraint.operator](*_comparable(left, right))
                for left, right in pairs
            )
        if failed:
            label = constraint.kind
            report.unmet(table.name, label)
            for name in names:
                report.add(f"{table.name}.{name}", "unmet", f"check_constraints.{label}")


def _enforce_comparison(table: TableProfile, columns: dict[str, list[Any]], constraint: ComparisonConstraint) -> None:
    operators = {
        "<": lambda left, right: left < right,
        "<=": lambda left, right: left <= right,
        "=": lambda left, right: left == right,
        "!=": lambda left, right: left != right,
        ">=": lambda left, right: left >= right,
        ">": lambda left, right: left > right,
    }
    pairs = zip(columns[constraint.left], columns[constraint.right], strict=True)
    for index, (left, right) in enumerate(pairs):
        comparable_left, comparable_right = _comparable(left, right)
        if (
            left is not None
            and right is not None
            and not operators[constraint.operator](comparable_left, comparable_right)
        ):
            if constraint.operator in ("<", "<="):
                repaired = _coerce_like(left, right)
                right_column = next(column for column in table.columns if column.name == constraint.right)
                columns[constraint.right][index] = (
                    _next_value(repaired, right_column, 1) if constraint.operator == "<" else repaired
                )
            elif constraint.operator in (">", ">="):
                repaired = _coerce_like(right, left)
                left_column = next(column for column in table.columns if column.name == constraint.left)
                columns[constraint.left][index] = (
                    _next_value(repaired, left_column, 1) if constraint.operator == ">" else repaired
                )
            elif constraint.operator == "=":
                columns[constraint.right][index] = left
            else:
                columns[constraint.right][index] = _different(right)


def _next_value(value: Any, column: ColumnProfile, direction: int) -> Any:  # noqa: ANN401
    kind = column.logical_type
    if kind in _INTEGER_TYPES:
        return value + direction
    if kind == LogicalType.DECIMAL:
        return value + Decimal(1).scaleb(-(column.scale or 0)) * direction
    if kind in (LogicalType.FLOAT32, LogicalType.FLOAT64):
        target = math.inf if direction > 0 else -math.inf
        return float(np.nextafter(value, target))
    if kind == LogicalType.DATE:
        return value + timedelta(days=direction)
    if kind in (LogicalType.TIMESTAMP, LogicalType.TIMESTAMP_NTZ):
        return value + timedelta(microseconds=direction)
    return _different(value)


def _comparable(left: Any, right: Any) -> tuple[Any, Any]:  # noqa: ANN401
    if isinstance(left, datetime) and isinstance(right, date) and not isinstance(right, datetime):
        return left.date(), right
    if isinstance(right, datetime) and isinstance(left, date) and not isinstance(left, datetime):
        return left, right.date()
    return left, right


def _coerce_like(value: Any, template: Any) -> Any:  # noqa: ANN401
    if isinstance(template, datetime) and isinstance(value, date) and not isinstance(value, datetime):
        return datetime.combine(value, time())
    if isinstance(template, date) and not isinstance(template, datetime) and isinstance(value, datetime):
        return value.date()
    return value


def _different(value: Any) -> Any:  # noqa: ANN401
    if isinstance(value, str):
        return value + "_"
    if isinstance(value, (int, float, Decimal)):
        return value + 1
    if isinstance(value, date):
        return value + timedelta(days=1)
    return None


def _expression_constraints(table: TableProfile) -> list[ExpressionConstraint]:
    return [constraint for constraint in table.check_constraints or [] if isinstance(constraint, ExpressionConstraint)]


def _apply_expressions(table: TableProfile, frame: DataFrame, count: int, report: ReportBuilder) -> DataFrame:
    constraints = _expression_constraints(table)
    if not constraints:
        return frame.orderBy("__row_id").drop("__row_id")
    filtered = frame
    failed: list[ExpressionConstraint] = []
    for constraint in constraints:
        try:
            filtered = filtered.filter(f"COALESCE(({constraint.expression}), false)")
        except Exception:
            failed.append(constraint)
    selected = filtered.orderBy("__row_id").limit(count)
    selected_count = selected.count()
    if not failed and selected_count == count:
        for column in table.columns:
            report.add(f"{table.name}.{column.name}", "used", "check_constraints.expression")
        return selected.drop("__row_id")
    for constraint in constraints:
        if constraint in failed or selected_count < count:
            report.unmet(table.name, constraint.expression)
            for column in table.columns:
                report.add(f"{table.name}.{column.name}", "unmet", "check_constraints.expression")
    return frame.orderBy("__row_id").limit(count).drop("__row_id")


def _apply_categorical_dependencies(
    table: TableProfile, columns: dict[str, list[Any]], seed: int, report: ReportBuilder
) -> None:
    for dependency in table.dependencies or []:
        if dependency.method != DependenceMethod.CRAMERS_V:
            continue
        left, right = dependency.columns
        if not columns[left] or not columns[right]:
            continue
        rng = _rng(seed, table.name, left, right, "cramers_v")
        strength = dependency.value
        mapping: dict[Any, Any] = {}
        right_pool = list(dict.fromkeys(columns[right]))
        for index, value in enumerate(dict.fromkeys(columns[left])):
            mapping[value] = right_pool[index % len(right_pool)]
        for index, value in enumerate(columns[left]):
            if rng.random() < strength:
                columns[right][index] = mapping[value]
        report.add(f"{table.name}.{left}", "used", "dependencies.cramers_v")
        report.add(f"{table.name}.{right}", "used", "dependencies.cramers_v")
        report.add(f"{table.name}.{right}", "approximated", "dependencies.cramers_v.mixture")


def _nearest_psd(matrix: np.ndarray) -> tuple[np.ndarray, bool]:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    repaired = bool(np.any(values < 0))
    result = vectors @ np.diag(np.maximum(values, 1e-8)) @ vectors.T
    diagonal = np.sqrt(np.diag(result))
    return np.asarray(result / np.outer(diagonal, diagonal)), repaired


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray([_standard_normal_cdf(float(value)) for value in values])
    return _open_unit(probabilities)


def _normal_ppf(values: np.ndarray) -> np.ndarray:
    return np.asarray([_inverse_normal(float(value)) for value in values])


def _standard_normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _standard_normal_cdf(value: float) -> float:
    """Stable Φ(z) via erfc; does not saturate the same way as ½(1+erf) in the far tails."""
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def _standard_normal_sf(value: float) -> float:
    """Stable survival 1-Φ(z) via erfc."""
    return 0.5 * math.erfc(value / math.sqrt(2.0))


def _standard_normal_logsf(value: float) -> float:
    """log(1-Φ(z)), with a mills-ratio expansion when erfc underflows."""
    survival = _standard_normal_sf(value)
    if survival > 0.0:
        return math.log(survival)
    if value <= 0.0:
        return math.log(max(1.0 - _standard_normal_cdf(value), np.nextafter(0.0, 1.0)))
    # sf(z) ~ φ(z)/z (1 - 1/z^2 + 3/z^4 - 15/z^6) for large z > 0.
    inverse = 1.0 / (value * value)
    series = 1.0 - inverse + 3.0 * inverse * inverse - 15.0 * inverse * inverse * inverse
    return (
        -0.5 * value * value
        - 0.5 * math.log(2.0 * math.pi)
        - math.log(value)
        + math.log(max(series, np.nextafter(0.0, 1.0)))
    )


def _cdf_diff(alpha: float, beta: float) -> float:
    """Φ(beta) - Φ(alpha) without catastrophic cancellation in either tail."""
    if alpha >= 0.0:
        return _standard_normal_sf(alpha) - _standard_normal_sf(beta)
    if beta <= 0.0:
        return _standard_normal_sf(-beta) - _standard_normal_sf(-alpha)
    return _standard_normal_cdf(beta) - _standard_normal_cdf(alpha)


# Acklam rational-approximation coefficients (public domain).
_ACKLAM_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577459334152e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)


def _inverse_normal_from_log(log_probability: float) -> float:
    """Φ^{-1}(exp(log_p)) for probabilities too small to represent in float64."""
    if not math.isfinite(log_probability):
        return -math.inf if log_probability < 0 else math.inf
    if log_probability >= 0.0:
        return _inverse_normal(min(math.exp(log_probability), 1.0))
    q = math.sqrt(-2.0 * log_probability)
    return (
        ((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q + _ACKLAM_C[4]) * q
        + _ACKLAM_C[5]
    ) / ((((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0)


def _inverse_normal(probability: float) -> float:
    """Acklam's rational approximation to Φ^{-1} with a Halley refinement (no ±8 cap)."""
    if probability <= 0.0:
        return -math.inf
    if probability >= 1.0:
        return math.inf
    if probability == 0.5:
        return 0.0
    plow = 0.02425
    phigh = 1.0 - plow
    if probability < plow:
        x = _inverse_normal_from_log(math.log(probability))
    elif probability > phigh:
        x = -_inverse_normal_from_log(math.log1p(-probability))
    else:
        q = probability - 0.5
        r = q * q
        x = (
            (
                ((((_ACKLAM_A[0] * r + _ACKLAM_A[1]) * r + _ACKLAM_A[2]) * r + _ACKLAM_A[3]) * r + _ACKLAM_A[4]) * r
                + _ACKLAM_A[5]
            )
            * q
            / (
                ((((_ACKLAM_B[0] * r + _ACKLAM_B[1]) * r + _ACKLAM_B[2]) * r + _ACKLAM_B[3]) * r + _ACKLAM_B[4]) * r
                + 1.0
            )
        )
    error = _standard_normal_cdf(x) - probability
    density = _standard_normal_pdf(x)
    if density > 0.0 and math.isfinite(error):
        x -= error / (density + x * error / 2.0)
    return x


def _truncated_standard_ppf(uniform: float, alpha: float, beta: float) -> float:  # noqa: PLR0911
    """Inverse CDF of a standard normal truncated to [alpha, beta], stable in far tails."""
    u = float(np.clip(uniform, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)))
    if not math.isfinite(alpha) or not math.isfinite(beta) or alpha >= beta:
        return 0.0
    if beta <= 0.0:
        # Reflect left-tail windows onto the right tail so survival arithmetic applies.
        return -_truncated_standard_ppf(1.0 - u, -beta, -alpha)
    if alpha >= 0.0:
        log_sa = _standard_normal_logsf(alpha)
        log_sb = _standard_normal_logsf(beta)
        ratio = math.exp(log_sb - log_sa)
        weight = 1.0 - u * (1.0 - ratio)
        weight = min(max(weight, np.nextafter(0.0, 1.0)), 1.0)
        log_s = log_sa + math.log(weight)
        if log_s >= math.log(0.5):
            return _inverse_normal(1.0 - math.exp(log_s))
        survival = math.exp(log_s)
        if survival > 0.0:
            return -_inverse_normal(survival)
        return -_inverse_normal_from_log(log_s)
    lower = _standard_normal_cdf(alpha)
    span = _cdf_diff(alpha, beta)
    if span <= 0.0 or not math.isfinite(span):
        return 0.5 * (alpha + beta)
    return _inverse_normal(lower + u * span)


def _truncated_moments(mean: float, stddev: float, low: float, high: float) -> tuple[float, float]:
    """Mean and stddev of Normal(mean, stddev) truncated to [low, high]."""
    if not math.isfinite(stddev) or stddev <= 0 or not (low < high):
        center = float(np.clip(mean, low, high)) if math.isfinite(mean) else 0.5 * (low + high)
        return center, 0.0
    factors = _truncated_moment_factors((low - mean) / stddev, (high - mean) / stddev)
    if factors is None:
        return 0.5 * (low + high), 0.0
    delta, scale = factors
    return mean + stddev * delta, stddev * scale


def _calibrate_truncated_parameters(mean: float, stddev: float, low: float, high: float) -> tuple[float, float]:
    """Finds (mu, sigma) whose truncation to [low, high] matches the profiled mean/stddev.

    Uses a bounded fixed-point solve. When no truncated-normal moments match the targets
    (common for heavy left skew with a wide right bound), returns the profiled parameters
    unchanged so callers can document the resulting mean/stddev shift.
    """
    if not math.isfinite(stddev) or stddev <= 0 or not (low < high) or not (low < mean < high):
        return mean, stddev
    mu, sigma = mean, stddev
    best = (mu, sigma)
    best_error = math.inf
    for _ in range(25):
        current_mean, current_std = _truncated_moments(mu, sigma, low, high)
        if not math.isfinite(current_mean) or not math.isfinite(current_std) or current_std <= 0:
            break
        error = abs(current_mean - mean) / max(abs(mean), 1.0) + abs(current_std - stddev) / stddev
        if error < best_error:
            best_error = error
            best = (mu, sigma)
        if error < 0.01:
            return mu, sigma
        mu += mean - current_mean
        sigma = max(sigma * (stddev / current_std), stddev * 1e-6)
        sigma = min(sigma, max(high - low, stddev) * 20.0)
        mu = min(max(mu, low - 50.0 * sigma), high + 50.0 * sigma)
    matched_mean, matched_std = _truncated_moments(*best, low, high)
    if (
        abs(matched_mean - mean) <= 0.05 * max(abs(mean), 1.0) + 0.05
        and abs(matched_std - stddev) <= 0.05 * stddev + 0.05
    ):
        return best
    return mean, stddev


def _open_unit(values: np.ndarray | float) -> np.ndarray:
    """Maps uniforms into (0, 1) so inverse-CDF transforms do not land on exact endpoints."""
    return np.clip(np.asarray(values, dtype=float), np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))


def _open_interval_between(low: np.ndarray | float, high: np.ndarray | float, uniforms: np.ndarray) -> np.ndarray:
    """Samples the open interval (low, high) via inverse CDF, avoiding exact endpoint atoms."""
    u = _open_unit(uniforms)
    lower = np.asarray(low, dtype=float)
    upper = np.asarray(high, dtype=float)
    open_low = np.nextafter(lower, upper)
    open_high = np.nextafter(upper, lower)
    width = open_high - open_low
    usable = width > 0
    midpoint = np.where(np.isfinite(lower + upper), (lower + upper) / 2.0, lower)
    return np.where(usable, open_low + u * width, midpoint)


def _truncated_normal(  # noqa: PLR0911
    uniforms: np.ndarray,
    mean: float,
    stddev: float,
    low: float,
    high: float,
    *,
    open_bounds: bool,
) -> np.ndarray:
    """Inverse-CDF samples from Normal(mean, stddev) truncated to the observed bounds.

    Copula uniforms remain usable: each draw is F^{-1}(u) of the truncated CDF. Far-tail
    windows use survival/log-survival arithmetic so Φ saturation cannot collapse the span.
    stddev <= 0 falls back to a point mass at the clipped mean.
    """
    u = _open_unit(uniforms)
    count = len(u)
    if count == 0:
        return np.asarray([], dtype=float)
    if not math.isfinite(low) or not math.isfinite(high):
        if not math.isfinite(stddev) or stddev <= 0:
            center = mean if math.isfinite(mean) else 0.0
            return np.full(count, center)
        return mean + stddev * _normal_ppf(u)
    lower, upper = (high, low) if high < low else (low, high)
    if upper == lower:
        return np.full(count, lower)
    if open_bounds:
        open_low = np.nextafter(lower, upper)
        open_high = np.nextafter(upper, lower)
        if not open_low < open_high:
            return np.full(count, (lower + upper) / 2.0)
        lower, upper = float(open_low), float(open_high)
    if not math.isfinite(stddev) or stddev <= 0:
        center = mean if math.isfinite(mean) else (lower + upper) / 2.0
        return np.full(count, float(np.clip(center, lower, upper)))
    alpha = (lower - mean) / stddev
    beta = (upper - mean) / stddev
    draws = np.asarray([mean + stddev * _truncated_standard_ppf(float(value), alpha, beta) for value in u])
    return np.clip(draws, lower, upper)


def _truncated_moment_factors(alpha: float, beta: float) -> tuple[float, float] | None:
    """Returns (location factor, scale factor) for a standard truncated normal on [alpha, beta]."""
    span = _cdf_diff(alpha, beta)
    if span <= 0.0 or not math.isfinite(span):
        return None
    density_a = _standard_normal_pdf(alpha)
    density_b = _standard_normal_pdf(beta)
    delta = (density_a - density_b) / span
    variance_factor = 1.0 + (alpha * density_a - beta * density_b) / span - delta * delta
    if variance_factor < 0.0 or not math.isfinite(variance_factor):
        return None
    return delta, math.sqrt(variance_factor)


def _exact_location(mean: float | str | Decimal, low: Decimal, high: Decimal) -> Decimal:
    """Parses mean into Decimal without casting bounds, mid, or offsets through float().

    Decimal/str means stay exact. A float mean (NumericStats JSON number) becomes
    `Decimal(mean)` (the float's exact dyadic value) and is clamped to [low, high].
    Callers that need a lossless large-magnitude mean must supply Decimal or str
    (tests use model_construct).
    """
    with localcontext() as ctx:
        ctx.prec = 50
        if isinstance(mean, Decimal):
            value = mean
        elif isinstance(mean, str):
            value = Decimal(mean)
        elif not math.isfinite(mean):
            value = (low + high) / Decimal(2)
        else:
            value = Decimal(mean)
        return min(high, max(low, value))


def _exact_scale(stddev: float | str | Decimal) -> Decimal:
    """Parses stddev into Decimal without float arithmetic on bounds or other exact quantities."""
    with localcontext() as ctx:
        ctx.prec = 50
        if isinstance(stddev, Decimal):
            return stddev
        if isinstance(stddev, str):
            return Decimal(stddev)
        if not math.isfinite(stddev) or stddev < 0:
            return Decimal(0)
        return Decimal(stddev)


def _standardized_bounds(low: Decimal, high: Decimal, mu: Decimal, sigma: Decimal) -> tuple[float, float]:
    """Returns (a, b) from Decimal (low-mu)/sigma and (high-mu)/sigma; float only after division."""
    with localcontext() as ctx:
        ctx.prec = 50
        return float((low - mu) / sigma), float((high - mu) / sigma)


def _calibrate_truncated_parameters_exact(
    mean: Decimal, stddev: Decimal, low: Decimal, high: Decimal
) -> tuple[Decimal, Decimal]:
    """Calibrates (mu, sigma) in Decimal; only standardized (a, b) and sd-unit factors use float."""
    if stddev <= 0 or not (low < high) or not (low < mean < high):
        return mean, stddev
    with localcontext() as ctx:
        ctx.prec = 50
        mu, sigma = mean, stddev
        best = (mu, sigma)
        best_error = Decimal("Infinity")
        for _ in range(25):
            alpha, beta = _standardized_bounds(low, high, mu, sigma)
            factors = _truncated_moment_factors(alpha, beta)
            if factors is None:
                break
            delta, scale = factors
            current_mean = mu + sigma * Decimal(str(delta))
            current_std = sigma * Decimal(str(scale))
            if current_std <= 0:
                break
            error = abs(current_mean - mean) / max(abs(mean), Decimal(1)) + abs(current_std - stddev) / stddev
            if error < best_error:
                best_error = error
                best = (mu, sigma)
            if error < Decimal("0.01"):
                return mu, sigma
            mu += mean - current_mean
            sigma = max(sigma * (stddev / current_std), stddev * Decimal("1e-6"))
            sigma = min(sigma, max(high - low, stddev) * Decimal(20))
            mu = min(max(mu, low - Decimal(50) * sigma), high + Decimal(50) * sigma)
        matched = _truncated_moment_factors(*_standardized_bounds(low, high, best[0], best[1]))
        if matched is not None:
            matched_mean = best[0] + best[1] * Decimal(str(matched[0]))
            matched_std = best[1] * Decimal(str(matched[1]))
            if abs(matched_mean - mean) <= Decimal("0.05") * max(abs(mean), Decimal(1)) + Decimal("0.05") and abs(
                matched_std - stddev
            ) <= Decimal("0.05") * stddev + Decimal("0.05"):
                return best
        return mean, stddev


def _truncated_moments_exact(mean: Decimal, stddev: Decimal, low: Decimal, high: Decimal) -> tuple[Decimal, Decimal]:
    """Exact-domain truncated mean/stddev using Decimal location/scale and standardized factors."""
    with localcontext() as ctx:
        ctx.prec = 50
        if stddev <= 0 or not (low < high):
            center = min(high, max(low, mean))
            return center, Decimal(0)
        factors = _truncated_moment_factors(*_standardized_bounds(low, high, mean, stddev))
        if factors is None:
            return (low + high) / Decimal(2), Decimal(0)
        delta, scale = factors
        return mean + stddev * Decimal(str(delta)), stddev * Decimal(str(scale))


def _exact_truncated_normal(
    uniforms: np.ndarray,
    mean: float | str | Decimal,
    stddev: float | str | Decimal,
    low: Decimal,
    high: Decimal,
    column: ColumnProfile,
) -> list[Any]:
    """Draws truncated-normal z in float, then maps mu+sigma*z with Decimal/int exact arithmetic.

    Exact quantities (min/max/mean/stddev/mid/offsets) stay Decimal; only standardized (a, b)
    and z are floats, each from a Decimal division (or inverse-CDF) first.
    """
    if high < low:
        low, high = high, low
    with localcontext() as ctx:
        ctx.prec = 50
        mean_d = _exact_location(mean, low, high)
        std_d = _exact_scale(stddev)
        if high == low or std_d <= 0:
            center = min(high, max(low, mean_d))
            return [_convert_exact(center, column) for _ in uniforms]
        mu_d, sigma_d = _calibrate_truncated_parameters_exact(mean_d, std_d, low, high)
        if sigma_d <= 0:
            return [_convert_exact(min(high, max(low, mu_d)), column) for _ in uniforms]
        alpha, beta = _standardized_bounds(low, high, mu_d, sigma_d)
        values: list[Any] = []
        for uniform in _open_unit(uniforms):
            z = _truncated_standard_ppf(float(uniform), alpha, beta)
            value = mu_d + sigma_d * Decimal(str(z))
            values.append(_convert_exact(min(high, max(low, value)), column))
        return values


def _away_from_endpoints(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Nudges continuous draws off exact finite endpoints (quantile plateaus, degenerate bins)."""
    if not math.isfinite(low) or not math.isfinite(high) or low == high:
        return values
    lower, upper = (high, low) if high < low else (low, high)
    result = np.asarray(values, dtype=float).copy()
    result[result == lower] = np.nextafter(lower, upper)
    result[result == upper] = np.nextafter(upper, lower)
    return result


def _number(value: Any, column: ColumnProfile) -> float:  # noqa: ANN401
    if column.logical_type in _TEMPORAL_TYPES:
        converted = _literal(value, column)
        if isinstance(converted, datetime):
            return converted.replace(tzinfo=converted.tzinfo or timezone.utc).timestamp() * 1_000_000
        if isinstance(converted, date):
            return float((converted - date(1970, 1, 1)).days)
        if isinstance(converted, time):
            return float(
                ((converted.hour * 60 + converted.minute) * 60 + converted.second) * 1_000_000 + converted.microsecond
            )
    return float(value)


def _convert_ordered(value: float, column: ColumnProfile) -> Any:  # noqa: ANN401
    kind = column.logical_type
    if kind in _INTEGER_TYPES:
        return round(value)
    if kind in (LogicalType.FLOAT32, LogicalType.FLOAT64):
        return float(value)
    if kind == LogicalType.DECIMAL:
        quantum = Decimal(1).scaleb(-(column.scale or 0))
        limit = Decimal(10) ** ((column.precision or 38) - (column.scale or 0)) - quantum
        return min(limit, max(-limit, Decimal(str(value)))).quantize(quantum)
    if kind == LogicalType.DATE:
        return date(1970, 1, 1) + timedelta(days=round(value))
    if kind == LogicalType.TIME:
        micros = round(value) % (24 * 60 * 60 * 1_000_000)
        return (datetime.min + timedelta(microseconds=micros)).time().isoformat()
    result = datetime(1970, 1, 1) + timedelta(microseconds=float(value))
    return result.replace(tzinfo=timezone.utc) if kind == LogicalType.TIMESTAMP else result


def _literal(value: Any, column: ColumnProfile) -> Any:  # noqa: ANN401, PLR0911
    if value is None:
        return None
    kind = column.logical_type
    if kind in _INTEGER_TYPES:
        return int(value)
    if kind in (LogicalType.FLOAT32, LogicalType.FLOAT64):
        return float(value)
    if kind == LogicalType.DECIMAL:
        return Decimal(str(value))
    if kind in _TEMPORAL_TYPES:
        parsed = parse_temporal(str(value), kind)
        if kind == LogicalType.TIMESTAMP_NTZ and isinstance(parsed, datetime):
            return parsed.replace(tzinfo=None)
        return parsed
    return value


def _field(column: ColumnProfile) -> st.StructField:
    return st.StructField(column.name, _spark_type(column), column.nullable)


def _construction_field(column: ColumnProfile) -> st.StructField:
    if column.logical_type == LogicalType.VARIANT:
        return st.StructField(column.name, st.StringType(), column.nullable)
    return _field(column)


def _arrow_field(column: ColumnProfile) -> pa.Field:
    return pa.field(column.name, _arrow_type(column), nullable=column.nullable)


def _arrow_type(column: ColumnProfile) -> pa.DataType:
    kind = column.logical_type
    primitive: dict[LogicalType, pa.DataType] = {
        LogicalType.NULL: pa.null(),
        LogicalType.BOOLEAN: pa.bool_(),
        LogicalType.INT8: pa.int8(),
        LogicalType.INT16: pa.int16(),
        LogicalType.INT32: pa.int32(),
        LogicalType.INT64: pa.int64(),
        LogicalType.UINT8: pa.int16(),
        LogicalType.UINT16: pa.int32(),
        LogicalType.UINT32: pa.int64(),
        LogicalType.UINT64: pa.decimal128(20, 0),
        LogicalType.FLOAT32: pa.float32(),
        LogicalType.FLOAT64: pa.float64(),
        LogicalType.STRING: pa.string(),
        LogicalType.BINARY: pa.binary(),
        LogicalType.DATE: pa.date32(),
        LogicalType.TIME: pa.string(),
        LogicalType.TIMESTAMP: pa.timestamp("us", tz="UTC"),
        LogicalType.TIMESTAMP_NTZ: pa.timestamp("us"),
        LogicalType.INTERVAL: (
            pa.int32()
            if "year" in column.source_type.lower() or "month" in column.source_type.lower()
            else pa.duration("us")
        ),
        LogicalType.VARIANT: pa.string(),
    }
    if kind == LogicalType.DECIMAL:
        return pa.decimal128(column.precision or 38, column.scale or 0)
    if kind == LogicalType.STRUCT:
        return pa.struct([_arrow_field(child) for child in column.children or []])
    if kind == LogicalType.ARRAY:
        return pa.list_(_arrow_field((column.children or [])[0]))
    if kind == LogicalType.MAP:
        key, value = column.children or []
        return pa.map_(_arrow_type(key), _arrow_type(value))
    return primitive[kind]


def _spark_type(column: ColumnProfile) -> st.DataType:
    kind = column.logical_type
    primitive: dict[LogicalType, st.DataType] = {
        LogicalType.NULL: st.NullType(),
        LogicalType.BOOLEAN: st.BooleanType(),
        LogicalType.INT8: st.ByteType(),
        LogicalType.INT16: st.ShortType(),
        LogicalType.INT32: st.IntegerType(),
        LogicalType.INT64: st.LongType(),
        LogicalType.UINT8: st.ShortType(),
        LogicalType.UINT16: st.IntegerType(),
        LogicalType.UINT32: st.LongType(),
        LogicalType.UINT64: st.DecimalType(20, 0),
        LogicalType.FLOAT32: st.FloatType(),
        LogicalType.FLOAT64: st.DoubleType(),
        LogicalType.STRING: st.StringType(),
        LogicalType.BINARY: st.BinaryType(),
        LogicalType.DATE: st.DateType(),
        LogicalType.TIME: st.StringType(),
        LogicalType.TIMESTAMP: st.TimestampType(),
        LogicalType.TIMESTAMP_NTZ: st.TimestampNTZType(),
        LogicalType.INTERVAL: (
            st.YearMonthIntervalType()
            if "year" in column.source_type.lower() or "month" in column.source_type.lower()
            else st.DayTimeIntervalType()
        ),
        LogicalType.VARIANT: st.VariantType(),
    }
    if kind == LogicalType.DECIMAL:
        return st.DecimalType(column.precision or 38, column.scale or 0)
    if kind == LogicalType.STRUCT:
        return st.StructType([_field(child) for child in column.children or []])
    if kind == LogicalType.ARRAY:
        child = (column.children or [])[0]
        return st.ArrayType(_spark_type(child), child.nullable)
    if kind == LogicalType.MAP:
        key, value = column.children or []
        return st.MapType(_spark_type(key), _spark_type(value), value.nullable)
    return primitive[kind]


def _is_ordered(column: ColumnProfile) -> bool:
    return column.logical_type in _ORDERED_TYPES


_INTEGER_TYPES = {
    LogicalType.INT8,
    LogicalType.INT16,
    LogicalType.INT32,
    LogicalType.INT64,
    LogicalType.UINT8,
    LogicalType.UINT16,
    LogicalType.UINT32,
    LogicalType.UINT64,
}
_TEMPORAL_TYPES = {LogicalType.DATE, LogicalType.TIME, LogicalType.TIMESTAMP, LogicalType.TIMESTAMP_NTZ}
_ORDERED_TYPES = _INTEGER_TYPES | {LogicalType.FLOAT32, LogicalType.FLOAT64, LogicalType.DECIMAL} | _TEMPORAL_TYPES
