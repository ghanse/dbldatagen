"""Reference reconstruction of PySpark DataFrames from profile information."""

from __future__ import annotations

import hashlib
import math
import re
import string
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
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
        _apply_relationships(table, columns, values, relationships, seed, report)
        _apply_keys(table, columns, report)
        _apply_constraints(table, columns, report)
        rows = [
            {column.name: columns[column.name][index] for column in table.columns} for index in range(candidate_count)
        ]
        arrow = pa.Table.from_pylist(rows, schema=pa.schema([_arrow_field(column) for column in table.columns]))
        construction_schema = st.StructType([_construction_field(column) for column in table.columns])
        frame = spark.createDataFrame(arrow, schema=construction_schema)
        frame = frame.select(
            *[
                (
                    F.parse_json(F.col(column.name)).alias(column.name)
                    if column.logical_type == LogicalType.VARIANT
                    else F.col(column.name)
                )
                for column in table.columns
            ]
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
    report.add(path, "used", "logical_type", "source_type", "nullable", "null_fraction")
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
        values = _generate_scalar(column, count, rng, report, path, uniforms)
    if column.null_fraction and count:
        null_count = min(count, round(column.null_fraction * count))
        indexes = rng.permutation(count)[:null_count]
        for index in indexes:
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
    return _apply_pool(column, values, rng, report, path)


def _ordered_values(  # noqa: PLR0912
    column: ColumnProfile,
    count: int,
    rng: np.random.Generator,
    report: ReportBuilder,
    path: str,
    uniforms: np.ndarray | None,
) -> list[Any]:
    stats = column.numeric or column.temporal
    u = rng.random(count) if uniforms is None else uniforms
    if column.numeric and column.numeric.histogram:
        histogram = column.numeric.histogram
        weights = np.asarray(histogram.counts, dtype=float)
        bins = np.searchsorted(np.cumsum(weights / weights.sum()), u, side="right")
        low = np.asarray([_number(histogram.edges[index], column) for index in bins])
        high = np.asarray([_number(histogram.edges[index + 1], column) for index in bins])
        raw = low + rng.random(count) * (high - low)
        report.add(path, "used", "numeric.histogram")
        for field in ("quantiles", "min", "max", "mean", "stddev"):
            if getattr(column.numeric, field) is not None:
                report.add(path, "ignored", f"numeric.{field}")
    elif stats and stats.quantiles:
        quantiles = sorted(stats.quantiles, key=lambda item: item.q)
        xp = np.asarray([item.q for item in quantiles], dtype=float)
        fp = np.asarray([_number(item.value, column) for item in quantiles], dtype=float)
        raw = np.interp(u, xp, fp)
        report.add(path, "used", f"{'numeric' if column.numeric else 'temporal'}.quantiles")
        prefix = "numeric" if column.numeric else "temporal"
        for field in ("min", "max", "mean", "stddev") if column.numeric else ("min", "max"):
            if getattr(stats, field) is not None:
                report.add(path, "ignored", f"{prefix}.{field}")
    elif stats and stats.min is not None and stats.max is not None:
        bound_low, bound_high = _number(stats.min, column), _number(stats.max, column)
        if column.numeric and column.numeric.mean is not None and column.numeric.stddev is not None:
            raw = np.clip(_normal_ppf(u) * column.numeric.stddev + column.numeric.mean, bound_low, bound_high)
            report.add(path, "used", "numeric.min", "numeric.max", "numeric.mean", "numeric.stddev")
            report.add(path, "approximated", "numeric.clipped_normal")
        else:
            raw = bound_low + u * (bound_high - bound_low)
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
    return [_convert_ordered(value, column) for value in raw]


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


def _apply_pool(
    column: ColumnProfile, values: list[Any], rng: np.random.Generator, report: ReportBuilder, path: str
) -> list[Any]:
    if column.cardinality == Cardinality.UNIQUE:
        return values
    size = column.distinct_count.value if column.distinct_count else _bucket_size(column.cardinality, len(values))
    if size is None or size <= 0 or not values:
        return values
    pool = values[: min(size, len(values))]
    report.add(path, "used", "distinct_count" if column.distinct_count else "cardinality")
    if column.distinct_count and not column.distinct_count.exact:
        report.add(path, "approximated", "distinct_count.approximate")
    return [pool[int(index)] for index in rng.integers(0, len(pool), len(values))]


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
        count = len(next(iter(columns.values()), []))
        covered = min(count, round(count * relationship.coverage))
        indexes = _fanout_indexes(covered, len(parents), relationship, rng)
        for child_index, child_name in enumerate(relationship.child_columns):
            values = [parents[parent][child_index] for parent in indexes]
            values.extend(columns[child_name][covered:])
            columns[child_name] = values
            path = f"{table.name}.{child_name}"
            report.add(path, "used", "relationship.coverage", "relationship.parent_key")
            if relationship.fan_out:
                report.add(path, "used", "relationship.fan_out")
                report.add(path, "approximated", "relationship.fan_out.weighted")
        _exclude_parent_keys(table, columns, relationship, parents, covered)


def _exclude_parent_keys(
    table: TableProfile,
    columns: dict[str, list[Any]],
    relationship: Relationship,
    parents: list[tuple[Any, ...]],
    covered: int,
) -> None:
    parent_set = set(parents)
    count = len(next(iter(columns.values()), []))
    last_name = relationship.child_columns[-1]
    last_column = next(column for column in table.columns if column.name == last_name)
    for index in range(covered, count):
        key = tuple(columns[name][index] for name in relationship.child_columns)
        if key in parent_set:
            columns[last_name][index] = _unique_value(
                columns[last_name][index], index + count, last_column, columns[last_name]
            )


def _fanout_indexes(count: int, parent_count: int, relationship: Relationship, rng: np.random.Generator) -> np.ndarray:
    if parent_count == 0:
        return np.zeros(count, dtype=int)
    if not relationship.fan_out or relationship.fan_out.stddev is None:
        return rng.integers(0, parent_count, count)
    weights = np.clip(rng.normal(relationship.fan_out.mean, relationship.fan_out.stddev, parent_count), 0, None)
    if not weights.any():
        weights[:] = 1
    return rng.choice(parent_count, count, p=weights / weights.sum())


def _apply_keys(table: TableProfile, columns: dict[str, list[Any]], report: ReportBuilder) -> None:
    keys = ([table.primary_key] if table.primary_key else []) + (table.unique_keys or [])
    count = len(next(iter(columns.values()), []))
    for key in keys:
        seen: set[tuple[Any, ...]] = set()
        for index in range(count):
            value = tuple(columns[name][index] for name in key.columns)
            needs_value = key is table.primary_key and any(item is None for item in value)
            duplicate = value in seen and not (key is not table.primary_key and any(item is None for item in value))
            if needs_value or duplicate:
                last = key.columns[-1]
                column = next(column for column in table.columns if column.name == last)
                columns[last][index] = _unique_value(columns[last][index], index, column, columns[last])
                value = tuple(columns[name][index] for name in key.columns)
            seen.add(value)
        for name in key.columns:
            report.add(f"{table.name}.{name}", "used", "primary_key" if key is table.primary_key else "unique_keys")


def _unique_value(value: Any, index: int, column: ColumnProfile, existing: list[Any]) -> Any:  # noqa: ANN401, PLR0911
    kind = column.logical_type
    if kind == LogicalType.STRING:
        return f"{value or ''}_{index}"
    if kind in _INTEGER_TYPES:
        present = [item for item in existing if item is not None]
        return (max(present) if present else 0) + index + 1
    if kind in (LogicalType.FLOAT32, LogicalType.FLOAT64):
        present = [item for item in existing if item is not None]
        return float(max(present) if present else 0) + index + 1
    if kind == LogicalType.DECIMAL:
        present = [item for item in existing if item is not None]
        base = max(present) if present else Decimal(0)
        return (base + index + 1).quantize(Decimal(1).scaleb(-(column.scale or 0)))
    if kind == LogicalType.DATE:
        return date(1970, 1, 1) + timedelta(days=index)
    if kind in (LogicalType.TIMESTAMP, LogicalType.TIMESTAMP_NTZ):
        return datetime(1970, 1, 1) + timedelta(microseconds=index)
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
            _enforce_comparison(columns, constraint)
            report.add(f"{table.name}.{constraint.left}", "used", "check_constraints.comparison")
            report.add(f"{table.name}.{constraint.right}", "used", "check_constraints.comparison")


def _enforce_comparison(columns: dict[str, list[Any]], constraint: ComparisonConstraint) -> None:
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
                columns[constraint.right][index] = _coerce_like(left, right)
            elif constraint.operator in (">", ">="):
                columns[constraint.left][index] = _coerce_like(right, left)
            elif constraint.operator == "=":
                columns[constraint.right][index] = left
            else:
                columns[constraint.right][index] = _different(right)


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
        return frame
    filtered = frame
    failed: list[ExpressionConstraint] = []
    for constraint in constraints:
        try:
            filtered = filtered.filter(f"COALESCE(({constraint.expression}), false)")
        except Exception:
            failed.append(constraint)
    if not failed and filtered.limit(count).count() == count:
        for column in table.columns:
            report.add(f"{table.name}.{column.name}", "used", "check_constraints.expression")
        return filtered.limit(count)
    for constraint in constraints:
        if constraint in failed or filtered.limit(count).count() < count:
            report.unmet(table.name, constraint.expression)
            for column in table.columns:
                report.add(f"{table.name}.{column.name}", "unmet", "check_constraints.expression")
    return frame.limit(count)


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
    return np.asarray([0.5 * (1 + math.erf(value / math.sqrt(2))) for value in values])


def _normal_ppf(values: np.ndarray) -> np.ndarray:
    return np.asarray([_inverse_normal(float(value)) for value in values])


def _inverse_normal(probability: float) -> float:
    low, high = -8.0, 8.0
    for _ in range(48):
        middle = (low + high) / 2
        if 0.5 * (1 + math.erf(middle / math.sqrt(2))) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


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
    if kind == LogicalType.DATE:
        return date.fromisoformat(str(value))
    if kind == LogicalType.TIME:
        return time.fromisoformat(str(value))
    if kind in (LogicalType.TIMESTAMP, LogicalType.TIMESTAMP_NTZ):
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if kind == LogicalType.TIMESTAMP else parsed.replace(tzinfo=None)
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
