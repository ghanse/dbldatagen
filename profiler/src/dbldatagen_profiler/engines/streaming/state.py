"""Mergeable profiling state for incremental (streaming) profiling.

A `TableState` holds partial statistics for the rows seen so far. States computed from disjoint sets of rows merge
with `merge_states`; the merge is associative and commutative. `profile_states` turns merged states into a profile.

Exact statistics (equal to the batch engine): row count, null fraction, value counts of nested columns, numeric and
temporal min/max (lossless for decimal and int64), string and collection length min/max/mean, and boolean true
fraction. Approximate statistics: numeric mean/stddev and collection length stddev are merged in floating point
(Chan et al.), so they match the batch engine to rounding error only. Distinct counts come from Spark's built-in
HyperLogLog sketch (Apache DataSketches HLL, lgConfigK=`HLL_LG_CONFIG_K`) and are flagged `exact=false`, with a
relative standard error of `DISTINCT_RELATIVE_ERROR`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, TypeVar

import pyspark
from pyspark.sql import Column, DataFrame, Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from dbldatagen_profiler.engines.spark.engine import (
    _NUMERIC_TYPES,
    _encode_numeric,
    _encode_temporal,
    _finite_float,
    _logical_type,
    _protect_column,
    _string_types,
    _temporal_extreme_literal,
    _temporal_types,
    _timestamp_ntz_types,
    _utc_session_time_zones,
)
from dbldatagen_profiler.model import (
    SPEC_VERSION,
    BooleanStats,
    ColumnProfile,
    DistinctCount,
    EngineInfo,
    LengthStats,
    NumericStats,
    PrivacyPolicy,
    Profile,
    QuasiIdentifierSet,
    Semantic,
    StringStats,
    TableProfile,
    TemporalStats,
    ValueTreatment,
)
from dbldatagen_profiler.pii.detect import _detect_rows, _find_quasi_identifier_sets


HLL_LG_CONFIG_K = 12
DISTINCT_RELATIVE_ERROR = 1.04 / math.sqrt(2**HLL_LG_CONFIG_K)
PII_SAMPLE_ROWS = 10_000
ENGINE_NAME = "pyspark-streaming"

PiiAnalysis = tuple[dict[str, Semantic], list[tuple[list[str], float, int]]]
SampleKey = tuple[int | str, ...]
SampleEntry = tuple[SampleKey, Row]

_VALUE = "__profile_value"
_ROW = "__profile_row"


@dataclass(frozen=True)
class Moments:
    """Count, mean, and sum of squared deviations of a set of floats."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def merge(self, other: Moments) -> Moments:
        """Returns the moments of both sets combined."""
        if not other.count:
            return self
        if not self.count:
            return other
        count = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * other.count / count
        m2 = self.m2 + other.m2 + delta * delta * self.count * other.count / count
        return Moments(count, mean, m2)

    def stddev(self) -> float | None:
        """Returns the sample standard deviation, or None for fewer than two values."""
        if self.count < 2:
            return None
        return _finite_float(math.sqrt(self.m2 / (self.count - 1))) if self.m2 >= 0 else None


@dataclass(frozen=True)
class Lengths:
    """Exact length totals of strings, arrays, or maps."""

    count: int = 0
    total: int = 0
    total_squares: int = 0
    min: int | None = None
    max: int | None = None

    def merge(self, other: Lengths) -> Lengths:
        """Returns the lengths of both sets combined."""
        return Lengths(
            self.count + other.count,
            self.total + other.total,
            self.total_squares + other.total_squares,
            _extreme(self.min, other.min, largest=False),
            _extreme(self.max, other.max, largest=True),
        )

    def stats(self, *, include_stddev: bool) -> LengthStats:
        """Returns the length statistics."""
        stddev = None
        if include_stddev and self.count >= 2:
            variance = (self.count * self.total_squares - self.total * self.total) / (self.count * (self.count - 1))
            stddev = math.sqrt(max(variance, 0.0))
        return LengthStats(
            min=self.min,
            max=self.max,
            mean=self.total / self.count if self.count else None,
            stddev=stddev,
        )


_Mergeable = TypeVar("_Mergeable", Moments, Lengths)
_Ordered = TypeVar("_Ordered", int, float, Decimal, tuple[Any, str])


@dataclass(frozen=True)
class ColumnState:
    """Partial statistics of one column, or of a nested field.

    Attributes:
        count: Number of values, including nulls.
        non_null: Number of non-null values.
        sketch: HyperLogLog sketch of the non-null values, or None when none were seen.
        distinct: Distinct-count estimate of `sketch`.
        min: Smallest value: numeric as Python int, Decimal, or float; temporal as a (sort key, literal) pair.
        max: Largest value, encoded like `min`.
        moments: Moments of numeric values cast to double.
        lengths: Lengths of strings, arrays, or maps.
        true_count: Number of true values of a boolean column.
        children: States of nested fields, in the order of the batch engine's `children`.
    """

    count: int = 0
    non_null: int = 0
    sketch: bytes | None = None
    distinct: int = 0
    min: Any = None
    max: Any = None
    moments: Moments | None = None
    lengths: Lengths | None = None
    true_count: int | None = None
    children: tuple[ColumnState, ...] = ()


@dataclass(frozen=True)
class TableState:
    """Partial statistics of one table.

    Attributes:
        schema: Spark schema of the table.
        row_count: Number of rows.
        columns: Column states in schema order.
        sample: Bounded PII detection sample: the first `sample_size` rows in arrival order, with their sort keys.
            Arrival order is given by the order columns of `state_for_dataframe`, then the batch engine's
            `(xxhash64, values)` sampling key.
        sample_size: Maximum sample size.
    """

    schema: T.StructType
    row_count: int = 0
    columns: tuple[ColumnState, ...] = ()
    sample: tuple[SampleEntry, ...] = field(default=(), repr=False)
    sample_size: int = PII_SAMPLE_ROWS


def empty_state(schema: T.StructType, *, sample_size: int = PII_SAMPLE_ROWS) -> TableState:
    """Returns the state of a table with no rows.

    Args:
        schema: Spark schema of the table.
        sample_size: Maximum PII sample size.

    Returns:
        A state that is the identity of `merge_states`.
    """
    return TableState(schema, 0, tuple(_empty_column(f.dataType) for f in schema.fields), (), sample_size)


def _empty_column(data_type: T.DataType) -> ColumnState:
    return ColumnState(children=tuple(_empty_column(t) for t in _child_types(data_type)))


def require_streaming_support(version: str | None = None) -> None:
    """Checks that PySpark has the HyperLogLog sketch functions the streaming engine needs.

    Args:
        version: PySpark version to check (default: the installed version).

    Raises:
        RuntimeError: If PySpark is older than 3.5.
    """
    found = version or pyspark.__version__
    major, minor = (int(part) for part in re.findall(r"\d+", found)[:2])
    if (major, minor) < (3, 5):
        raise RuntimeError(f"StreamingProfiler requires pyspark >= 3.5; found {found}")


def state_for_dataframe(
    df: DataFrame,
    *,
    sample_size: int = PII_SAMPLE_ROWS,
    collect_sample: bool = True,
    order_columns: Sequence[str] = (),
) -> TableState:
    """Computes the state of a DataFrame, profiled as one micro-batch.

    Args:
        df: Batch DataFrame to profile.
        sample_size: Maximum PII sample size.
        collect_sample: Whether to collect the PII detection sample (default true).
        order_columns: Columns of `df` that give the arrival order of rows for the PII sample. They are not profiled.

    Returns:
        The table state of every row in `df` except the order columns.

    Raises:
        RuntimeError: If PySpark is older than 3.5.
    """
    require_streaming_support()
    profiled = df.drop(*order_columns)
    with _utc_session_time_zones([df]):
        row_count, columns = _aggregate(profiled, [(profiled[f.name], f.dataType) for f in profiled.schema.fields])
        sample = _sample(df, profiled.schema, order_columns, sample_size) if collect_sample else ()
    return TableState(profiled.schema, row_count, tuple(columns), sample, sample_size)


def merge_states(left: TableState, right: TableState, *, spark: SparkSession | None = None) -> TableState:
    """Merges the states of two disjoint sets of rows of the same table.

    Merging is associative and commutative. Sketches are merged with Spark's `hll_union`, so a Spark session is used.

    Args:
        left: State of the first set of rows.
        right: State of the second set of rows.
        spark: Spark session for the sketch union (default: the active session).

    Returns:
        The state of both sets of rows.

    Raises:
        ValueError: If the states have different schemas or sample sizes.
    """
    if left.schema != right.schema or left.sample_size != right.sample_size:
        raise ValueError("cannot merge states of different schemas or sample sizes")
    pairs = [pair for a, b in zip(left.columns, right.columns, strict=True) for pair in _sketch_pairs(a, b)]
    unions = iter(_union_sketches(pairs, spark or SparkSession.active()))
    columns = tuple(_merge_column(a, b, unions) for a, b in zip(left.columns, right.columns, strict=True))
    sample = tuple(sorted((*left.sample, *right.sample), key=lambda entry: entry[0])[: left.sample_size])
    return TableState(left.schema, left.row_count + right.row_count, columns, sample, left.sample_size)


def analyze_sample(state: TableState, spark: SparkSession | None = None) -> PiiAnalysis:
    """Detects semantics and quasi-identifiers from a state's PII sample with the batch engine's rules.

    Args:
        state: Table state.
        spark: Spark session used to build an empty DataFrame carrying the schema (default: the active session).

    Returns:
        Semantics keyed by column path, and quasi-identifier sets.
    """
    frame = (spark or SparkSession.active()).createDataFrame([], state.schema)
    rows = [row for _, row in state.sample]
    semantics = _detect_rows(frame, rows)
    return semantics, _find_quasi_identifier_sets(frame, semantics, rows)


def profile_states(
    states: Mapping[str, TableState],
    *,
    privacy: PrivacyPolicy | None = None,
    pii: Mapping[str, PiiAnalysis] | None = None,
) -> Profile:
    """Builds a validated profile from table states.

    Args:
        states: Table states keyed by profile table name.
        privacy: Optional privacy policy applied to emitted literals.
        pii: Optional PII analysis per table; tables without one carry no semantics.

    Returns:
        A strictly validated profile.

    Raises:
        ValueError: If no states are supplied.
    """
    if not states:
        raise ValueError("at least one table is required")
    policy = privacy or PrivacyPolicy()
    result = Profile(
        spec_version=SPEC_VERSION,
        created_at=datetime.now(timezone.utc),
        engine=EngineInfo(name=ENGINE_NAME, version=pyspark.__version__),
        privacy=policy,
        tables=[_table_profile(name, state, policy, (pii or {}).get(name)) for name, state in states.items()],
    )
    return Profile.model_validate(result.model_dump(), context={"strict": True})


def _table_profile(name: str, state: TableState, policy: PrivacyPolicy, analysis: PiiAnalysis | None) -> TableProfile:
    semantics, quasi_sets = analysis or ({}, [])
    columns = [
        _protect_column(_column_profile(f, column, policy, nested=False), f.name, semantics)
        for f, column in zip(state.schema.fields, state.columns, strict=True)
    ]
    quasi = [QuasiIdentifierSet(columns=cols, unique_fraction=unique, k_anonymity=k) for cols, unique, k in quasi_sets]
    return TableProfile(name=name, row_count=state.row_count, columns=columns, quasi_identifier_sets=quasi or None)


def _column_profile(f: T.StructField, state: ColumnState, policy: PrivacyPolicy, *, nested: bool) -> ColumnProfile:
    data_type = f.dataType
    include_extremes = policy.extremes != ValueTreatment.SUPPRESSED
    attributes: dict[str, Any] = {
        "name": f.name,
        "logical_type": _logical_type(data_type),
        "source_type": data_type.simpleString(),
        "nullable": f.nullable,
        "null_fraction": (state.count - state.non_null) / state.count if state.count else 0.0,
        "distinct_count": DistinctCount(value=min(state.distinct, state.non_null), exact=False),
    }
    if nested:
        attributes["value_count"] = state.count
    if isinstance(data_type, T.DecimalType):
        attributes.update(precision=data_type.precision, scale=data_type.scale)
    if isinstance(data_type, _NUMERIC_TYPES):
        moments = state.moments or Moments()
        attributes["numeric"] = NumericStats(
            min=_encode_numeric(state.min, data_type) if include_extremes else None,
            max=_encode_numeric(state.max, data_type) if include_extremes else None,
            mean=_finite_float(moments.mean) if moments.count else None,
            stddev=moments.stddev(),
        )
    elif isinstance(data_type, _string_types()):
        attributes["string"] = StringStats(length=(state.lengths or Lengths()).stats(include_stddev=False))
    elif isinstance(data_type, _temporal_types()):
        attributes["temporal"] = (
            TemporalStats(
                min=_encode_temporal(state.min[1], data_type) if state.min else None,
                max=_encode_temporal(state.max[1], data_type) if state.max else None,
            )
            if include_extremes
            else TemporalStats()
        )
    elif isinstance(data_type, T.BooleanType) and state.non_null:
        attributes["boolean"] = BooleanStats(true_fraction=(state.true_count or 0) / state.non_null)
    elif isinstance(data_type, (T.ArrayType, T.MapType)):
        attributes["collection_length"] = (state.lengths or Lengths()).stats(include_stddev=True)
    if state.children:
        attributes["children"] = [
            _column_profile(child_field, child, policy, nested=True)
            for child_field, child in zip(_child_fields(data_type), state.children, strict=True)
        ]
    return ColumnProfile(**attributes)


def _aggregate(df: DataFrame, columns: Sequence[tuple[Column, T.DataType]]) -> tuple[int, list[ColumnState]]:
    """Computes the states of columns of one DataFrame with a single Spark aggregate, plus one per nested level."""
    expressions = [F.count(F.lit(1)).alias("__row_count")]
    for index, (column, data_type) in enumerate(columns):
        expressions.extend(_expressions(column, data_type, f"c{index}"))
    row = df.agg(*expressions).first()
    if row is None:
        raise RuntimeError("Spark aggregate returned no row")
    values = row.asDict()
    count = int(values["__row_count"])
    states = [
        _column_state(values, f"c{index}", count, data_type, _children(df, column, data_type))
        for index, (column, data_type) in enumerate(columns)
    ]
    return count, states


def _children(df: DataFrame, column: Column, data_type: T.DataType) -> tuple[ColumnState, ...]:
    if isinstance(data_type, T.StructType):
        names = [f"__v{index}" for index in range(len(data_type.fields))]
        child_df = df.select(*(column.getField(f.name).alias(n) for f, n in zip(data_type.fields, names, strict=True)))
        return tuple(
            _aggregate(child_df, [(child_df[n], f.dataType) for f, n in zip(data_type.fields, names, strict=True)])[1]
        )
    if isinstance(data_type, T.ArrayType):
        child_df = df.select(F.explode(column).alias(_VALUE))
        return tuple(_aggregate(child_df, [(child_df[_VALUE], data_type.elementType)])[1])
    if isinstance(data_type, T.MapType):
        child_df = df.select(F.explode(column).alias("__k", "__v"))
        pairs = [(child_df["__k"], data_type.keyType), (child_df["__v"], data_type.valueType)]
        return tuple(_aggregate(child_df, pairs)[1])
    return ()


def _expressions(column: Column, data_type: T.DataType, prefix: str) -> list[Column]:
    expressions = [F.count(column).alias(f"{prefix}_non_null")]
    sketch_input = _sketch_input(column, data_type)
    if sketch_input is not None:
        sketch = F.hll_sketch_agg(sketch_input, HLL_LG_CONFIG_K)
        expressions.extend(
            [sketch.alias(f"{prefix}_sketch"), F.hll_sketch_estimate(sketch).alias(f"{prefix}_distinct")]
        )
    if isinstance(data_type, _NUMERIC_TYPES):
        approximate = column.cast("double")
        expressions.extend(
            [
                F.min(column).alias(f"{prefix}_min"),
                F.max(column).alias(f"{prefix}_max"),
                F.avg(approximate).alias(f"{prefix}_mean"),
                (F.var_pop(approximate) * F.count(approximate)).alias(f"{prefix}_m2"),
            ]
        )
    elif isinstance(data_type, _string_types()):
        expressions.extend(_length_expressions(F.length(column), prefix))
    elif isinstance(data_type, (T.ArrayType, T.MapType)):
        expressions.extend(_length_expressions(F.size(column), prefix))
    elif isinstance(data_type, _temporal_types()):
        expressions.extend(
            [
                _temporal_key(F.min(column), data_type).alias(f"{prefix}_min_key"),
                _temporal_key(F.max(column), data_type).alias(f"{prefix}_max_key"),
                _temporal_extreme_literal(F.min(column), data_type).alias(f"{prefix}_min"),
                _temporal_extreme_literal(F.max(column), data_type).alias(f"{prefix}_max"),
            ]
        )
    elif isinstance(data_type, T.BooleanType):
        expressions.append(F.count(F.when(column, True)).alias(f"{prefix}_true_count"))
    return expressions


def _length_expressions(length: Column, prefix: str) -> list[Column]:
    exact = length.cast(T.DecimalType(38, 0))
    return [
        F.count(length).alias(f"{prefix}_length_count"),
        F.sum(exact).alias(f"{prefix}_length_total"),
        F.sum(exact * exact).alias(f"{prefix}_length_squares"),
        F.min(length).alias(f"{prefix}_length_min"),
        F.max(length).alias(f"{prefix}_length_max"),
    ]


def _sketch_input(column: Column, data_type: T.DataType) -> Column | None:
    """Returns a sketchable form of a column whose distinct values match the batch engine's distinct values.

    DataSketches ignores empty strings and byte arrays, so string forms get a one-character prefix.
    """
    if isinstance(data_type, T.NullType):
        return None
    if isinstance(data_type, (T.ByteType, T.ShortType, T.IntegerType, T.LongType)):
        return column.cast("bigint")
    if isinstance(data_type, T.BinaryType):
        return F.concat(F.lit(b"v"), column)
    if isinstance(data_type, T.MapType):
        text = F.to_json(F.sort_array(F.map_entries(column)))
    elif isinstance(data_type, (T.StructType, T.ArrayType)):
        text = F.to_json(column)
    else:
        text = column.cast("string")
    return F.concat(F.lit("v"), text)


def _temporal_key(extreme: Column, data_type: T.DataType) -> Column:
    if isinstance(data_type, T.DateType):
        return F.unix_date(extreme)
    if isinstance(data_type, T.TimestampType):
        return F.unix_micros(extreme)
    if isinstance(data_type, _timestamp_ntz_types()):
        return F.unix_micros(extreme.cast("timestamp"))
    return extreme.cast("string")


def _column_state(
    values: Mapping[str, Any], prefix: str, count: int, data_type: T.DataType, children: tuple[ColumnState, ...]
) -> ColumnState:
    non_null = int(values[f"{prefix}_non_null"])
    state = ColumnState(
        count=count,
        non_null=non_null,
        sketch=values.get(f"{prefix}_sketch"),
        distinct=int(values.get(f"{prefix}_distinct") or 0),
        children=children,
    )
    if isinstance(data_type, _NUMERIC_TYPES):
        mean = values[f"{prefix}_mean"]
        moments = Moments(non_null, float(mean), float(values[f"{prefix}_m2"])) if non_null else Moments()
        return replace(state, min=values[f"{prefix}_min"], max=values[f"{prefix}_max"], moments=moments)
    if f"{prefix}_length_count" in values:
        lengths = Lengths(
            int(values[f"{prefix}_length_count"]),
            int(values[f"{prefix}_length_total"] or 0),
            int(values[f"{prefix}_length_squares"] or 0),
            values[f"{prefix}_length_min"],
            values[f"{prefix}_length_max"],
        )
        return replace(state, lengths=lengths)
    if f"{prefix}_min_key" in values:
        low = (values[f"{prefix}_min_key"], values[f"{prefix}_min"]) if non_null else None
        high = (values[f"{prefix}_max_key"], values[f"{prefix}_max"]) if non_null else None
        return replace(state, min=low, max=high)
    if isinstance(data_type, T.BooleanType):
        return replace(state, true_count=int(values[f"{prefix}_true_count"]))
    return state


def _sketch_pairs(left: ColumnState, right: ColumnState) -> Iterator[tuple[bytes, bytes]]:
    """Yields the sketch pairs that `_merge_column` unions, in the order it consumes them."""
    if left.sketch is not None and right.sketch is not None:
        yield left.sketch, right.sketch
    for a, b in zip(left.children, right.children, strict=True):
        yield from _sketch_pairs(a, b)


def _merge_column(left: ColumnState, right: ColumnState, unions: Iterator[tuple[bytes, int]]) -> ColumnState:
    sketch: bytes | None
    if left.sketch is not None and right.sketch is not None:
        sketch, distinct = next(unions)
    else:
        sketch, distinct = (left.sketch, left.distinct) if right.sketch is None else (right.sketch, right.distinct)
    return ColumnState(
        count=left.count + right.count,
        non_null=left.non_null + right.non_null,
        sketch=sketch,
        distinct=distinct,
        min=_extreme(left.min, right.min, largest=False),
        max=_extreme(left.max, right.max, largest=True),
        moments=_merge_optional(left.moments, right.moments),
        lengths=_merge_optional(left.lengths, right.lengths),
        true_count=(
            None
            if left.true_count is None and right.true_count is None
            else (left.true_count or 0) + (right.true_count or 0)
        ),
        children=tuple(_merge_column(a, b, unions) for a, b in zip(left.children, right.children, strict=True)),
    )


def _union_sketches(pairs: Sequence[tuple[bytes, bytes]], spark: SparkSession) -> list[tuple[bytes, int]]:
    """Unions sketch pairs with one Spark query, returning each union and its estimate."""
    if not pairs:
        return []
    unions = [F.hll_union(F.lit(a), F.lit(b)) for a, b in pairs]
    columns = [
        c for i, union in enumerate(unions) for c in (union.alias(f"u{i}"), F.hll_sketch_estimate(union).alias(f"e{i}"))
    ]
    row = spark.range(1).select(*columns).first()
    if row is None:
        raise RuntimeError("Spark sketch union returned no row")
    return [(row[f"u{i}"], int(row[f"e{i}"])) for i in range(len(pairs))]


def _merge_optional(left: _Mergeable | None, right: _Mergeable | None) -> _Mergeable | None:
    if left is None:
        return right
    return left if right is None else left.merge(right)


def _extreme(left: _Ordered | None, right: _Ordered | None, *, largest: bool) -> _Ordered | None:
    """Returns the smaller or larger of two optional values, ordering NaN above every float as Spark does."""
    if left is None:
        return right
    if right is None:
        return left
    choose = max if largest else min
    if isinstance(left, float) and isinstance(right, float):
        return choose(left, right, key=lambda value: (math.isnan(value), 0.0 if math.isnan(value) else value))
    return choose(left, right)


def _sample(
    df: DataFrame, schema: T.StructType, order_columns: Sequence[str], sample_size: int
) -> tuple[SampleEntry, ...]:
    """Returns the first rows of `df` in arrival order, with their keys.

    Values of types the checkpoint cannot store (intervals other than day-time, variant, spatial, null) are sampled
    as null.
    """
    if not schema.fields:
        return ()
    texts = [F.coalesce(F.col(f.name).cast("string"), F.lit("\u0000")) for f in schema.fields]
    keys = [F.col(name) for name in order_columns] + [F.xxhash64(F.lit(0), *texts), *texts]
    names = [f"__profile_key{index}" for index in range(len(keys))]
    values = [F.col(f.name) if sampleable(f.dataType) else F.lit(None).cast(f.dataType) for f in schema.fields]
    rows = (
        df.select(
            F.struct(*(v.alias(f.name) for v, f in zip(values, schema.fields, strict=True))).alias(_ROW),
            *(k.alias(n) for k, n in zip(keys, names, strict=True)),
        )
        .orderBy(*names)
        .limit(sample_size)
        .collect()
    )
    return tuple((tuple(row[n] for n in names), row[_ROW]) for row in rows)


def sampleable(data_type: T.DataType) -> bool:
    """Returns whether values of a type can be held in the PII sample of a checkpoint."""
    if isinstance(data_type, T.StructType):
        return all(sampleable(f.dataType) for f in data_type.fields)
    if isinstance(data_type, T.ArrayType):
        return sampleable(data_type.elementType)
    if isinstance(data_type, T.MapType):
        return sampleable(data_type.keyType) and sampleable(data_type.valueType)
    return isinstance(
        data_type,
        (T.BooleanType, *_NUMERIC_TYPES, *_string_types(), T.BinaryType, *_temporal_types(), T.DayTimeIntervalType),
    )


def _child_types(data_type: T.DataType) -> list[T.DataType]:
    return [f.dataType for f in _child_fields(data_type)]


def _child_fields(data_type: T.DataType) -> list[T.StructField]:
    if isinstance(data_type, T.StructType):
        return [T.StructField(f.name, f.dataType, f.nullable) for f in data_type.fields]
    if isinstance(data_type, T.ArrayType):
        return [T.StructField("element", data_type.elementType, data_type.containsNull)]
    if isinstance(data_type, T.MapType):
        return [
            T.StructField("key", data_type.keyType, False),
            T.StructField("value", data_type.valueType, data_type.valueContainsNull),
        ]
    return []
