"""Baseline PySpark profiling engine."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import date, datetime, time, timezone
from decimal import Decimal
from math import isfinite
from typing import Any

import pyspark
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from dbldatagen_profiler.model import (
    SPEC_VERSION,
    BooleanStats,
    ColumnProfile,
    DistinctCount,
    EngineInfo,
    LengthStats,
    LogicalType,
    NumericStats,
    PrivacyPolicy,
    Profile,
    QuasiIdentifierSet,
    StringStats,
    TableProfile,
    TemporalStats,
    ValueTreatment,
)
from dbldatagen_profiler.pii.detect import analyze_table


_VALUE = "__profile_value"
_NUMERIC_TYPES = (
    T.ByteType,
    T.ShortType,
    T.IntegerType,
    T.LongType,
    T.FloatType,
    T.DoubleType,
    T.DecimalType,
)
_TEMPORAL_TYPES = (T.DateType, T.TimestampType)


def profile_tables(
    tables: Mapping[str, DataFrame], *, privacy: PrivacyPolicy | None = None, detect_pii: bool = True
) -> Profile:
    """Profiles named Spark DataFrames.

    Args:
        tables: DataFrames keyed by their profile table name.
        privacy: Optional privacy policy applied to emitted literals.
        detect_pii: Whether to detect sensitive semantics and quasi-identifiers (default true).

    Returns:
        A validated profile containing every input table.

    Raises:
        ValueError: If no tables are supplied.
    """
    if not tables:
        raise ValueError("at least one table is required")
    policy = privacy or PrivacyPolicy()
    with _utc_session_time_zones(tables.values()):
        analyses = {name: analyze_table(frame) for name, frame in tables.items()} if detect_pii else {}
        result = Profile(
            spec_version=SPEC_VERSION,
            created_at=datetime.now(timezone.utc),
            engine=EngineInfo(name="pyspark", version=pyspark.__version__),
            privacy=policy,
            tables=[_profile_table(name, frame, policy, analyses.get(name)) for name, frame in tables.items()],
        )
        return Profile.model_validate(result.model_dump(), context={"strict": True})


def profile_dataframe(
    name: str, df: DataFrame, *, privacy: PrivacyPolicy | None = None, detect_pii: bool = True
) -> Profile:
    """Profiles one Spark DataFrame.

    Args:
        name: Profile table name.
        df: Spark DataFrame to profile.
        privacy: Optional privacy policy applied to emitted literals.
        detect_pii: Whether to detect sensitive semantics and quasi-identifiers (default true).

    Returns:
        A validated single-table profile.
    """
    return profile_tables({name: df}, privacy=privacy, detect_pii=detect_pii)


def _profile_table(
    name: str,
    df: DataFrame,
    privacy: PrivacyPolicy,
    analysis: tuple[dict[str, Any], list[tuple[list[str], float, int]]] | None,
) -> TableProfile:
    semantics, quasi_sets = analysis or ({}, [])
    expressions: list[Column] = [F.count(F.lit(1)).alias("__row_count")]
    for index, field in enumerate(df.schema.fields):
        expressions.extend(_aggregate_expressions(df[field.name], field.dataType, index))
    values = _collect_aggregate(df, expressions)
    row_count = int(values["__row_count"])
    columns = [
        _column_from_aggregate(field, values, index, row_count, privacy, nested=False, df=df)
        for index, field in enumerate(df.schema.fields)
    ]
    protected = [_protect_column(column, column.name, semantics) for column in columns]
    quasi = [
        QuasiIdentifierSet(columns=names, unique_fraction=unique, k_anonymity=k) for names, unique, k in quasi_sets
    ]
    return TableProfile(name=name, row_count=row_count, columns=protected, quasi_identifier_sets=quasi or None)


def _protect_column(column: ColumnProfile, path: str, semantics: Mapping[str, Any]) -> ColumnProfile:
    updates: dict[str, Any] = {}
    semantic = semantics.get(path)
    if semantic is not None:
        updates["semantic"] = semantic
        if semantic.sensitivity.value in {"direct_identifier", "secret"}:
            if column.numeric is not None:
                updates["numeric"] = column.numeric.model_copy(
                    update={"min": None, "max": None, "quantiles": None, "histogram": None}
                )
            if column.temporal is not None:
                updates["temporal"] = column.temporal.model_copy(update={"min": None, "max": None, "quantiles": None})
            if column.string is not None:
                updates["string"] = column.string.model_copy(update={"top_values": None})
    if column.children is not None:
        updates["children"] = [_protect_column(child, f"{path}.{child.name}", semantics) for child in column.children]
    return column.model_copy(update=updates) if updates else column


@contextmanager
def _utc_session_time_zones(frames: Iterable[DataFrame]) -> Iterator[None]:
    """Temporarily configures each input Spark session to use UTC."""
    sessions: list[tuple[SparkSession, str | None]] = []
    seen: set[int] = set()
    key = "spark.sql.session.timeZone"
    try:
        for frame in frames:
            session = frame.sparkSession
            if id(session) in seen:
                continue
            seen.add(id(session))
            sessions.append((session, session.conf.get(key, None)))
            session.conf.set(key, "UTC")
        yield
    finally:
        for session, previous in reversed(sessions):
            if previous is None:
                session.conf.unset(key)
            else:
                session.conf.set(key, previous)


def _aggregate_expressions(column: Column, data_type: T.DataType, index: int) -> list[Column]:
    prefix = f"c{index}"
    if isinstance(data_type, T.MapType):
        distinct_value = F.to_json(F.sort_array(F.map_entries(column)))
    elif isinstance(data_type, (T.StructType, T.ArrayType)):
        distinct_value = F.to_json(column)
    elif isinstance(data_type, (*_variant_types(), *_spatial_types())):
        distinct_value = column.cast("string")
    else:
        distinct_value = column
    expressions = [
        F.count(column).alias(f"{prefix}_non_null"),
        F.countDistinct(distinct_value).alias(f"{prefix}_distinct"),
    ]
    if isinstance(data_type, _NUMERIC_TYPES):
        approximate = column.cast("double")
        expressions.extend(
            [
                F.min(column).alias(f"{prefix}_min"),
                F.max(column).alias(f"{prefix}_max"),
                F.avg(approximate).alias(f"{prefix}_mean"),
                F.stddev_samp(approximate).alias(f"{prefix}_stddev"),
            ]
        )
    elif isinstance(data_type, _string_types()):
        length = F.length(column)
        expressions.extend(
            [
                F.min(length).alias(f"{prefix}_length_min"),
                F.max(length).alias(f"{prefix}_length_max"),
                F.avg(length).alias(f"{prefix}_length_mean"),
            ]
        )
    elif isinstance(data_type, _temporal_types()):
        expressions.extend(
            [
                F.min(column).cast("string").alias(f"{prefix}_min"),
                F.max(column).cast("string").alias(f"{prefix}_max"),
            ]
        )
    elif isinstance(data_type, T.BooleanType):
        expressions.append(F.avg(column.cast("double")).alias(f"{prefix}_true_fraction"))
    elif isinstance(data_type, (T.ArrayType, T.MapType)):
        length = F.size(column)
        expressions.extend(
            [
                F.min(length).alias(f"{prefix}_length_min"),
                F.max(length).alias(f"{prefix}_length_max"),
                F.avg(length).alias(f"{prefix}_length_mean"),
                F.stddev_samp(length).alias(f"{prefix}_length_stddev"),
            ]
        )
    return expressions


def _column_from_aggregate(
    field: T.StructField,
    values: Mapping[str, Any],
    index: int,
    value_count: int,
    privacy: PrivacyPolicy,
    *,
    nested: bool,
    df: DataFrame,
) -> ColumnProfile:
    prefix = f"c{index}"
    non_null = int(values[f"{prefix}_non_null"])
    logical_type = _logical_type(field.dataType)
    attributes: dict[str, Any] = {
        "name": field.name,
        "logical_type": logical_type,
        "source_type": field.dataType.simpleString(),
        "nullable": field.nullable,
        "null_fraction": (value_count - non_null) / value_count if value_count else 0.0,
        "distinct_count": DistinctCount(value=int(values[f"{prefix}_distinct"]), exact=True),
    }
    if nested:
        attributes["value_count"] = value_count
    if isinstance(field.dataType, T.DecimalType):
        attributes.update(precision=field.dataType.precision, scale=field.dataType.scale)
    if isinstance(field.dataType, _NUMERIC_TYPES):
        attributes["numeric"] = _numeric_stats(field.dataType, values, prefix, privacy)
    elif isinstance(field.dataType, _string_types()):
        attributes["string"] = StringStats(length=_length_stats(values, prefix, include_stddev=False))
    elif isinstance(field.dataType, _temporal_types()):
        attributes["temporal"] = _temporal_stats(field.dataType, values, prefix, privacy)
    elif isinstance(field.dataType, T.BooleanType) and values.get(f"{prefix}_true_fraction") is not None:
        attributes["boolean"] = BooleanStats(true_fraction=values[f"{prefix}_true_fraction"])
    elif isinstance(field.dataType, (T.ArrayType, T.MapType)):
        attributes["collection_length"] = _length_stats(values, prefix, include_stddev=True)
    if isinstance(field.dataType, (T.StructType, T.ArrayType, T.MapType)):
        nested_df = df if nested else df.select(df[field.name].alias(_VALUE))
        attributes["children"] = _profile_children(nested_df, field, privacy)
    return ColumnProfile(**attributes)


def _profile_children(df: DataFrame, field: T.StructField, privacy: PrivacyPolicy) -> list[ColumnProfile]:
    data_type = field.dataType
    value = df[_VALUE]
    if isinstance(data_type, T.StructType):
        return [
            _profile_nested_column(
                df.select(value.getField(child.name).alias(_VALUE)),
                T.StructField(child.name, child.dataType, child.nullable),
                privacy,
            )
            for child in data_type.fields
        ]
    if isinstance(data_type, T.ArrayType):
        child_df = df.select(F.explode(value).alias(_VALUE))
        return [
            _profile_nested_column(
                child_df, T.StructField("element", data_type.elementType, data_type.containsNull), privacy
            )
        ]
    if not isinstance(data_type, T.MapType):
        raise TypeError(f"unsupported nested Spark data type: {data_type}")
    exploded = df.select(F.explode(value).alias("key", "value"))
    return [
        _profile_nested_column(
            exploded.select(F.col(source).alias(_VALUE)),
            T.StructField(name, child_type, nullable),
            privacy,
        )
        for name, source, child_type, nullable in (
            ("key", "key", data_type.keyType, False),
            ("value", "value", data_type.valueType, data_type.valueContainsNull),
        )
    ]


def _profile_nested_column(df: DataFrame, field: T.StructField, privacy: PrivacyPolicy) -> ColumnProfile:
    expressions = [F.count(F.lit(1)).alias("__row_count"), *_aggregate_expressions(df[_VALUE], field.dataType, 0)]
    values = _collect_aggregate(df, expressions)
    return _column_from_aggregate(field, values, 0, int(values["__row_count"]), privacy, nested=True, df=df)


def _collect_aggregate(df: DataFrame, expressions: list[Column]) -> dict[str, Any]:
    """Collects the single row produced by aggregate expressions."""
    row = df.agg(*expressions).first()
    if row is None:
        raise RuntimeError("Spark aggregate returned no row")
    return row.asDict()


def _numeric_stats(
    data_type: T.DataType, values: Mapping[str, Any], prefix: str, privacy: PrivacyPolicy
) -> NumericStats:
    include_extremes = privacy.extremes != ValueTreatment.SUPPRESSED
    return NumericStats(
        min=_encode_numeric(values.get(f"{prefix}_min"), data_type) if include_extremes else None,
        max=_encode_numeric(values.get(f"{prefix}_max"), data_type) if include_extremes else None,
        mean=_finite_float(values.get(f"{prefix}_mean")),
        stddev=_finite_float(values.get(f"{prefix}_stddev")),
    )


def _encode_numeric(value: object, data_type: T.DataType) -> int | float | str | None:
    if value is None:
        return None
    if isinstance(data_type, (T.LongType, T.DecimalType)):
        return str(value) if not isinstance(value, Decimal) else format(value, "f")
    if isinstance(data_type, (T.FloatType, T.DoubleType)):
        return _finite_float(value)
    if isinstance(value, int | float):
        return value
    raise TypeError(f"unexpected numeric value {value!r}")


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    result = float(str(value))
    return result if isfinite(result) else None


def _length_stats(values: Mapping[str, Any], prefix: str, *, include_stddev: bool) -> LengthStats:
    return LengthStats(
        min=values.get(f"{prefix}_length_min"),
        max=values.get(f"{prefix}_length_max"),
        mean=_finite_float(values.get(f"{prefix}_length_mean")),
        stddev=_finite_float(values.get(f"{prefix}_length_stddev")) if include_stddev else None,
    )


def _temporal_stats(
    data_type: T.DataType, values: Mapping[str, Any], prefix: str, privacy: PrivacyPolicy
) -> TemporalStats:
    if privacy.extremes == ValueTreatment.SUPPRESSED:
        return TemporalStats()
    return TemporalStats(
        min=_encode_temporal(values.get(f"{prefix}_min"), data_type),
        max=_encode_temporal(values.get(f"{prefix}_max"), data_type),
    )


def _encode_temporal(value: object, data_type: T.DataType) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _encode_temporal_string(value, data_type)
    if isinstance(data_type, T.DateType) and isinstance(value, date):
        return value.isoformat()
    if isinstance(data_type, _time_types()) and isinstance(value, time):
        return value.isoformat()
    if not isinstance(value, datetime):
        raise TypeError(f"unexpected temporal value {value!r}")
    rendered = value.isoformat()
    if isinstance(data_type, _timestamp_ntz_types()):
        return rendered
    return value.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _encode_temporal_string(value: str, data_type: T.DataType) -> str:
    """Encodes a temporal value rendered in the UTC Spark session."""
    rendered = value.replace(" ", "T")
    if isinstance(data_type, (*_timestamp_ntz_types(), T.DateType, *_time_types())):
        return rendered
    return f"{rendered}Z"


def _logical_type(data_type: T.DataType) -> LogicalType:
    mappings: list[tuple[type[T.DataType], LogicalType]] = [
        (T.NullType, LogicalType.NULL),
        (T.BooleanType, LogicalType.BOOLEAN),
        (T.ByteType, LogicalType.INT8),
        (T.ShortType, LogicalType.INT16),
        (T.IntegerType, LogicalType.INT32),
        (T.LongType, LogicalType.INT64),
        (T.FloatType, LogicalType.FLOAT32),
        (T.DoubleType, LogicalType.FLOAT64),
        (T.DecimalType, LogicalType.DECIMAL),
        (T.StringType, LogicalType.STRING),
        *((kind, LogicalType.STRING) for kind in _bounded_string_types()),
        (T.BinaryType, LogicalType.BINARY),
        (T.DateType, LogicalType.DATE),
        *((kind, LogicalType.TIME) for kind in _time_types()),
        (T.TimestampType, LogicalType.TIMESTAMP),
        *((kind, LogicalType.TIMESTAMP_NTZ) for kind in _timestamp_ntz_types()),
        (T.CalendarIntervalType, LogicalType.INTERVAL),
        *((kind, LogicalType.INTERVAL) for kind in _interval_types()),
        (T.StructType, LogicalType.STRUCT),
        (T.ArrayType, LogicalType.ARRAY),
        (T.MapType, LogicalType.MAP),
        *((kind, LogicalType.VARIANT) for kind in _variant_types()),
        *((kind, LogicalType.VARIANT) for kind in _spatial_types()),
    ]
    for spark_type, logical_type in mappings:
        if isinstance(data_type, spark_type):
            return logical_type
    if isinstance(data_type, T.UserDefinedType):
        return _logical_type(data_type.sqlType())
    raise TypeError(f"unsupported Spark data type: {data_type}")


def _timestamp_ntz_types() -> tuple[type[T.DataType], ...]:
    kind = getattr(T, "TimestampNTZType", None)
    return (kind,) if kind is not None else ()


def _interval_types() -> tuple[type[T.DataType], ...]:
    return tuple(
        kind
        for name in ("DayTimeIntervalType", "YearMonthIntervalType")
        if (kind := getattr(T, name, None)) is not None
    )


def _variant_types() -> tuple[type[T.DataType], ...]:
    kind = getattr(T, "VariantType", None)
    return (kind,) if kind is not None else ()


def _temporal_types() -> tuple[type[T.DataType], ...]:
    return (*_TEMPORAL_TYPES, *_time_types(), *_timestamp_ntz_types())


def _bounded_string_types() -> tuple[type[T.DataType], ...]:
    return tuple(kind for name in ("CharType", "VarcharType") if (kind := getattr(T, name, None)) is not None)


def _string_types() -> tuple[type[T.DataType], ...]:
    return (T.StringType, *_bounded_string_types())


def _time_types() -> tuple[type[T.DataType], ...]:
    kind = getattr(T, "TimeType", None)
    return (kind,) if kind is not None else ()


def _spatial_types() -> tuple[type[T.DataType], ...]:
    return tuple(kind for name in ("GeometryType", "GeographyType") if (kind := getattr(T, name, None)) is not None)
