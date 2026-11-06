"""Bounded, deterministic driver-side samples of Spark tables, flattened into leaf columns.

A table is flattened into leaves: every primitive top-level column and struct field, plus the length of every array,
map, and binary value. Array elements and map entries are exploded into their own frames, so nested values are
compared as values, not rows. Rows are selected by a seeded hash of their content, which is independent of
partitioning and selects identical rows on both sides of a comparison.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd
from pyspark.sql import Column as SparkColumn
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from profiler_bench.catalog.manifest import Column, Table


CHARACTER_CLASSES: tuple[tuple[str, Callable[[str], bool]], ...] = (
    ("#digits", str.isdigit),
    ("#letters", str.isalpha),
    ("#punctuation", lambda c: not c.isalnum() and not c.isspace()),
)
ROW_FRAME = ""
HASH = "__h"
HASH_RANGE = 2**62
UTC_ZONES = ("UTC", "Etc/UTC", "Z")


@dataclass(frozen=True)
class Leaf:
    """One flattened value of a table.

    Attributes:
        path: Dotted path; `[]` marks array elements, `[key]` and `[value]` map entries, `#length` a length.
        root: Top-level column the leaf belongs to.
        frame: Path of the exploded collection the leaf lives in; empty for row-level leaves.
        dtype: `number` (numbers, days for dates, microseconds for timestamps), `boolean`, `string`, or `length`.
        top_level: Whether the leaf is a top-level column.
        truth: Manifest column of the leaf, when declared.
    """

    path: str
    root: str
    frame: str
    dtype: str
    top_level: bool
    truth: Column | None


@dataclass(frozen=True)
class _Planned:
    leaf: Leaf
    expr: str


@dataclass(frozen=True)
class _Collection:
    path: str
    root: str
    expr: str
    dtype: T.ArrayType | T.MapType
    truth: Column | None


def string_features(values: pd.Series, formats: Sequence[str]) -> dict[str, pd.Series]:
    """Returns features derived from a string column, keyed by the suffix added to its path.

    `#chars` is the length, `#digits`, `#letters`, and `#punctuation` count character classes, and `#format` (when
    the manifest declares formats) is the index of the first format the value fully matches, or `unmatched`. They let
    joint structure of identifier-like columns be compared without comparing the values themselves.
    """
    present = values.dropna().astype(str)
    features = {"#chars": present.map(len)}
    for suffix, test in CHARACTER_CLASSES:
        features[suffix] = present.map(lambda s, t=test: sum(map(t, s)))
    if formats:
        compiled = [re.compile(f) for f in formats]
        features["#format"] = present.map(
            lambda s: next((str(i) for i, p in enumerate(compiled) if p.fullmatch(s)), "unmatched")
        )
    return {suffix: feature.reindex(values.index) for suffix, feature in features.items()}


def leaf_kind(leaf: Leaf, values: pd.Series, max_distinct: int) -> str:
    """Returns how a leaf is compared, judged from its real values.

    Lengths are `length`; booleans, and numbers or strings with at most `max_distinct` distinct values, are
    `categorical`; other numbers are `numeric` and other strings `string`.
    """
    if leaf.dtype == "length":
        return "length"
    if leaf.dtype == "boolean" or values.dropna().nunique() <= max_distinct:
        return "categorical"
    return "numeric" if leaf.dtype == "number" else "string"


def require_utc(frames: Sequence[DataFrame]) -> None:
    """Raises ValueError unless the Spark session time zone is UTC, which timestamp leaves assume."""
    if frames and frames[0].sparkSession.conf.get("spark.sql.session.timeZone") not in UTC_ZONES:
        raise ValueError("fidelity metrics need spark.sql.session.timeZone=UTC")


def identifier(name: str) -> str:
    """Returns `name` as a back-quoted Spark SQL identifier."""
    return f"`{name.replace('`', '``')}`"


def quoted(name: str) -> SparkColumn:
    """Returns the column called `name`, quoted so dots and spaces are literal."""
    return F.col(identifier(name))


def row_hash(df: DataFrame, seed: int) -> SparkColumn:
    """Returns a seeded 64-bit hash of each row's full content."""
    columns = ", ".join(identifier(c) for c in df.columns)
    return F.expr(f"xxhash64({int(seed)}, to_json(struct({columns})))")


def bounded(df: DataFrame, rows: int, cap: int, seed: int) -> DataFrame:
    """Returns about `cap` rows of `df`, chosen by `row_hash`, with the hash as column `__h`."""
    hashed = df.withColumn(HASH, row_hash(df, seed))
    if rows <= cap:
        return hashed
    threshold = int(HASH_RANGE * cap / rows)
    return hashed.where(F.pmod(F.col(HASH), F.lit(HASH_RANGE)) < F.lit(threshold))


def collect(
    df: DataFrame, truth: Table | None, rows: int, cap: int, seed: int
) -> tuple[list[Leaf], dict[str, pd.DataFrame]]:
    """Returns the leaves of `df` and their sampled values as one pandas frame per collection, keyed by `Leaf.frame`.

    The row frame is sorted by the row hash and keeps it as column `__h`; leaf columns are named by `Leaf.path`.
    """
    sample = bounded(df, rows, cap, seed).cache()
    try:
        planned, collections = _plan_frame(sample, ROW_FRAME, truth)
        exprs = [*(f"{p.expr} AS c{i}" for i, p in enumerate(planned)), HASH]
        row: pd.DataFrame = sample.selectExpr(*exprs).toPandas()
        row.columns = [*(p.leaf.path for p in planned), HASH]
        frames = {ROW_FRAME: row.sort_values(HASH, kind="stable").reset_index(drop=True)}
        found = [p.leaf for p in planned]
        sources = {c.path: sample for c in collections}
        while collections:
            collection = collections.pop(0)
            exploded = _explode(sources[collection.path], collection)
            frame_planned, nested = _plan_collection(collection)
            found += [p.leaf for p in frame_planned]
            frame: pd.DataFrame = exploded.selectExpr(
                *(f"{p.expr} AS c{i}" for i, p in enumerate(frame_planned))
            ).toPandas()
            frame.columns = [p.leaf.path for p in frame_planned]
            frames[collection.path] = frame
            sources.update({c.path: exploded for c in nested})
            collections += nested
        return found, frames
    finally:
        sample.unpersist()


def _explode(source: DataFrame, collection: _Collection) -> DataFrame:
    if isinstance(collection.dtype, T.MapType):
        return source.selectExpr(f"explode({collection.expr}) AS (key, value)")
    return source.selectExpr(f"explode({collection.expr}) AS element")


def _plan_collection(collection: _Collection) -> tuple[list[_Planned], list[_Collection]]:
    truth, path, root = collection.truth, collection.path, collection.root
    if isinstance(collection.dtype, T.MapType):
        key = _plan("key", collection.dtype.keyType, f"{path}[key]", root, path, truth and truth.key)
        value = _plan("value", collection.dtype.valueType, f"{path}[value]", root, path, truth and truth.value)
        return key[0] + value[0], key[1] + value[1]
    return _plan("element", collection.dtype.elementType, f"{path}[]", root, path, truth and truth.element)


def _plan_frame(df: DataFrame, frame: str, truth: Table | None) -> tuple[list[_Planned], list[_Collection]]:
    columns = {c.name: c for c in truth.columns} if truth is not None else {}
    planned: list[_Planned] = []
    collections: list[_Collection] = []
    for field in df.schema.fields:
        if field.name == HASH:
            continue
        p, c = _plan(
            identifier(field.name), field.dataType, field.name, field.name, frame, columns.get(field.name), True
        )
        planned += p
        collections += c
    return planned, collections


def _plan(
    expr: str,
    dtype: T.DataType,
    path: str,
    root: str,
    frame: str,
    truth: Column | None,
    top_level: bool = False,
) -> tuple[list[_Planned], list[_Collection]]:
    if isinstance(dtype, T.StructType):
        fields = {f.name: f for f in truth.fields} if truth is not None else {}
        planned: list[_Planned] = []
        collections: list[_Collection] = []
        for field in dtype.fields:
            p, c = _plan(
                f"{expr}.{identifier(field.name)}",
                field.dataType,
                f"{path}.{field.name}",
                root,
                frame,
                fields.get(field.name),
            )
            planned += p
            collections += c
        return planned, collections
    if isinstance(dtype, T.ArrayType | T.MapType | T.BinaryType):
        binary = isinstance(dtype, T.BinaryType)
        length = (
            f"CAST(length({expr}) AS DOUBLE)" if binary else f"CAST(IF({expr} IS NULL, NULL, size({expr})) AS DOUBLE)"
        )
        leaf = Leaf(f"{path}#length", root, frame, "length", top_level, truth)
        nested = [] if isinstance(dtype, T.BinaryType) else [_Collection(path, root, expr, dtype, truth)]
        return [_Planned(leaf, length)], nested
    converted = _convert(expr, dtype)
    if converted is None:
        return [], []
    value, kind = converted
    return [_Planned(Leaf(path, root, frame, kind, top_level, truth), value)], []


_CONVERSIONS: tuple[tuple[tuple[type[T.DataType], ...], str, str], ...] = (
    ((T.BooleanType,), "{}", "boolean"),
    ((T.StringType, T.VarcharType, T.CharType), "{}", "string"),
    ((T.DateType,), "CAST(unix_date({}) AS DOUBLE)", "number"),
    ((T.TimestampNTZType,), "CAST(unix_micros(CAST({} AS TIMESTAMP)) AS DOUBLE)", "number"),
    ((T.TimestampType,), "CAST(unix_micros({}) AS DOUBLE)", "number"),
    ((T.NumericType,), "CAST({} AS DOUBLE)", "number"),
)


def _convert(expr: str, dtype: T.DataType) -> tuple[str, str] | None:
    for types, template, kind in _CONVERSIONS:
        if isinstance(dtype, types):
            return template.format(expr), kind
    return None
