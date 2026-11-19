"""Versioned JSON checkpoint of streaming profiling state.

The file is plain JSON, so loading it never executes code. Sketches are base64 bytes; int64, decimal, and exact
length totals are decimal strings; floats are hexadecimal strings (`float.hex`), so NaN, infinities, and every bit
survive. PII sample rows are JSON values typed by the table schema.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from pyspark.sql import Row
from pyspark.sql import types as T

from dbldatagen_profiler.engines.spark.engine import (
    _NUMERIC_TYPES,
    _string_types,
    _temporal_types,
    _time_types,
    _timestamp_ntz_types,
)
from dbldatagen_profiler.engines.streaming.state import (
    ColumnState,
    Lengths,
    Moments,
    PiiAnalysis,
    TableState,
    _child_types,
    sampleable,
)
from dbldatagen_profiler.model import Semantic


VERSION = 1
BatchRanges = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Checkpoint:
    """Persisted profiling state of one streaming table.

    Attributes:
        batches: Merged micro-batch ids, as sorted, disjoint, inclusive `(first, last)` ranges.
        state: Merged state of every merged micro-batch.
        pii: PII analysis frozen after the warmup rows, or None before.
    """

    batches: BatchRanges
    state: TableState
    pii: PiiAnalysis | None = None

    def has_batch(self, batch_id: int) -> bool:
        """Returns whether a micro-batch id has been merged."""
        return any(first <= batch_id <= last for first, last in self.batches)


def add_batch(batches: BatchRanges, batch_id: int) -> BatchRanges:
    """Returns `batches` with `batch_id` added, joining adjacent ranges."""
    ranges = sorted([*batches, (batch_id, batch_id)])
    joined: list[tuple[int, int]] = []
    for first, last in ranges:
        if joined and first <= joined[-1][1] + 1:
            joined[-1] = (joined[-1][0], max(joined[-1][1], last))
        else:
            joined.append((first, last))
    return tuple(joined)


def write_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    """Writes a checkpoint atomically: to a temporary file that replaces `path`, then syncs the directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(encode_checkpoint(checkpoint), handle, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_checkpoint(path: Path) -> Checkpoint:
    """Reads a checkpoint written by `write_checkpoint`.

    Raises:
        ValueError: If the file is not a valid checkpoint of a supported version.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"checkpoint {path} is not valid JSON: {error}") from error
    if not isinstance(data, dict) or data.get("version") != VERSION:
        raise ValueError(f"checkpoint {path} has unsupported version {_version(data)!r}; expected {VERSION}")
    try:
        return decode_checkpoint(data)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, InvalidOperation, binascii.Error) as error:
        raise ValueError(f"checkpoint {path} is invalid: {error!r}") from error


def _version(data: object) -> object:
    return data.get("version") if isinstance(data, dict) else None


def encode_checkpoint(checkpoint: Checkpoint) -> dict[str, Any]:
    """Returns the JSON object of a checkpoint."""
    state = checkpoint.state
    fields = state.schema.fields
    return {
        "version": VERSION,
        "batches": [list(item) for item in checkpoint.batches],
        "schema": state.schema.jsonValue(),
        "row_count": state.row_count,
        "sample_size": state.sample_size,
        "columns": [_encode_column(c, f.dataType) for c, f in zip(state.columns, fields, strict=True)],
        "sample": [{"key": list(key), "row": _encode_value(row, state.schema)} for key, row in state.sample],
        "pii": None if checkpoint.pii is None else _encode_pii(checkpoint.pii),
    }


def decode_checkpoint(data: dict[str, Any]) -> Checkpoint:
    """Returns the checkpoint of a JSON object written by `encode_checkpoint`."""
    schema = T.StructType.fromJson(data["schema"])
    columns = data["columns"]
    if len(columns) != len(schema.fields):
        raise ValueError("column states do not match the schema")
    state = TableState(
        schema=schema,
        row_count=_integer(data["row_count"]),
        columns=tuple(_decode_column(c, f.dataType) for c, f in zip(columns, schema.fields, strict=True)),
        sample=tuple((_key(entry["key"]), _decode_value(entry["row"], schema)) for entry in data["sample"]),
        sample_size=_integer(data["sample_size"]),
    )
    batches = tuple((_integer(first), _integer(last)) for first, last in data["batches"])
    pii = None if data["pii"] is None else _decode_pii(data["pii"])
    return Checkpoint(batches, state, pii)


def _encode_column(state: ColumnState, data_type: T.DataType) -> dict[str, Any]:
    moments, lengths = state.moments, state.lengths
    return {
        "count": state.count,
        "non_null": state.non_null,
        "sketch": None if state.sketch is None else base64.b64encode(state.sketch).decode("ascii"),
        "distinct": state.distinct,
        "min": _encode_extreme(state.min, data_type),
        "max": _encode_extreme(state.max, data_type),
        "moments": None if moments is None else [moments.count, moments.mean.hex(), moments.m2.hex()],
        "lengths": (
            None
            if lengths is None
            else [lengths.count, str(lengths.total), str(lengths.total_squares), lengths.min, lengths.max]
        ),
        "true_count": state.true_count,
        "children": [_encode_column(c, t) for c, t in zip(state.children, _child_types(data_type), strict=True)],
    }


def _decode_column(data: dict[str, Any], data_type: T.DataType) -> ColumnState:
    moments, lengths = data["moments"], data["lengths"]
    child_types = _child_types(data_type)
    if len(data["children"]) != len(child_types):
        raise ValueError("nested column states do not match the schema")
    return ColumnState(
        count=_integer(data["count"]),
        non_null=_integer(data["non_null"]),
        sketch=None if data["sketch"] is None else base64.b64decode(data["sketch"], validate=True),
        distinct=_integer(data["distinct"]),
        min=_decode_extreme(data["min"], data_type),
        max=_decode_extreme(data["max"], data_type),
        moments=(
            None
            if moments is None
            else Moments(_integer(moments[0]), float.fromhex(moments[1]), float.fromhex(moments[2]))
        ),
        lengths=(
            None
            if lengths is None
            else Lengths(
                _integer(lengths[0]),
                int(lengths[1]),
                int(lengths[2]),
                _optional_int(lengths[3]),
                _optional_int(lengths[4]),
            )
        ),
        true_count=_optional_int(data["true_count"]),
        children=tuple(_decode_column(c, t) for c, t in zip(data["children"], child_types, strict=True)),
    )


def _encode_extreme(value: object, data_type: T.DataType) -> object:
    if value is None:
        return None
    if isinstance(value, tuple):
        return list(value)
    return _encode_number(value)


def _decode_extreme(value: object, data_type: T.DataType) -> object:
    if value is None:
        return None
    if isinstance(data_type, _temporal_types()):
        if not isinstance(value, list) or len(value) != 2 or not isinstance(value[1], str):
            raise ValueError(f"invalid temporal extreme {value!r}")
        return (_key([value[0]])[0], value[1])
    if not isinstance(data_type, _NUMERIC_TYPES):
        raise ValueError(f"unexpected extreme for {data_type.simpleString()}")
    return _decode_number(value, data_type)


def _encode_number(value: object) -> str:
    if isinstance(value, bool):
        raise TypeError(f"unexpected boolean {value!r}")
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, int):
        return str(value)
    raise TypeError(f"unexpected number {value!r}")


def _decode_number(value: object, data_type: T.DataType) -> int | float | Decimal:
    if not isinstance(value, str):
        raise ValueError(f"invalid number {value!r}")
    if isinstance(data_type, (T.FloatType, T.DoubleType)):
        return float.fromhex(value)
    if isinstance(data_type, T.DecimalType):
        return Decimal(value)
    return int(value)


def _encode_value(value: object, data_type: T.DataType) -> object:
    """Encodes a collected Spark value as JSON, by its Spark type."""
    if value is None or not sampleable(data_type):
        return None
    if isinstance(data_type, T.StructType):
        return {f.name: _encode_value(value[f.name], f.dataType) for f in data_type.fields}  # type: ignore[index]
    if isinstance(data_type, T.ArrayType):
        return [_encode_value(item, data_type.elementType) for item in value]  # type: ignore[attr-defined]
    if isinstance(data_type, T.MapType):
        return [
            [_encode_value(k, data_type.keyType), _encode_value(v, data_type.valueType)]
            for k, v in value.items()  # type: ignore[attr-defined]
        ]
    return _codec(data_type)[0](value)


def _decode_value(value: object, data_type: T.DataType) -> Any:  # noqa: ANN401
    """Decodes a JSON value written by `_encode_value` to the Python value Spark collects."""
    if value is None:
        return None
    if not sampleable(data_type):
        raise ValueError(f"unexpected sample value for {data_type.simpleString()}")
    if isinstance(data_type, T.StructType):
        if not isinstance(value, dict):
            raise ValueError(f"invalid struct value {value!r}")
        return Row(**{f.name: _decode_value(value[f.name], f.dataType) for f in data_type.fields})
    if isinstance(data_type, T.ArrayType):
        return [_decode_value(item, data_type.elementType) for item in _list(value)]
    if isinstance(data_type, T.MapType):
        return {
            _decode_value(k, data_type.keyType): _decode_value(v, data_type.valueType)
            for k, v in (_pair(item) for item in _list(value))
        }
    return _codec(data_type)[1](value)


def _codec(data_type: T.DataType) -> tuple[Callable[[Any], object], Callable[[object], object]]:
    """Returns the JSON encoder and decoder of a sampleable scalar type."""
    codecs: list[tuple[tuple[type[T.DataType], ...], Callable[[Any], object], Callable[[object], object]]] = [
        ((T.BooleanType,), _same, _boolean),
        (_string_types(), _same, _text),
        ((T.BinaryType,), _base64, lambda v: bytearray(base64.b64decode(_text(v), validate=True))),
        (
            (T.DayTimeIntervalType,),
            lambda v: str(v // timedelta(microseconds=1)),
            lambda v: timedelta(microseconds=int(_text(v))),
        ),
        ((T.DateType,), _isoformat, lambda v: date.fromisoformat(_text(v))),
        ((T.TimestampType, *_timestamp_ntz_types()), _isoformat, lambda v: datetime.fromisoformat(_text(v))),
        (_time_types(), _isoformat, lambda v: time.fromisoformat(_text(v))),
        (_NUMERIC_TYPES, _encode_number, lambda v: _decode_number(v, data_type)),
    ]
    for types, encode, decode in codecs:
        if isinstance(data_type, types):
            return encode, decode
    raise ValueError(f"no checkpoint encoding for {data_type.simpleString()}")


def _same(value: object) -> object:
    return value


def _base64(value: bytes) -> str:
    return base64.b64encode(bytes(value)).decode("ascii")


def _isoformat(value: date | time) -> str:
    return value.isoformat()


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"invalid boolean {value!r}")
    return value


def _encode_pii(pii: PiiAnalysis) -> dict[str, Any]:
    semantics, quasi = pii
    return {
        "semantics": {path: semantic.model_dump(mode="json") for path, semantic in semantics.items()},
        "quasi_identifier_sets": [[columns, unique, k] for columns, unique, k in quasi],
    }


def _decode_pii(data: dict[str, Any]) -> PiiAnalysis:
    semantics = {str(path): Semantic.model_validate(value) for path, value in data["semantics"].items()}
    quasi = [
        ([str(c) for c in _list(columns)], float(unique), _integer(k))
        for columns, unique, k in data["quasi_identifier_sets"]
    ]
    return semantics, quasi


def _key(values: object) -> tuple[int | str, ...]:
    items = _list(values)
    if any(isinstance(item, bool) or not isinstance(item, int | str) for item in items):
        raise ValueError(f"invalid sample key {values!r}")
    return tuple(items)


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"invalid integer {value!r}")
    return value


def _optional_int(value: object) -> int | None:
    return None if value is None else _integer(value)


def _list(value: object) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"invalid list {value!r}")
    return value


def _pair(value: object) -> tuple[Any, Any]:
    items = _list(value)
    if len(items) != 2:
        raise ValueError(f"invalid map entry {value!r}")
    return items[0], items[1]


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"invalid string {value!r}")
    return value
