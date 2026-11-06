"""Canonical logical hash of Arrow tables, independent of writer, compression, file layout, and row order.

The hash covers the Spark schema (column names and type strings), the row count, and every value.
Rows are put in a canonical order by sorting on all non-nested columns in schema order, so tables
whose rows are identical on those columns must not differ in their nested columns. Each column is
encoded as its validity bitmap plus its values with null slots zeroed; struct fields inherit the
struct's nulls, lists and maps are encoded as lengths plus their flattened values, and every NaN is
encoded with the same bits.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal
from typing import Protocol

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from profiler_bench.catalog.registry import spark_type


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...


def table_hash(table: pa.Table) -> str:
    """Returns the SHA-256 logical hash of `table`."""
    columns = [table.column(i).combine_chunks() for i in range(table.num_columns)]
    keys = [(f.name, "ascending") for f, c in zip(table.schema, columns, strict=True) if not _nested(f.type)]
    order = pc.sort_indices(table, sort_keys=keys, null_placement="at_end") if keys else None
    digest = hashlib.sha256()
    digest.update(f"rows={table.num_rows}\n".encode())
    for field, column in zip(table.schema, columns, strict=True):
        digest.update(f"{field.name}:{spark_type(field.type)}\n".encode())
        _feed(digest, column if order is None else column.take(order))
    return digest.hexdigest()


def _nested(arrow_type: pa.DataType) -> bool:
    return bool(pa.types.is_struct(arrow_type) or pa.types.is_list(arrow_type) or pa.types.is_map(arrow_type))


def _feed(digest: _Digest, array: pa.Array) -> None:
    arrow_type = array.type
    valid = array.is_valid().to_numpy(zero_copy_only=False)
    digest.update(len(array).to_bytes(8, "little") + np.packbits(valid).tobytes())
    if pa.types.is_struct(arrow_type):
        for child in array.flatten():
            _feed(digest, child)
    elif pa.types.is_list(arrow_type) or pa.types.is_map(arrow_type):
        offsets = array.offsets.to_numpy(zero_copy_only=False)
        spans = np.diff(offsets)
        digest.update(np.ascontiguousarray(np.where(valid, spans, 0), dtype=np.int64).tobytes())
        if pa.types.is_map(arrow_type):
            children = pa.StructArray.from_arrays([array.keys, array.items], names=["key", "value"])
        else:
            children = array.values
        window = children.slice(int(offsets[0]), int(offsets[-1] - offsets[0]))
        _feed(digest, window.filter(pa.array(np.repeat(valid, spans))))
    elif pa.types.is_string(arrow_type) or pa.types.is_binary(arrow_type):
        filled = pc.fill_null(array, pa.scalar("" if pa.types.is_string(arrow_type) else b"", arrow_type))
        offsets = np.frombuffer(filled.buffers()[1], dtype=np.int32)[filled.offset : filled.offset + len(filled) + 1]
        data = filled.buffers()[2]
        digest.update(np.ascontiguousarray(offsets - offsets[0], dtype=np.int64).tobytes())
        if data is not None:
            digest.update(data.to_pybytes()[offsets[0] : offsets[-1]])
    elif pa.types.is_decimal(arrow_type):
        filled = pc.fill_null(array, pa.scalar(Decimal(0), arrow_type))
        width = arrow_type.byte_width
        digest.update(filled.buffers()[1].to_pybytes()[filled.offset * width : (filled.offset + len(filled)) * width])
    elif pa.types.is_boolean(arrow_type):
        digest.update(pc.fill_null(array, False).to_numpy(zero_copy_only=False).astype(np.uint8).tobytes())
    else:
        values = _numeric(array)
        digest.update(np.ascontiguousarray(values).tobytes())


def _numeric(array: pa.Array) -> np.ndarray:
    arrow_type = array.type
    if pa.types.is_timestamp(arrow_type):
        array = array.cast(pa.timestamp("us", tz=arrow_type.tz)).view(pa.int64())
    elif pa.types.is_date64(arrow_type):
        array = array.view(pa.int64())
    elif pa.types.is_date32(arrow_type):
        array = array.view(pa.int32())
    filled = pc.fill_null(array, pa.scalar(0, array.type))
    values = np.array(filled.to_numpy(zero_copy_only=False))
    if np.issubdtype(values.dtype, np.floating):
        values[np.isnan(values)] = np.nan
    return values
