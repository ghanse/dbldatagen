"""Dataset registry, scale tiers, and in-memory generation."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa

from profiler_bench.catalog import gen, manifest
from profiler_bench.catalog.gen import Rng, rng_factory
from profiler_bench.catalog.manifest import Column, ForeignKey, Manifest, Table


TIERS: dict[str, int] = {"xs": 5_000, "s": 100_000, "m": 2_000_000}
DEFAULT_TIER = "s"


class Build(NamedTuple):
    """Output of a dataset generator.

    Attributes:
        tables: Generated tables by name.
        truth: Planted structure of every table; `type` and `row_count` are filled in afterwards.
        foreign_keys: Foreign keys between the tables.
    """

    tables: dict[str, pa.Table]
    truth: list[Table]
    foreign_keys: list[ForeignKey]


@dataclass(frozen=True)
class Dataset:
    """Catalog entry.

    Attributes:
        name: Dataset name.
        description: What the dataset exercises.
        build: Generator taking the tier's base row count and a stream factory.
    """

    name: str
    description: str
    build: Callable[[int, Rng], Build]

    def fingerprint(self) -> str:
        """Returns a hash of the generator source and the shared catalog code it depends on.

        The fingerprint is part of the cache key, so editing a generator invalidates its cache.
        """
        sources = [inspect.getsourcefile(self.build), gen.__file__, manifest.__file__, __file__]
        digest = hashlib.sha256()
        for source in sources:
            digest.update(Path(str(source)).read_bytes())
        return digest.hexdigest()[:16]


def generate(dataset: Dataset, tier: str, seed: int) -> tuple[dict[str, pa.Table], Manifest]:
    """Generates a dataset in memory and returns its tables with the completed manifest.

    Args:
        dataset: Catalog entry.
        tier: Scale tier name.
        seed: Seed for every random stream.

    Raises:
        KeyError: If `tier` is unknown.
        ValueError: If the declared columns do not match the generated tables.
    """
    if tier not in TIERS:
        raise KeyError(f"unknown tier {tier!r}; expected one of {sorted(TIERS)}")
    build = dataset.build(TIERS[tier], rng_factory(seed, dataset.name))
    if sorted(build.tables) != sorted(t.name for t in build.truth):
        raise ValueError(f"{dataset.name}: declared tables do not match generated tables")
    truth = tuple(_complete(t, build.tables[t.name]) for t in build.truth)
    result = Manifest(
        dataset=dataset.name,
        tier=tier,
        seed=seed,
        description=dataset.description,
        tables=truth,
        foreign_keys=tuple(build.foreign_keys),
    )
    return build.tables, result


def _complete(table: Table, data: pa.Table) -> Table:
    if [c.name for c in table.columns] != data.column_names:
        raise ValueError(f"{table.name}: declared columns {[c.name for c in table.columns]} != {data.column_names}")
    columns = tuple(_typed(c, data.schema.field(c.name).type) for c in table.columns)
    return table.model_copy(update={"row_count": data.num_rows, "columns": columns})


def _typed(column: Column, arrow_type: pa.DataType) -> Column:
    update: dict[str, object] = {"type": spark_type(arrow_type)}
    if pa.types.is_struct(arrow_type):
        names = [arrow_type.field(i).name for i in range(arrow_type.num_fields)]
        if [f.name for f in column.fields] != names:
            raise ValueError(f"{column.name}: declared fields do not match {names}")
        update["fields"] = tuple(_typed(f, arrow_type.field(f.name).type) for f in column.fields)
    elif pa.types.is_map(arrow_type):
        if column.key is None or column.value is None:
            raise ValueError(f"{column.name}: map columns declare key and value")
        update["key"] = _typed(column.key, arrow_type.key_type)
        update["value"] = _typed(column.value, arrow_type.item_type)
    elif pa.types.is_list(arrow_type):
        if column.element is None:
            raise ValueError(f"{column.name}: array columns declare element")
        update["element"] = _typed(column.element, arrow_type.value_type)
    return column.model_copy(update=update)


_SIMPLE_TYPES = {
    pa.bool_(): "boolean",
    pa.int8(): "tinyint",
    pa.int16(): "smallint",
    pa.int32(): "int",
    pa.int64(): "bigint",
    pa.float32(): "float",
    pa.float64(): "double",
    pa.string(): "string",
    pa.binary(): "binary",
    pa.date32(): "date",
}


def spark_type(arrow_type: pa.DataType) -> str:
    """Returns the Spark SQL type string Spark reports when reading `arrow_type` from parquet.

    Raises:
        ValueError: If the type is not used by the catalog.
    """
    if arrow_type in _SIMPLE_TYPES:
        return _SIMPLE_TYPES[arrow_type]
    if pa.types.is_decimal(arrow_type):
        return f"decimal({arrow_type.precision},{arrow_type.scale})"
    if pa.types.is_timestamp(arrow_type):
        return "timestamp" if arrow_type.tz else "timestamp_ntz"
    if pa.types.is_struct(arrow_type):
        fields = ",".join(f"{f.name}:{spark_type(f.type)}" for f in arrow_type)
        return f"struct<{fields}>"
    if pa.types.is_map(arrow_type):
        return f"map<{spark_type(arrow_type.key_type)},{spark_type(arrow_type.item_type)}>"
    if pa.types.is_list(arrow_type):
        return f"array<{spark_type(arrow_type.value_type)}>"
    raise ValueError(f"unsupported Arrow type {arrow_type}")
