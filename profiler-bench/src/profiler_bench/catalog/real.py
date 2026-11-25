"""Real datasets exported from a Databricks SQL warehouse into the catalog cache.

A `RealDataset` declares its source tables, the SQL that exports each, the expected row counts, the schema,
keys, foreign keys, and privacy classes, and the logical hash of every exported table, pinned after the first
export. Real data has no planted ground truth, so its manifest is marked `ground_truth: none` and claims only
that structure.

Tiers: `full` is the export. `s` is a deterministic subset of it: a table with more than `REAL_SUBSET_CAP`
rows (~50,000) keeps the rows whose primary key (all non-nested columns when it has none) hashes below
`REAL_SUBSET_CAP / rows`; then, parents first, a row is dropped when a parent it references through a declared
foreign key was dropped. Full-export pinned hashes are unchanged.

Exports use the Statement Execution API with `EXTERNAL_LINKS` and `ARROW_STREAM`. Each chunk's link is
requested just before the chunk is downloaded; an expired link is requested again, up to `LINK_ATTEMPTS` times with
backoff. Each table's source `SELECT COUNT(*)` must match the rows received and the declared count. See `connect`
for the environment variables. Any network or authentication failure raises `Unavailable`; a cached instance
needs no warehouse.
"""

from __future__ import annotations

import configparser
import hashlib
import http.client
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import reduce
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import ipc

from profiler_bench.catalog.hashing import table_hash
from profiler_bench.catalog.manifest import Column, ForeignKey, Manifest, Privacy, Table
from profiler_bench.catalog.registry import TIER_FILES, spark_type
from profiler_bench.catalog.store import META_FILE, Materialized, cache_dir, is_intact, read_instance, write_instance


REAL_TIERS = ("s", "full")
REAL_SUBSET_CAP = 50_000
FULL_FILE_ROWS = 1_000_000
POLL_SECONDS = 2.0
RETRY_SECONDS = 2.0
LINK_ATTEMPTS = 3
WAREHOUSE_ENV = "PROFILER_BENCH_DATABRICKS_WAREHOUSE_ID"
PROFILE_ENV = "PROFILER_BENCH_DATABRICKS_PROFILE"
TIMEOUT = 120.0
_NULL = "\0"
_SEPARATOR = "\x1f"


class Unavailable(Exception):
    """The warehouse cannot be used: no credentials or warehouse configured, rejected credentials, or offline."""


@dataclass(frozen=True)
class RealTable:
    """One exported table.

    Attributes:
        name: Table name in the catalog.
        source: Fully qualified source table.
        rows: Expected row count of the export.
        columns: Schema as `name type` pairs separated by commas, with Spark SQL type strings.
        primary_key: Primary key columns, when the documented key holds in the data.
        unique_keys: Other unique column sets.
        privacy: Privacy class by column path; nested paths join names with dots (`messages.element.message`).
        description: What the table holds.
    """

    name: str
    source: str
    rows: int
    columns: str
    primary_key: tuple[str, ...] = ()
    unique_keys: tuple[tuple[str, ...], ...] = ()
    privacy: Mapping[str, Privacy] = field(default_factory=dict)
    description: str = ""

    @property
    def sql(self) -> str:
        """Returns the export statement; rows are ordered by the primary key, or by every column without one."""
        return f"SELECT * FROM {self.source} ORDER BY {', '.join(self.primary_key) or 'ALL'}"

    def manifest_columns(self) -> tuple[Column, ...]:
        """Returns the declared columns with their nested children and privacy classes.

        Raises:
            ValueError: If a privacy path names no column.
        """
        columns = tuple(_column(n, t, self.privacy, n) for n, t in (p.split(" ", 1) for p in _split(self.columns)))
        unknown = sorted(set(self.privacy) - {path for c in columns for path in _paths(c, "")})
        if unknown:
            raise ValueError(f"{self.name}: privacy paths {unknown} name no column")
        return columns


@dataclass(frozen=True)
class RealDataset:
    """A real dataset exported from a warehouse.

    Attributes:
        name: Dataset name.
        description: What the dataset contains and why it is in the catalog.
        license: Licence the data is distributed under.
        terms: Terms of use and required attribution.
        tables: Exported tables.
        foreign_keys: Documented foreign keys that hold for every non-null child key.
        hashes: Pinned logical hash of each table of the full export (see `profiler_bench.catalog.hashing`).
    """

    name: str
    description: str
    license: str
    terms: str
    tables: tuple[RealTable, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()
    hashes: Mapping[str, str] = field(default_factory=dict)

    def manifest(self, tier: str = "full", tables: Mapping[str, pa.Table] | None = None) -> Manifest:
        """Returns the manifest of a tier; row counts and file counts come from `tables` when given."""
        specs = []
        for spec in self.tables:
            rows = spec.rows if tables is None else tables[spec.name].num_rows
            files = TIER_FILES["s"] if tier == "s" else math.ceil(rows / FULL_FILE_ROWS)
            specs.append(
                Table(
                    name=spec.name,
                    row_count=rows,
                    description=f"Exported from {spec.source}. {spec.description}".strip(),
                    primary_key=spec.primary_key,
                    unique_keys=spec.unique_keys,
                    columns=spec.manifest_columns(),
                    files=max(1, min(files, rows)),
                )
            )
        return Manifest(
            ground_truth="none",
            dataset=self.name,
            tier=tier,
            structure_seed=0,
            description=f"{self.description} Licence: {self.license}.",
            tables=tuple(specs),
            foreign_keys=self.foreign_keys,
        )


def _split(text: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        depth += (char in "<(") - (char in ">)")
        if char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    return [*parts, text[start:].strip()]


def _column(name: str, kind: str, privacy: Mapping[str, Privacy], path: str) -> Column:
    children: dict[str, Any] = {}
    if kind.startswith("struct<"):
        fields = (f.split(":", 1) for f in _split(kind[len("struct<") : -1]))
        children["fields"] = tuple(_column(n, t, privacy, f"{path}.{n}") for n, t in fields)
    elif kind.startswith("array<"):
        children["element"] = _column("element", kind[len("array<") : -1], privacy, f"{path}.element")
    elif kind.startswith("map<"):
        key, value = _split(kind[len("map<") : -1])
        children["key"] = _column("key", key, privacy, f"{path}.key")
        children["value"] = _column("value", value, privacy, f"{path}.value")
    return Column(name=name, type=kind, privacy=privacy.get(path), **children)


def _paths(column: Column, prefix: str) -> Iterator[str]:
    yield f"{prefix}{column.name}"
    for child in column.children():
        yield from _paths(child, f"{prefix}{column.name}.")


def instance_dir(dataset: RealDataset, tier: str, root: Path | None = None) -> Path:
    """Returns the cache directory of one tier of a real dataset."""
    return (root or cache_dir()) / "real" / dataset.name / tier


def materialize(
    dataset: RealDataset, tier: str = "s", root: Path | None = None, check_hash: bool = True
) -> Materialized:
    """Returns a tier of a real dataset from the cache, exporting the full tier first if needed.

    Args:
        dataset: Real dataset.
        tier: `s` (the subset) or `full` (the export).
        root: Cache root; defaults to `cache_dir()`.
        check_hash: Whether to recompute the logical hashes of a cached instance before using it.

    Raises:
        KeyError: If `tier` is not a real-dataset tier.
        Unavailable: If the full tier is not cached and the warehouse cannot be used.
        ValueError: If the export differs from the declared row counts, schema, or pinned hashes.
    """
    if tier not in REAL_TIERS:
        raise KeyError(f"unknown tier {tier!r}; expected one of {list(REAL_TIERS)}")
    target = instance_dir(dataset, tier, root)
    if (target / META_FILE).exists():
        cached = is_intact(target, check_hash) and (tier != "full" or _pinned(dataset, read_instance(target)))
        if cached:
            return read_instance(target)
        shutil.rmtree(target)
    if tier == "full":
        tables = export(dataset, connect())
    else:
        full = materialize(dataset, "full", root, check_hash)
        tables = subset(dataset, {t.name: full.read_table(t.name) for t in full.manifest.tables}, REAL_SUBSET_CAP)
    truth = dataset.manifest(tier, tables)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=".tmp-"))
    try:
        write_instance(staging, tables, truth, 0)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return read_instance(target)


def _pinned(dataset: RealDataset, instance: Materialized) -> bool:
    return all(instance.table_hashes.get(t) == h for t, h in dataset.hashes.items())


def subset(dataset: RealDataset, tables: Mapping[str, pa.Table], cap: int) -> dict[str, pa.Table]:
    """Returns the deterministic tier-s subset of `tables`, closed under the declared foreign keys.

    Args:
        dataset: Real dataset the tables belong to.
        tables: Full tables by name.
        cap: Most rows a table keeps before foreign-key closure.
    """
    specs = {t.name: t for t in dataset.tables}
    parents = {name: {fk.parent_table for fk in dataset.foreign_keys if fk.child_table == name} for name in specs}
    kept: dict[str, pa.Table] = {}
    for name in TopologicalSorter(parents).static_order():
        table = tables[name]
        if table.num_rows > cap:
            columns = specs[name].primary_key or [f.name for f in table.schema if not pa.types.is_nested(f.type)]
            table = table.filter(_hash_below(table, columns, name, cap / table.num_rows))
        for fk in (fk for fk in dataset.foreign_keys if fk.child_table == name):
            table = table.filter(_references(table, fk, kept[fk.parent_table]))
        kept[name] = table
    return {name: kept[name] for name in specs}


def _key_text(table: pa.Table, columns: tuple[str, ...] | list[str]) -> pa.ChunkedArray:
    text = [pc.fill_null(pc.cast(table.column(c), pa.string()), _NULL) for c in columns]
    return text[0] if len(text) == 1 else pc.binary_join_element_wise(*text, _SEPARATOR)


def _hash_below(table: pa.Table, columns: tuple[str, ...] | list[str], salt: str, fraction: float) -> pa.Array:
    threshold = int(fraction * 2**64)
    digests = (
        hashlib.blake2b(f"{salt}{_SEPARATOR}{v}".encode(), digest_size=8).digest()
        for v in _key_text(table, columns).to_pylist()
    )
    return pa.array([int.from_bytes(d, "little") < threshold for d in digests], pa.bool_())


def _references(child: pa.Table, fk: ForeignKey, parent: pa.Table) -> pa.ChunkedArray:
    present = reduce(pc.and_, [pc.is_valid(child.column(c)) for c in fk.child_columns])
    found = pc.is_in(
        _key_text(child, fk.child_columns), value_set=_key_text(parent, fk.parent_columns).combine_chunks()
    )
    return pc.or_(pc.invert(present), found)


@dataclass(frozen=True)
class Warehouse:
    """A SQL warehouse and the credentials to call it.

    Attributes:
        host: Workspace URL.
        token: Bearer token.
        warehouse_id: SQL warehouse ID.
    """

    host: str
    token: str = field(repr=False)
    warehouse_id: str


def connect() -> Warehouse:
    """Returns the configured warehouse.

    The warehouse is `PROFILER_BENCH_DATABRICKS_WAREHOUSE_ID`. The host and token are `DATABRICKS_HOST` and
    `DATABRICKS_TOKEN`, or else come from the profile `PROFILER_BENCH_DATABRICKS_PROFILE` (default: the CLI's
    `DATABRICKS_CONFIG_PROFILE`, then `DEFAULT`) through `~/.databrickscfg` and `databricks auth token`.

    Raises:
        Unavailable: If no warehouse, host, or token can be found.
    """
    warehouse_id = os.environ.get(WAREHOUSE_ENV)
    if not warehouse_id:
        raise Unavailable(f"set {WAREHOUSE_ENV} to the SQL warehouse to export from")
    host, token = os.environ.get("DATABRICKS_HOST"), os.environ.get("DATABRICKS_TOKEN")
    if not (host and token):
        profile = os.environ.get(PROFILE_ENV) or os.environ.get("DATABRICKS_CONFIG_PROFILE") or "DEFAULT"
        host = host or _profile_host(profile)
        token = _cli_token(profile)
    host = host.rstrip("/")
    return Warehouse(host if "://" in host else f"https://{host}", token, warehouse_id)


def _profile_host(profile: str) -> str:
    config = configparser.ConfigParser()
    config.read(os.environ.get("DATABRICKS_CONFIG_FILE", Path.home() / ".databrickscfg"))
    if not config.has_option(profile, "host"):
        raise Unavailable(f"set DATABRICKS_HOST and DATABRICKS_TOKEN, or {PROFILE_ENV} to a profile with a host")
    return config.get(profile, "host")


def _cli_token(profile: str) -> str:
    command = ["databricks", "auth", "token", "-p", profile]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Unavailable(f"set DATABRICKS_TOKEN, or install the databricks CLI ({error})") from error
    if result.returncode != 0:
        reason = (result.stderr.strip().splitlines() or ["no output"])[-1]
        raise Unavailable(f"`databricks auth token` failed for the configured profile: {reason}")
    return str(json.loads(result.stdout)["access_token"])


def _call(warehouse: Warehouse, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        warehouse.host + path,
        data=None if body is None else json.dumps(body).encode(),
        method=method,
        headers={"Authorization": f"Bearer {warehouse.token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            result: dict[str, Any] = json.load(response)
            return result
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise Unavailable(f"{warehouse.host} rejected the credentials (HTTP {error.code})") from error
        if error.code == 429 or error.code >= 500:
            raise Unavailable(f"{warehouse.host} is unavailable (HTTP {error.code})") from error
        raise
    except (OSError, http.client.HTTPException) as error:
        raise Unavailable(f"cannot reach {warehouse.host}: {error}") from error


def export(dataset: RealDataset, warehouse: Warehouse) -> dict[str, pa.Table]:
    """Exports every table of `dataset` and checks it against the source, the declaration, and the pinned hashes.

    Each table's `SELECT COUNT(*)` on the source, the rows received, and the declared row count must agree.

    Raises:
        Unavailable: If the warehouse or a result link cannot be used.
        ValueError: If a table differs from its source count, its declaration, or its pinned hash.
    """
    tables = {}
    for spec in dataset.manifest().tables:
        source = next(t for t in dataset.tables if t.name == spec.name)
        counted = _count(warehouse, source)
        table = tables[spec.name] = _export_table(warehouse, source)
        label = f"{dataset.name}.{spec.name}"
        if not counted == table.num_rows == spec.row_count:
            raise ValueError(
                f"{label}: source COUNT(*) {counted}, received {table.num_rows}, declared {spec.row_count}"
            )
        actual = [(f.name, spark_type(f.type)) for f in table.schema]
        expected = [(c.name, c.type) for c in spec.columns]
        if actual != expected:
            raise ValueError(f"{label}: schema {actual} != declared {expected}")
        pinned, digest = dataset.hashes.get(spec.name), table_hash(table)
        if pinned is not None and digest != pinned:
            raise ValueError(f"{label}: hash {digest} != pinned {pinned}")
    return tables


def _run(warehouse: Warehouse, statement: str, disposition: str, result_format: str) -> dict[str, Any]:
    body = {
        "warehouse_id": warehouse.warehouse_id,
        "statement": statement,
        "wait_timeout": "30s",
        "on_wait_timeout": "CONTINUE",
        "disposition": disposition,
        "format": result_format,
    }
    response = _call(warehouse, "POST", "/api/2.0/sql/statements", body)
    while response["status"]["state"] in ("PENDING", "RUNNING"):
        time.sleep(POLL_SECONDS)
        response = _call(warehouse, "GET", f"/api/2.0/sql/statements/{response['statement_id']}")
    status = response["status"]
    if status["state"] != "SUCCEEDED":
        raise RuntimeError(f"{statement}: {status['state']}: {status.get('error', {}).get('message')}")
    return response


def _count(warehouse: Warehouse, spec: RealTable) -> int:
    response = _run(warehouse, f"SELECT COUNT(*) FROM {spec.source}", "INLINE", "JSON_ARRAY")
    return int(response["result"]["data_array"][0][0])


def _export_table(warehouse: Warehouse, spec: RealTable) -> pa.Table:
    response = _run(warehouse, spec.sql, "EXTERNAL_LINKS", "ARROW_STREAM")
    chunks = response["manifest"].get("total_chunk_count", 0)
    if not chunks:
        raise ValueError(f"{spec.source}: the export returned no rows")
    return pa.concat_tables([_chunk(warehouse, response["statement_id"], i) for i in range(chunks)])


def _chunk(warehouse: Warehouse, statement: str, index: int) -> pa.Table:
    """Returns one result chunk, fetching a fresh link before each of `LINK_ATTEMPTS` attempts while links expire."""
    for attempt in range(LINK_ATTEMPTS):
        if attempt:
            time.sleep(RETRY_SECONDS * 2 ** (attempt - 1))
        path = f"/api/2.0/sql/statements/{statement}/result/chunks/{index}"
        links = [link["external_link"] for link in _call(warehouse, "GET", path)["external_links"]]
        try:
            return pa.concat_tables([_download(url) for url in links])
        except urllib.error.HTTPError as error:
            if error.code != 403:
                raise Unavailable(f"chunk {index} download failed (HTTP {error.code})") from error
        except (OSError, http.client.HTTPException, pa.ArrowInvalid) as error:
            raise Unavailable(f"chunk {index} download failed: {error}") from error
    raise Unavailable(f"chunk {index} link expired on {LINK_ATTEMPTS} attempts")


def _download(url: str) -> pa.Table:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return ipc.open_stream(response.read()).read_all()


def table_sizes(instance: Materialized) -> dict[str, int]:
    """Returns the parquet bytes of each table of an instance."""
    return {
        t.name: sum(p.stat().st_size for p in instance.table_path(t.name).glob("*.parquet"))
        for t in instance.manifest.tables
    }
