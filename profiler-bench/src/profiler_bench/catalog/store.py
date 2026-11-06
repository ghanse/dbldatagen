"""Parquet materialization, logical content hashing, and loading of catalog datasets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import DataFrame, SparkSession

from profiler_bench.catalog.datasets import DATASETS
from profiler_bench.catalog.hashing import table_hash
from profiler_bench.catalog.manifest import Manifest
from profiler_bench.catalog.registry import DEFAULT_TIER, Dataset, generate


CACHE_ENV = "PROFILER_BENCH_CACHE"
DEFAULT_CACHE = Path(__file__).resolve().parents[3] / ".cache" / "catalog"
MANIFEST_FILE = "manifest.json"
META_FILE = "meta.json"


@dataclass(frozen=True)
class Materialized:
    """A dataset instance written to the cache.

    Attributes:
        path: Directory holding one parquet directory per table, the manifest, and the metadata.
        content_hash: SHA-256 over the logical table hashes and the manifest.
        table_hashes: Logical hash of each table (see `profiler_bench.catalog.hashing`).
        manifest: Ground-truth manifest.
    """

    path: Path
    content_hash: str
    table_hashes: dict[str, str]
    manifest: Manifest

    def table_path(self, table: str) -> Path:
        """Returns the parquet directory of `table`; it holds `Table.files` part files."""
        return self.path / f"{table}.parquet"

    def read_table(self, table: str) -> pa.Table:
        """Returns `table` as one Arrow table, its part files concatenated in storage order."""
        return pq.read_table(self.table_path(table), partitioning=None)


def cache_dir() -> Path:
    """Returns the cache root: `$PROFILER_BENCH_CACHE`, or `profiler-bench/.cache/catalog`."""
    return Path(os.environ.get(CACHE_ENV, DEFAULT_CACHE))


def get(name: str) -> Dataset:
    """Returns the catalog entry called `name`.

    Raises:
        KeyError: If no dataset has that name.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; expected one of {sorted(DATASETS)}")
    return DATASETS[name]


def instance_dir(name: str, tier: str, seed: int, root: Path | None = None) -> Path:
    """Returns the cache directory of one dataset instance; the generator fingerprint is part of the path."""
    return (root or cache_dir()) / name / tier / f"seed-{seed}" / get(name).fingerprint()


def materialize(
    name: str, tier: str = DEFAULT_TIER, seed: int = 0, root: Path | None = None, check_hash: bool = True
) -> Materialized:
    """Writes a dataset instance to the cache unless it is already there, and returns it.

    A cached instance is regenerated when its manifest or metadata cannot be read, when its content hash
    no longer matches the recorded table hashes and manifest, or (with `check_hash`) when its parquet
    files no longer match the recorded table hashes.

    Args:
        name: Dataset name.
        tier: Scale tier name.
        seed: Generator seed.
        root: Cache root; defaults to `cache_dir()`.
        check_hash: Whether to recompute the logical hashes of a cached instance before using it.
            Turning this off saves a full read of the data, which matters at the `m` tier.
    """
    target = instance_dir(name, tier, seed, root)
    if (target / META_FILE).exists():
        if _is_intact(target, check_hash):
            return _read(target)
        shutil.rmtree(target)
    tables, truth = generate(get(name), tier, seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=".tmp-"))
    try:
        _write(staging, tables, truth)
        staging.rename(target)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        if not (target / META_FILE).exists():
            raise
    return _read(target)


def _is_intact(path: Path, check_files: bool) -> bool:
    try:
        cached = _read(path)
    except (OSError, ValueError, KeyError):
        return False
    if content_hash(cached.table_hashes, path / MANIFEST_FILE) != cached.content_hash:
        return False
    return not check_files or file_hashes(cached) == cached.table_hashes


def file_hashes(instance: Materialized) -> dict[str, str]:
    """Returns the logical hash of each table as read back from its parquet files."""
    return {t.name: table_hash(instance.read_table(t.name)) for t in instance.manifest.tables}


def _write(path: Path, tables: dict[str, pa.Table], truth: Manifest) -> None:
    for spec in truth.tables:
        table, directory = tables[spec.name], path / f"{spec.name}.parquet"
        directory.mkdir()
        bounds = np.linspace(0, table.num_rows, spec.files + 1).astype(int)
        for part, (start, stop) in enumerate(pairwise(bounds)):
            pq.write_table(table.slice(start, stop - start), directory / f"part-{part:05d}.parquet", compression="zstd")
    (path / MANIFEST_FILE).write_text(truth.model_dump_json(indent=2) + "\n", encoding="utf-8")
    hashes = {name: table_hash(tables[name]) for name in sorted(tables)}
    meta = {"content_hash": content_hash(hashes, path / MANIFEST_FILE), "tables": hashes}
    (path / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    written = file_hashes(_read(path))
    if written != hashes:
        raise RuntimeError(f"logical hashes changed on write: {sorted(t for t in hashes if hashes[t] != written[t])}")


def _read(path: Path) -> Materialized:
    meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
    truth = Manifest.model_validate_json((path / MANIFEST_FILE).read_text(encoding="utf-8"))
    return Materialized(path, meta["content_hash"], meta["tables"], truth)


def content_hash(table_hashes: dict[str, str], manifest_path: Path) -> str:
    """Returns the SHA-256 over the sorted logical table hashes and the manifest file."""
    digest = hashlib.sha256()
    for table_name in sorted(table_hashes):
        digest.update(f"{table_name}={table_hashes[table_name]}\n".encode())
    digest.update(manifest_path.read_bytes())
    return digest.hexdigest()


def load(
    name: str,
    tier: str = DEFAULT_TIER,
    seed: int = 0,
    spark: SparkSession | None = None,
    check_hash: bool = True,
) -> dict[str, DataFrame]:
    """Returns the tables of a dataset instance as Spark DataFrames, materializing it first if needed.

    Args:
        name: Dataset name.
        tier: Scale tier name.
        seed: Generator seed.
        spark: Spark session; defaults to the active session.
        check_hash: Whether to verify the cached files against their logical hashes, regenerating on
            mismatch; pass False to skip the full read at the `m` tier.

    Raises:
        RuntimeError: If no session is given and none is active.
    """
    session = spark or SparkSession.getActiveSession()
    if session is None:
        raise RuntimeError("load needs a SparkSession")
    instance = materialize(name, tier, seed, check_hash=check_hash)
    return {t.name: session.read.parquet(str(instance.table_path(t.name))) for t in instance.manifest.tables}
