"""Parquet materialization, content hashing, and loading of catalog datasets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import DataFrame, SparkSession

from profiler_bench.catalog.datasets import DATASETS
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
        path: Directory holding one parquet file per table, the manifest, and the metadata.
        content_hash: SHA-256 over the parquet files and the manifest.
        table_hashes: SHA-256 of each table's parquet file.
        manifest: Ground-truth manifest.
    """

    path: Path
    content_hash: str
    table_hashes: dict[str, str]
    manifest: Manifest

    def table_path(self, table: str) -> Path:
        """Returns the parquet file of `table`."""
        return self.path / f"{table}.parquet"


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
    return (root or cache_dir()) / name / tier / f"seed={seed}" / get(name).fingerprint()


def materialize(name: str, tier: str = DEFAULT_TIER, seed: int = 0, root: Path | None = None) -> Materialized:
    """Writes a dataset instance to the cache unless it is already there, and returns it.

    Args:
        name: Dataset name.
        tier: Scale tier name.
        seed: Generator seed.
        root: Cache root; defaults to `cache_dir()`.
    """
    target = instance_dir(name, tier, seed, root)
    if not (target / META_FILE).exists():
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


def _write(path: Path, tables: dict[str, pa.Table], truth: Manifest) -> None:
    for table_name, table in tables.items():
        pq.write_table(table, path / f"{table_name}.parquet", compression="zstd")
    (path / MANIFEST_FILE).write_text(truth.model_dump_json(indent=2) + "\n", encoding="utf-8")
    table_hashes = {t: _sha256(path / f"{t}.parquet") for t in sorted(tables)}
    meta = {"content_hash": content_hash(table_hashes, path / MANIFEST_FILE), "tables": table_hashes}
    (path / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _read(path: Path) -> Materialized:
    meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
    truth = Manifest.model_validate_json((path / MANIFEST_FILE).read_text(encoding="utf-8"))
    return Materialized(path, meta["content_hash"], meta["tables"], truth)


def content_hash(table_hashes: dict[str, str], manifest_path: Path) -> str:
    """Returns the SHA-256 over the sorted table hashes and the manifest file."""
    digest = hashlib.sha256()
    for table_name in sorted(table_hashes):
        digest.update(f"{table_name}={table_hashes[table_name]}\n".encode())
    digest.update(manifest_path.read_bytes())
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, tier: str = DEFAULT_TIER, seed: int = 0, spark: SparkSession | None = None) -> dict[str, DataFrame]:
    """Returns the tables of a dataset instance as Spark DataFrames, materializing it first if needed.

    Args:
        name: Dataset name.
        tier: Scale tier name.
        seed: Generator seed.
        spark: Spark session; defaults to the active session.

    Raises:
        RuntimeError: If no session is given and none is active.
    """
    session = spark or SparkSession.getActiveSession()
    if session is None:
        raise RuntimeError("load needs a SparkSession")
    instance = materialize(name, tier, seed)
    return {t.name: session.read.parquet(str(instance.table_path(t.name))) for t in instance.manifest.tables}
