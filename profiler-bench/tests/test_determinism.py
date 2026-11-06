import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from pyspark.sql import SparkSession

from profiler_bench.catalog import DATASETS, generate, materialize, registry
from profiler_bench.catalog.gen import rng_factory
from profiler_bench.catalog.hashing import table_hash


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_same_seed_gives_identical_content_hash(name: str, tmp_path: Path) -> None:
    first = materialize(name, "xs", 7, tmp_path / "a")
    second = materialize(name, "xs", 7, tmp_path / "b")
    assert first.content_hash == second.content_hash
    assert first.table_hashes == second.table_hashes
    assert first.manifest == second.manifest


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_different_seeds_give_different_data(name: str) -> None:
    first, _ = generate(DATASETS[name], "xs", 0)
    second, _ = generate(DATASETS[name], "xs", 1)
    assert any(table_hash(first[t]) != table_hash(second[t]) for t in first if first[t].num_rows)


_HASHES = """
import json, sys
from profiler_bench.catalog import DATASETS, materialize
print(json.dumps({n: materialize(n, sys.argv[1], 0).table_hashes for n in DATASETS}))
"""


def _hashes_in_subprocess(tier: str, cache: Path, hash_seed: str) -> dict[str, dict[str, str]]:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PROFILER_BENCH_CACHE": str(cache)}
    out = subprocess.run([sys.executable, "-c", _HASHES, tier], env=env, check=True, capture_output=True, text=True)
    result: dict[str, dict[str, str]] = json.loads(out.stdout)
    return result


def test_logical_hash_is_independent_of_process_and_spark_layout(spark: SparkSession, tmp_path: Path) -> None:
    first = _hashes_in_subprocess("xs", tmp_path / "a", "0")
    second = _hashes_in_subprocess("xs", tmp_path / "b", "4242")
    assert first == second
    settings = {
        "spark.sql.parquet.outputTimestampType": "TIMESTAMP_MICROS",
        "spark.sql.parquet.datetimeRebaseModeInWrite": "CORRECTED",
    }
    previous = {key: spark.conf.get(key, None) for key in settings}
    for key, value in settings.items():
        spark.conf.set(key, value)
    layouts = itertools.cycle([(7, "gzip"), (1, "snappy"), (3, "uncompressed")])
    try:
        for name in DATASETS:
            instance = materialize(name, "xs", 0, tmp_path / "a")
            for table, (partitions, codec) in zip(instance.manifest.tables, layouts, strict=False):
                out = tmp_path / "spark" / name / table.name
                source = spark.read.parquet(str(instance.table_path(table.name)))
                source.repartition(partitions).write.option("compression", codec).parquet(str(out))
                assert table_hash(pq.read_table(out, partitioning=None)) == first[name][table.name], (name, table.name)
    finally:
        for key, old in previous.items():
            if old is None:
                spark.conf.unset(key)
            else:
                spark.conf.set(key, old)


def test_tiers_s_and_m_write_multiple_files(tmp_path: Path) -> None:
    instance = materialize("numeric", "s", 0, tmp_path)
    parts = sorted(instance.table_path("measurements").glob("*.parquet"))
    assert len(parts) == instance.manifest.table("measurements").files == registry.TIER_FILES["s"] > 1
    assert table_hash(instance.read_table("measurements")) == instance.table_hashes["measurements"]


def test_cached_instance_is_regenerated_when_its_files_change(tmp_path: Path) -> None:
    first = materialize("numeric", "xs", 0, tmp_path)
    (part,) = first.table_path("measurements").glob("*.parquet")
    table = pq.read_table(part, partitioning=None)
    pq.write_table(table.slice(1), part)
    assert materialize("numeric", "xs", 0, tmp_path, check_hash=False).table_hashes == first.table_hashes
    assert pq.read_table(part, partitioning=None).num_rows == table.num_rows - 1
    second = materialize("numeric", "xs", 0, tmp_path)
    assert second.content_hash == first.content_hash
    assert second.read_table("measurements").num_rows == table.num_rows


def test_materialize_reuses_the_cached_instance(tmp_path: Path) -> None:
    first = materialize("numeric", "xs", 0, tmp_path)
    (part,) = first.table_path("measurements").glob("*.parquet")
    mtime = part.stat().st_mtime_ns
    second = materialize("numeric", "xs", 0, tmp_path)
    assert second.path == first.path
    assert part.stat().st_mtime_ns == mtime


def test_cache_path_includes_the_generator_fingerprint(tmp_path: Path) -> None:
    instance = materialize("temporal", "xs", 0, tmp_path)
    assert instance.path == tmp_path / "temporal" / "xs" / "seed-0" / DATASETS["temporal"].fingerprint()


def test_fingerprint_includes_library_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    before = DATASETS["numeric"].fingerprint()
    monkeypatch.setattr(registry, "version", lambda package: "0.0.0" if package == "pyarrow" else "1")
    assert DATASETS["numeric"].fingerprint() != before
    assert set(registry.FINGERPRINT_PACKAGES) == {"numpy", "pandas", "pyarrow", "pyspark"}


def test_streams_are_independent_of_request_order() -> None:
    rng = rng_factory(0, "d")
    a_first = rng("a").random(3)
    rng("b").random(3)
    assert (rng("a").random(3) == a_first).all()
    assert not (rng("b").random(3) == a_first).all()
