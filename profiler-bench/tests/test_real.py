import socket
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from pyspark.sql import SparkSession

from profiler_bench.catalog import DATASETS, REAL_DATASETS, verify
from profiler_bench.catalog.__main__ import main
from profiler_bench.catalog.manifest import ForeignKey
from profiler_bench.catalog.real import RealDataset, RealTable, Unavailable, instance_dir, materialize, subset
from profiler_bench.catalog.real_datasets import WANDERBRICKS
from profiler_bench.catalog.store import write_instance


ENV = ("DATABRICKS_WAREHOUSE_ID", "DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_CONFIG_PROFILE")

FIXTURE = RealDataset(
    name="fixture",
    description="Three tables: parents, children with nullable parent keys, and keyless grandchildren.",
    license="none",
    terms="none",
    tables=(
        RealTable("parent", "x.y.parent", 300, "id bigint, label string", ("id",)),
        RealTable("child", "x.y.child", 1_000, "cid bigint, pid bigint", ("cid",)),
        RealTable("grandchild", "x.y.grandchild", 2_000, "cid bigint, value double"),
    ),
    foreign_keys=(
        ForeignKey(child_table="child", child_columns=("pid",), parent_table="parent", parent_columns=("id",)),
        ForeignKey(child_table="grandchild", child_columns=("cid",), parent_table="child", parent_columns=("cid",)),
    ),
)


def _tables(order_seed: int | None = None) -> dict[str, pa.Table]:
    rng = np.random.default_rng(7)
    pid = rng.integers(0, 300, 1_000)
    tables = {
        "parent": pa.table({"id": np.arange(300), "label": [f"p{i}" for i in range(300)]}),
        "child": pa.table({"cid": np.arange(1_000), "pid": pa.array(pid, mask=np.arange(1_000) % 10 == 0)}),
        "grandchild": pa.table({"cid": rng.integers(0, 1_000, 2_000), "value": rng.normal(size=2_000)}),
    }
    if order_seed is None:
        return tables
    shuffle = np.random.default_rng(order_seed)
    return {name: t.take(shuffle.permutation(t.num_rows)) for name, t in tables.items()}


def _rows(table: pa.Table) -> list[tuple[object, ...]]:
    return sorted(zip(*(c.to_pylist() for c in table.columns), strict=True), key=repr)


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(tmp_path / "missing.cfg"))


def test_real_manifests_validate() -> None:
    assert not set(REAL_DATASETS) & set(DATASETS)
    for dataset in REAL_DATASETS.values():
        truth = dataset.manifest()
        assert truth.ground_truth == "none"
        assert truth.tier == "full"
        assert sorted(dataset.hashes) == sorted(t.name for t in dataset.tables)
        assert dataset.license and dataset.terms
        for spec, table in zip(dataset.tables, truth.tables, strict=True):
            assert table.row_count == spec.rows
            assert spec.sql.startswith(f"SELECT * FROM {spec.source} ORDER BY ")
            assert all(c.distribution is None and not c.formats for c in table.columns)


def test_privacy_paths_must_name_columns() -> None:
    table = RealTable("t", "x.y.t", 1, "a struct<b:string>", privacy={"a.c": "free_text"})
    with pytest.raises(ValueError, match=r"a\.c"):
        table.manifest_columns()
    nested = RealTable("t", "x.y.t", 1, "a struct<b:string>", privacy={"a.b": "free_text"})
    assert nested.manifest_columns()[0].fields[0].privacy == "free_text"


def test_export_without_a_warehouse_is_unavailable(offline: None, tmp_path: Path) -> None:
    with pytest.raises(Unavailable, match="DATABRICKS_WAREHOUSE_ID"):
        materialize(FIXTURE, "full", tmp_path)
    assert not instance_dir(FIXTURE, "full", tmp_path).exists()


def test_export_without_credentials_is_unavailable(
    offline: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "w")
    with pytest.raises(Unavailable, match="DATABRICKS_HOST"):
        materialize(FIXTURE, "s", tmp_path)


def test_export_offline_is_unavailable(offline: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "w")
    monkeypatch.setenv("DATABRICKS_HOST", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("DATABRICKS_TOKEN", "t")
    with pytest.raises(Unavailable, match="cannot reach"):
        materialize(FIXTURE, "full", tmp_path)


def test_cli_skips_unavailable_datasets(offline: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["real", "nyctaxi_trips", "--cache-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.startswith("skipped nyctaxi_trips: set DATABRICKS_WAREHOUSE_ID")


def test_subset_is_deterministic_capped_and_closed() -> None:
    first = subset(FIXTURE, _tables(), 200)
    assert all(_rows(first[n]) == _rows(t) for n, t in subset(FIXTURE, _tables(order_seed=3), 200).items())
    assert {n: t.num_rows for n, t in first.items()} != {n: t.num_rows for n, t in _tables().items()}
    assert all(t.num_rows <= 200 for t in first.values())
    parents = set(first["parent"].column("id").to_pylist())
    pids = first["child"].column("pid").to_pylist()
    assert None in pids
    assert {p for p in pids if p is not None} <= parents
    assert set(first["grandchild"].column("cid").to_pylist()) <= set(first["child"].column("cid").to_pylist())
    assert first["grandchild"].num_rows > 0


def test_subset_keeps_small_tables_whole() -> None:
    tables = _tables()
    assert all(_rows(t) == _rows(tables[n]) for n, t in subset(FIXTURE, tables, 10_000).items())


def test_tier_s_derives_from_a_cached_export(offline: None, tmp_path: Path, spark: SparkSession) -> None:
    full = instance_dir(FIXTURE, "full", tmp_path)
    full.mkdir(parents=True)
    write_instance(full, _tables(), FIXTURE.manifest(), 0)
    instance = materialize(FIXTURE, "s", tmp_path)
    assert instance.manifest.ground_truth == "none"
    assert instance.manifest.tier == "s"
    frames = {t.name: spark.read.parquet(str(instance.table_path(t.name))) for t in instance.manifest.tables}
    assert verify(instance.manifest, frames) == []
    assert [t.row_count for t in instance.manifest.tables] == [300, 1_000, 2_000]
    assert materialize(FIXTURE, "s", tmp_path).content_hash == instance.content_hash


@pytest.mark.network
def test_export_one_table_end_to_end(tmp_path: Path) -> None:
    amenities = next(t for t in WANDERBRICKS.tables if t.name == "amenities")
    dataset = RealDataset(
        name="amenities_probe",
        description="One tiny wanderbricks table.",
        license=WANDERBRICKS.license,
        terms=WANDERBRICKS.terms,
        tables=(amenities,),
        hashes={"amenities": WANDERBRICKS.hashes["amenities"]},
    )
    try:
        instance = materialize(dataset, "full", tmp_path)
    except Unavailable as reason:
        pytest.skip(f"no warehouse: {reason}")
    assert instance.table_hashes == dataset.hashes
    assert instance.read_table("amenities").num_rows == amenities.rows
