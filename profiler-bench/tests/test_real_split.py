"""Benchmark parts of real datasets: root-entity homes, foreign-key closure by copying parents, and shares."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from profiler_bench.catalog.manifest import ForeignKey
from profiler_bench.catalog.real import (
    PARTS,
    REAL_CACHE_ENV,
    SPLIT_REPORT,
    WAREHOUSE_ENV,
    RealDataset,
    RealTable,
    Unavailable,
    instance_dir,
    materialize_part,
    part_dir,
    real_cache_dir,
    split,
    split_dir,
    split_report,
)
from profiler_bench.catalog.store import write_instance


def _fk(child: str, column: str, parent: str, parent_column: str) -> ForeignKey:
    return ForeignKey(child_table=child, child_columns=(column,), parent_table=parent, parent_columns=(parent_column,))


STAR = RealDataset(
    name="star_fixture",
    description="A TPC-DS-shaped star: a customer root with two parents and a skewed fact table with four parents.",
    license="none",
    terms="none",
    tables=(
        RealTable("address", "x.y.address", 1_500, "aid bigint", ("aid",)),
        RealTable("day", "x.y.day", 400, "did bigint", ("did",)),
        RealTable("item", "x.y.item", 200, "iid bigint", ("iid",)),
        RealTable("customer", "x.y.customer", 3_000, "cid bigint, aid bigint, did bigint", ("cid",)),
        RealTable("sale", "x.y.sale", 30_000, "sid bigint, cid bigint, iid bigint, did bigint, aid bigint", ("sid",)),
        RealTable("payment", "x.y.payment", 900, "pid bigint, amount double", ("pid",)),
    ),
    foreign_keys=(
        _fk("customer", "aid", "address", "aid"),
        _fk("customer", "did", "day", "did"),
        _fk("sale", "cid", "customer", "cid"),
        _fk("sale", "iid", "item", "iid"),
        _fk("sale", "did", "day", "did"),
        _fk("sale", "aid", "address", "aid"),
    ),
    split_roots=("customer",),
)


def _tables() -> dict[str, pa.Table]:
    rng = np.random.default_rng(3)
    weights = 1 / (np.arange(3_000) + 30)
    fanout = weights / weights.sum()

    def nullable(values: np.ndarray, every: int) -> pa.Array:
        return pa.array(values, mask=np.arange(values.size) % every == 0)

    return {
        "address": pa.table({"aid": np.arange(1_500)}),
        "day": pa.table({"did": np.arange(400)}),
        "item": pa.table({"iid": np.arange(200)}),
        "customer": pa.table(
            {
                "cid": np.arange(3_000),
                "aid": nullable(rng.integers(0, 1_500, 3_000), 20),
                "did": rng.integers(0, 400, 3_000),
            }
        ),
        "sale": pa.table(
            {
                "sid": np.arange(30_000),
                "cid": nullable(rng.choice(3_000, 30_000, p=fanout), 25),
                "iid": rng.integers(0, 200, 30_000),
                "did": rng.integers(0, 400, 30_000),
                "aid": rng.integers(0, 1_500, 30_000),
            }
        ),
        "payment": pa.table({"pid": np.arange(900), "amount": rng.normal(size=900)}),
    }


def _rows(table: pa.Table) -> set[tuple[object, ...]]:
    return set(zip(*(table.column(c).to_pylist() for c in table.column_names), strict=True))


def test_split_is_deterministic_and_independent_of_row_order() -> None:
    tables = _tables()
    shuffled = {n: t.take(np.random.default_rng(1).permutation(t.num_rows)) for n, t in tables.items()}
    first, again, reordered = split(STAR, tables), split(STAR, tables), split(STAR, shuffled)
    assert list(first) == list(PARTS)
    for part in PARTS:
        for name in tables:
            assert first[part][name].equals(again[part][name])
            assert _rows(first[part][name]) == _rows(reordered[part][name])


def test_split_keeps_every_row_and_satisfies_foreign_keys() -> None:
    tables = _tables()
    parts = split(STAR, tables)
    for name, table in tables.items():
        assert set().union(*(_rows(parts[p][name]) for p in PARTS)) == _rows(table), name
    for part in PARTS:
        for fk in STAR.foreign_keys:
            child = parts[part][fk.child_table].column(fk.child_columns[0]).drop_null().to_pylist()
            parent = set(parts[part][fk.parent_table].column(fk.parent_columns[0]).to_pylist())
            assert set(child) <= parent, (part, fk)


def test_split_shares_reach_about_thirty_percent_and_report_copies() -> None:
    tables = _tables()
    report = split_report(STAR, tables)
    parts = split(STAR, tables)
    for name, by_part in report.items():
        homes = 0
        for part, entry in by_part.items():
            assert entry["rows"] == parts[part][name].num_rows
            assert entry["share"] >= 0.28, (name, part, entry)
            homes += entry["rows"] - entry["copied"]
        assert homes == tables[name].num_rows, name
    assert all(report[root][p]["copied"] == 0 for root in ("customer", "payment") for p in PARTS)
    assert all(report["sale"][p]["copied"] == 0 for p in PARTS)
    assert all(report["item"][p]["copied"] > 0 for p in PARTS)


def test_customer_homes_carry_their_sales() -> None:
    parts = split(STAR, _tables())
    for part in PARTS:
        customers = set(parts[part]["customer"].column("cid").to_pylist())
        others = set().union(*(set(parts[p]["customer"].column("cid").to_pylist()) for p in PARTS if p != part))
        assert customers.isdisjoint(others)
        sold = parts[part]["sale"].column("cid").drop_null().to_pylist()
        assert set(sold) <= customers


def _cache_full(root: Path) -> None:
    tables = _tables()
    target = instance_dir(STAR, "full", root)
    target.mkdir(parents=True)
    write_instance(target, tables, STAR.manifest("full", tables), 0)


def test_materialize_part_splits_a_cached_export_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(WAREHOUSE_ENV, raising=False)
    _cache_full(tmp_path)
    instances = {part: materialize_part(STAR, "s", part, tmp_path) for part in PARTS}
    expected = split(STAR, _tables())
    for part, instance in instances.items():
        assert instance.path == part_dir(STAR, "s", part, tmp_path)
        assert instance.manifest.ground_truth == "none"
        assert _rows(instance.read_table("sale")) == _rows(expected[part]["sale"])
    assert all((part_dir(STAR, "full", part, tmp_path) / "meta.json").exists() for part in PARTS)
    report = json.loads((split_dir(STAR, tmp_path) / SPLIT_REPORT).read_text())
    assert report == split_report(STAR, _tables())


def test_materialize_part_without_export_or_warehouse_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(WAREHOUSE_ENV, raising=False)
    with pytest.raises(Unavailable, match=WAREHOUSE_ENV):
        materialize_part(STAR, "s", "profiled", tmp_path)
    with pytest.raises(KeyError):
        materialize_part(STAR, "xs", "profiled", tmp_path)


def test_real_cache_defaults_outside_checkouts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert real_cache_dir(tmp_path) == tmp_path / "real"
    monkeypatch.setenv(REAL_CACHE_ENV, str(tmp_path / "shared"))
    assert real_cache_dir() == tmp_path / "shared"
    monkeypatch.delenv(REAL_CACHE_ENV)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert real_cache_dir() == tmp_path / "home" / ".cache" / "profiler-bench" / "real"
