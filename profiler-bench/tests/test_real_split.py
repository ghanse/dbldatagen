"""Benchmark parts of real datasets: deterministic, disjoint, and closed under the declared foreign keys."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from profiler_bench.catalog.manifest import ForeignKey
from profiler_bench.catalog.real import (
    PARTS,
    WAREHOUSE_ENV,
    RealDataset,
    RealTable,
    Unavailable,
    instance_dir,
    materialize_part,
    part_dir,
    split,
)
from profiler_bench.catalog.store import write_instance


def _fk(child: str, column: str, parent: str, parent_column: str) -> ForeignKey:
    return ForeignKey(child_table=child, child_columns=(column,), parent_table=parent, parent_columns=(parent_column,))


SPLIT_FIXTURE = RealDataset(
    name="split_fixture",
    description="Two parents, a child of both with a nullable key, and a keyless grandchild.",
    license="none",
    terms="none",
    tables=(
        RealTable("user", "x.y.user", 300, "id bigint, name string", ("id",)),
        RealTable("place", "x.y.place", 60, "id bigint", ("id",)),
        RealTable("visit", "x.y.visit", 1_000, "vid bigint, uid bigint, pid bigint", ("vid",)),
        RealTable("note", "x.y.note", 2_000, "vid bigint, value double"),
    ),
    foreign_keys=(
        _fk("visit", "uid", "user", "id"),
        _fk("visit", "pid", "place", "id"),
        _fk("note", "vid", "visit", "vid"),
    ),
)


def _tables() -> dict[str, pa.Table]:
    rng = np.random.default_rng(3)
    return {
        "user": pa.table({"id": np.arange(300), "name": [f"u{i}" for i in range(300)]}),
        "place": pa.table({"id": np.arange(60)}),
        "visit": pa.table(
            {
                "vid": np.arange(1_000),
                "uid": rng.integers(0, 300, 1_000),
                "pid": pa.array(rng.integers(0, 60, 1_000), mask=np.arange(1_000) % 4 == 0),
            }
        ),
        "note": pa.table({"vid": rng.integers(0, 1_000, 2_000), "value": rng.normal(size=2_000)}),
    }


def _rows(table: pa.Table) -> set[tuple[object, ...]]:
    return set(zip(*(table.column(c).to_pylist() for c in table.column_names), strict=True))


def test_split_is_deterministic_and_independent_of_row_order() -> None:
    tables = _tables()
    shuffled = {n: t.take(np.random.default_rng(1).permutation(t.num_rows)) for n, t in tables.items()}
    first, again, reordered = split(SPLIT_FIXTURE, tables), split(SPLIT_FIXTURE, tables), split(SPLIT_FIXTURE, shuffled)
    assert list(first) == list(PARTS)
    for part in PARTS:
        for name in tables:
            assert first[part][name].equals(again[part][name])
            assert _rows(first[part][name]) == _rows(reordered[part][name])


def test_split_parts_are_disjoint_and_nonempty() -> None:
    tables = _tables()
    parts = split(SPLIT_FIXTURE, tables)
    for name, table in tables.items():
        rows = [_rows(parts[part][name]) for part in PARTS]
        assert all(rows), name
        assert sum(len(r) for r in rows) == len(set().union(*rows))
        assert set().union(*rows) <= _rows(table)
    assert sum(parts[p]["user"].num_rows for p in PARTS) == 300


def test_split_parts_satisfy_foreign_keys_and_drop_straddling_rows() -> None:
    tables = _tables()
    parts = split(SPLIT_FIXTURE, tables)
    for part in PARTS:
        for fk in SPLIT_FIXTURE.foreign_keys:
            child = parts[part][fk.child_table].column(fk.child_columns[0]).drop_null().to_pylist()
            parent = set(parts[part][fk.parent_table].column(fk.parent_columns[0]).to_pylist())
            assert set(child) <= parent, (part, fk)
    kept = sum(parts[p]["visit"].num_rows for p in PARTS)
    assert 0 < kept < tables["visit"].num_rows
    null_place = sum(parts[p]["visit"].column("pid").null_count for p in PARTS)
    assert null_place == sum(1 for v in tables["visit"].column("pid").to_pylist() if v is None)


def _cache_full(root: Path) -> None:
    tables = _tables()
    target = instance_dir(SPLIT_FIXTURE, "full", root)
    target.mkdir(parents=True)
    write_instance(target, tables, SPLIT_FIXTURE.manifest("full", tables), 0)


def test_materialize_part_splits_a_cached_export_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(WAREHOUSE_ENV, raising=False)
    _cache_full(tmp_path)
    instances = {part: materialize_part(SPLIT_FIXTURE, "s", part, tmp_path) for part in PARTS}
    expected = split(SPLIT_FIXTURE, _tables())
    for part, instance in instances.items():
        assert instance.path == part_dir(SPLIT_FIXTURE, "s", part, tmp_path)
        assert instance.manifest.ground_truth == "none"
        assert _rows(instance.read_table("visit")) == _rows(expected[part]["visit"])
    assert all((part_dir(SPLIT_FIXTURE, "full", part, tmp_path) / "meta.json").exists() for part in PARTS)


def test_materialize_part_without_export_or_warehouse_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(WAREHOUSE_ENV, raising=False)
    with pytest.raises(Unavailable, match=WAREHOUSE_ENV):
        materialize_part(SPLIT_FIXTURE, "s", "profiled", tmp_path)
    with pytest.raises(KeyError):
        materialize_part(SPLIT_FIXTURE, "xs", "profiled", tmp_path)
