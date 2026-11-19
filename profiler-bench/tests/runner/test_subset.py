"""Deterministic FK-preserving subset sampling for the data-volume objective."""

from __future__ import annotations

from pathlib import Path

import pytest
from pyspark.sql import DataFrame, SparkSession

from profiler_bench.catalog import load, materialize
from profiler_bench.catalog.manifest import Column, ForeignKey, Manifest, Table
from profiler_bench.runner import subset as subset_mod
from profiler_bench.runner.subset import apply_table_scales, subset_tables


def test_subset_fraction_one_is_identity(spark: SparkSession, catalog_cache: Path) -> None:
    del catalog_cache
    frames = load("numeric", "xs", seed=0, spark=spark)
    manifest = materialize("numeric", "xs", seed=0).manifest
    first = subset_tables(frames, manifest, fraction=1.0, subset_seed=0)
    assert first.subset_row_counts == first.full_row_counts
    assert all(scale == pytest.approx(1.0) for scale in first.scales.values())
    assert first.overall_actual_fraction == pytest.approx(1.0)


def test_subset_is_deterministic(spark: SparkSession, catalog_cache: Path) -> None:
    del catalog_cache
    frames = load("retail", "xs", seed=0, spark=spark)
    manifest = materialize("retail", "xs", seed=0).manifest
    first = subset_tables(frames, manifest, fraction=0.1, subset_seed=7)
    second = subset_tables(frames, manifest, fraction=0.1, subset_seed=7)
    assert first.subset_row_counts == second.subset_row_counts
    assert first.scales == second.scales
    assert first.actual_fractions == second.actual_fractions
    for name in first.tables:
        assert first.tables[name].collect() == second.tables[name].collect()


def test_subset_fk_preserving_retail(spark: SparkSession, catalog_cache: Path) -> None:
    del catalog_cache
    frames = load("retail", "xs", seed=0, spark=spark)
    manifest = materialize("retail", "xs", seed=0).manifest
    subset = subset_tables(frames, manifest, fraction=0.2, subset_seed=3)
    customers = {row.customer_id for row in subset.tables["customers"].collect()}
    products = {row.product_id for row in subset.tables["products"].collect()}
    orders = subset.tables["orders"].collect()
    items = subset.tables["order_items"].collect()
    assert customers
    assert products
    assert all(row.customer_id in customers for row in orders)
    order_ids = {row.order_id for row in orders}
    assert all(row.order_id in order_ids for row in items)
    assert all(row.product_id in products for row in items)
    assert subset.method == "fk_preserving_uniform"
    for name, full in subset.full_row_counts.items():
        kept = subset.subset_row_counts[name]
        assert 0 < kept <= full
        assert subset.scales[name] == pytest.approx(full / kept)
        assert subset.actual_fractions[name] == pytest.approx(kept / full)
    assert 0.0 < subset.overall_actual_fraction <= 1.0


def test_subset_self_reference(spark: SparkSession) -> None:
    employees = spark.createDataFrame(
        [(1, None), (2, 1), (3, 2), (4, 1), (5, 4), (6, 3), (7, 6), (8, 5)],
        ["id", "manager_id"],
    )
    manifest = Manifest(
        dataset="self_ref",
        tier="xs",
        structure_seed=0,
        description="org chart",
        tables=(Table(name="employees", columns=(Column(name="id"), Column(name="manager_id")), primary_key=("id",)),),
        foreign_keys=(
            ForeignKey(
                child_table="employees",
                child_columns=("manager_id",),
                parent_table="employees",
                parent_columns=("id",),
            ),
        ),
    )
    subset = subset_tables({"employees": employees}, manifest, fraction=0.25, subset_seed=11)
    rows = {row.id: row.manager_id for row in subset.tables["employees"].collect()}
    assert rows
    for emp_id, manager_id in rows.items():
        if manager_id is not None:
            assert manager_id in rows, f"employee {emp_id} references missing manager {manager_id}"


def test_subset_two_table_cycle(spark: SparkSession) -> None:
    left = spark.createDataFrame([(1, 10), (2, 20), (3, 30), (4, 40)], ["id", "right_id"])
    right = spark.createDataFrame([(10, 1), (20, 2), (30, 3), (40, 4)], ["id", "left_id"])
    manifest = Manifest(
        dataset="cycle",
        tier="xs",
        structure_seed=0,
        description="two-table cycle",
        tables=(
            Table(name="left", columns=(Column(name="id"), Column(name="right_id")), primary_key=("id",)),
            Table(name="right", columns=(Column(name="id"), Column(name="left_id")), primary_key=("id",)),
        ),
        foreign_keys=(
            ForeignKey(child_table="left", child_columns=("right_id",), parent_table="right", parent_columns=("id",)),
            ForeignKey(child_table="right", child_columns=("left_id",), parent_table="left", parent_columns=("id",)),
        ),
    )
    subset = subset_tables({"left": left, "right": right}, manifest, fraction=0.5, subset_seed=5)
    left_ids = {row.id for row in subset.tables["left"].collect()}
    right_ids = {row.id for row in subset.tables["right"].collect()}
    for row in subset.tables["left"].collect():
        assert row.right_id in right_ids
    for row in subset.tables["right"].collect():
        assert row.left_id in left_ids
    assert left_ids and right_ids


def test_subset_composite_fk(spark: SparkSession) -> None:
    parents = spark.createDataFrame(
        [(1, "a"), (1, "b"), (2, "a"), (2, "b"), (3, "a")],
        ["group_id", "code"],
    )
    children = spark.createDataFrame(
        [(10, 1, "a"), (11, 1, "b"), (12, 2, "a"), (13, 2, "b"), (14, 3, "a")],
        ["id", "group_id", "code"],
    )
    manifest = Manifest(
        dataset="composite",
        tier="xs",
        structure_seed=0,
        description="composite FK",
        tables=(
            Table(
                name="parents",
                columns=(Column(name="group_id"), Column(name="code")),
                primary_key=("group_id", "code"),
            ),
            Table(
                name="children",
                columns=(Column(name="id"), Column(name="group_id"), Column(name="code")),
                primary_key=("id",),
            ),
        ),
        foreign_keys=(
            ForeignKey(
                child_table="children",
                child_columns=("group_id", "code"),
                parent_table="parents",
                parent_columns=("group_id", "code"),
            ),
        ),
    )
    subset = subset_tables({"parents": parents, "children": children}, manifest, fraction=0.4, subset_seed=9)
    parent_keys = {(row.group_id, row.code) for row in subset.tables["parents"].collect()}
    for row in subset.tables["children"].collect():
        assert (row.group_id, row.code) in parent_keys


def test_subset_multi_parent_child_pulls_parents(spark: SparkSession) -> None:
    users = spark.createDataFrame([(1,), (2,), (3,), (4,)], ["user_id"])
    projects = spark.createDataFrame([(10,), (20,), (30,), (40,)], ["project_id"])
    # Force a sparse independent sample of memberships so upward closure must pull parents.
    memberships = spark.createDataFrame(
        [(100, 1, 10), (101, 2, 20), (102, 3, 30), (103, 4, 40)],
        ["id", "user_id", "project_id"],
    )
    manifest = Manifest(
        dataset="multi_parent",
        tier="xs",
        structure_seed=0,
        description="child with two parents",
        tables=(
            Table(name="users", columns=(Column(name="user_id"),), primary_key=("user_id",)),
            Table(name="projects", columns=(Column(name="project_id"),), primary_key=("project_id",)),
            Table(
                name="memberships",
                columns=(Column(name="id"), Column(name="user_id"), Column(name="project_id")),
                primary_key=("id",),
            ),
        ),
        foreign_keys=(
            ForeignKey(
                child_table="memberships",
                child_columns=("user_id",),
                parent_table="users",
                parent_columns=("user_id",),
            ),
            ForeignKey(
                child_table="memberships",
                child_columns=("project_id",),
                parent_table="projects",
                parent_columns=("project_id",),
            ),
        ),
    )
    subset = subset_tables(
        {"users": users, "projects": projects, "memberships": memberships},
        manifest,
        fraction=0.25,
        subset_seed=2,
    )
    user_ids = {row.user_id for row in subset.tables["users"].collect()}
    project_ids = {row.project_id for row in subset.tables["projects"].collect()}
    members = subset.tables["memberships"].collect()
    assert members  # independent sample kept at least one child
    for row in members:
        assert row.user_id in user_ids
        assert row.project_id in project_ids


def test_subset_isolated_table(spark: SparkSession) -> None:
    alone = spark.createDataFrame([(i, f"v{i}") for i in range(20)], ["id", "value"])
    manifest = Manifest(
        dataset="isolated",
        tier="xs",
        structure_seed=0,
        description="no FKs",
        tables=(Table(name="alone", columns=(Column(name="id"), Column(name="value")), primary_key=("id",)),),
    )
    subset = subset_tables({"alone": alone}, manifest, fraction=0.2, subset_seed=0)
    assert 1 <= subset.subset_row_counts["alone"] <= 20
    assert subset.actual_fractions["alone"] == pytest.approx(subset.subset_row_counts["alone"] / 20)


def test_subset_tiny_fraction_keeps_at_least_one_row(spark: SparkSession) -> None:
    frame = spark.createDataFrame([(i,) for i in range(50)], ["id"])
    manifest = Manifest(
        dataset="tiny",
        tier="xs",
        structure_seed=0,
        description="tiny f",
        tables=(Table(name="t", columns=(Column(name="id"),), primary_key=("id",)),),
    )
    subset = subset_tables({"t": frame}, manifest, fraction=0.001, subset_seed=99)
    assert subset.subset_row_counts["t"] >= 1
    assert "t" in subset.scales
    assert subset.scales["t"] == pytest.approx(50 / subset.subset_row_counts["t"])


def test_subset_determinism_across_partition_counts(spark: SparkSession) -> None:
    rows = [(i, i % 7, f"x{i}") for i in range(40)]
    base = spark.createDataFrame(rows, ["id", "group_id", "label"])
    one = base.repartition(1)
    many = base.repartition(7)
    manifest = Manifest(
        dataset="parts",
        tier="xs",
        structure_seed=0,
        description="partition independence",
        tables=(
            Table(
                name="t",
                columns=(Column(name="id"), Column(name="group_id"), Column(name="label")),
                primary_key=("id",),
            ),
        ),
    )
    first = subset_tables({"t": one}, manifest, fraction=0.3, subset_seed=42)
    second = subset_tables({"t": many}, manifest, fraction=0.3, subset_seed=42)
    assert {row.id for row in first.tables["t"].collect()} == {row.id for row in second.tables["t"].collect()}


def test_apply_table_scales_sets_full_row_counts() -> None:
    profile = {
        "tables": [
            {"name": "a", "row_count": 100},
            {"name": "b", "row_count": 40},
        ]
    }
    scaled = apply_table_scales(profile, {"a": 10.0, "b": 2.5})
    assert scaled["tables"][0]["row_count"] == 1000
    assert scaled["tables"][1]["row_count"] == 100
    assert profile["tables"][0]["row_count"] == 100


def _self_ref_chain(spark: SparkSession, depth: int) -> tuple[DataFrame, Manifest]:
    """Builds a depth-long manager chain ending at a root with manager_id=None."""
    rows: list[tuple[int, int | None]] = [(1, None)]
    for emp_id in range(2, depth + 1):
        rows.append((emp_id, emp_id - 1))
    employees = spark.createDataFrame(rows, ["id", "manager_id"])
    manifest = Manifest(
        dataset="self_ref_chain",
        tier="xs",
        structure_seed=0,
        description="long self-ref chain",
        tables=(Table(name="employees", columns=(Column(name="id"), Column(name="manager_id")), primary_key=("id",)),),
        foreign_keys=(
            ForeignKey(
                child_table="employees",
                child_columns=("manager_id",),
                parent_table="employees",
                parent_columns=("id",),
            ),
        ),
    )
    return employees, manifest


def test_deep_self_ref_chain_converges_within_cap(spark: SparkSession) -> None:
    employees, manifest = _self_ref_chain(spark, depth=200)
    # Seeded hash sample of only the leaf; upward closure must walk the full chain.
    leaf = employees.where("id = 200")
    full = {"employees": employees}
    selected = {"employees": leaf}
    closed = subset_mod._upward_closure(
        full,
        selected,
        manifest.foreign_keys,
        max_closure_iterations=200,
    )
    ids = {row.id for row in closed["employees"].collect()}
    assert ids == set(range(1, 201))


def test_closure_exceeding_small_cap_fails_loudly(spark: SparkSession) -> None:
    employees, manifest = _self_ref_chain(spark, depth=20)
    leaf = employees.where("id = 20")
    with pytest.raises(RuntimeError, match="max_closure_iterations=5"):
        subset_mod._upward_closure(
            {"employees": employees},
            {"employees": leaf},
            manifest.foreign_keys,
            max_closure_iterations=5,
        )


def test_closure_job_count_per_iteration_constant(spark: SparkSession) -> None:
    """Lineage truncation keeps Spark job count flat across closure iterations."""
    employees, manifest = _self_ref_chain(spark, depth=24)
    leaf = employees.where("id = 24")
    sc = spark.sparkContext
    group = "dv-closure-job-const"
    sc.setJobGroup(group, "closure constancy", interruptOnCancel=False)
    before = len(sc.statusTracker().getJobIdsForGroup(group))
    snapshots: list[int] = []

    def on_iteration_abs(_depth: int, changed: bool) -> None:
        if changed:
            snapshots.append(len(sc.statusTracker().getJobIdsForGroup(group)))

    try:
        closed = subset_mod._upward_closure(
            {"employees": employees},
            {"employees": leaf},
            manifest.foreign_keys,
            max_closure_iterations=64,
            _on_iteration=on_iteration_abs,
        )
    finally:
        sc.setLocalProperty("spark.jobGroup.id", "")
        sc.setLocalProperty("spark.job.description", "")
    assert {row.id for row in closed["employees"].collect()} == set(range(1, 25))
    assert len(snapshots) >= 10
    per_iter = [snapshots[0] - before] + [snapshots[i] - snapshots[i - 1] for i in range(1, len(snapshots))]
    # After the first materialization-heavy iteration, each subsequent iteration should stay
    # within a small constant band (no lineage-driven growth with depth).
    steady = per_iter[1:]
    assert max(steady) <= min(steady) + 2, f"job counts grew with depth: {per_iter}"
