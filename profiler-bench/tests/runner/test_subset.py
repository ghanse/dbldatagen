"""Deterministic FK-preserving subset sampling for the data-volume objective."""

from __future__ import annotations

from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from profiler_bench.catalog import load, materialize
from profiler_bench.runner.subset import apply_table_scales, subset_tables


def test_subset_fraction_one_is_identity(spark: SparkSession, catalog_cache: Path) -> None:
    del catalog_cache
    frames = load("numeric", "xs", seed=0, spark=spark)
    manifest = materialize("numeric", "xs", seed=0).manifest
    first = subset_tables(frames, manifest, fraction=1.0, subset_seed=0)
    assert first.subset_row_counts == first.full_row_counts
    assert all(scale == pytest.approx(1.0) for scale in first.scales.values())


def test_subset_is_deterministic(spark: SparkSession, catalog_cache: Path) -> None:
    del catalog_cache
    frames = load("retail", "xs", seed=0, spark=spark)
    manifest = materialize("retail", "xs", seed=0).manifest
    first = subset_tables(frames, manifest, fraction=0.1, subset_seed=7)
    second = subset_tables(frames, manifest, fraction=0.1, subset_seed=7)
    assert first.subset_row_counts == second.subset_row_counts
    assert first.scales == second.scales
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
