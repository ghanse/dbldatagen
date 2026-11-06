import pytest
from pyspark.sql import SparkSession

from profiler_bench.catalog import Realized, realized_statistics


def test_realized_statistics_reports_what_the_instance_contains(spark: SparkSession) -> None:
    stats = realized_statistics("retail", "xs", 0, spark)
    assert isinstance(stats, Realized)
    orders = next(t for t in stats.tables if t.name == "orders")
    ship_date = next(c for c in orders.columns if c.name == "ship_date")
    assert ship_date.null_rate == pytest.approx(0.15, abs=0.05)
    status = next(c for c in orders.columns if c.name == "status")
    assert sum(status.category_shares.values()) == pytest.approx(1.0)
    (customers_fk, items_fk, _) = stats.foreign_keys
    assert customers_fk.coverage == items_fk.coverage == 1.0
    assert items_fk.fanout_mean == pytest.approx(4.0, abs=0.2)
    assert items_fk.fanout_zero_share == 0.0


def test_realized_statistics_cover_dependencies_and_quasi_identifiers(spark: SparkSession) -> None:
    stats = realized_statistics("strings", "xs", 0, spark)
    (contacts,) = stats.tables
    (dependency,) = contacts.dependencies
    assert dependency.measure == "cramers_v"
    assert 0.2 < dependency.value < 0.35
    (qi,) = contacts.quasi_identifiers
    assert qi.unique_fraction > 0.99
    email = next(c for c in contacts.columns if c.name == "email")
    assert list(email.format_shares.values()) == [1.0]
