"""Spark fixtures for engine tests."""

from collections.abc import Iterator

import pytest
from pyspark.sql import SparkSession


def pytest_configure(config: pytest.Config) -> None:
    """Registers the Spark marker."""
    config.addinivalue_line("markers", "spark: requires a local Spark session")


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    """Provides a session-scoped local Spark session."""
    session = (
        SparkSession.builder.master("local[2]")
        .appName("dbldatagen_profiler_test")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()
