from collections.abc import Iterator
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from profiler_bench.catalog.store import CACHE_ENV


@pytest.fixture(scope="session", autouse=True)
def catalog_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Points the catalog cache at a per-run temporary directory."""
    root = tmp_path_factory.mktemp("catalog")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(CACHE_ENV, str(root))
        yield root


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Returns a local Spark session with a UTC session time zone."""
    active = SparkSession.getActiveSession()
    if active is not None:
        active.conf.set("spark.sql.session.timeZone", "UTC")
        return active
    return (
        SparkSession.builder.master("local[2]")
        .appName("profiler_bench_test")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
