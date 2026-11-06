"""End-to-end CLI tests for Spark profiling."""

import json
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from dbldatagen_profiler.cli import main
from dbldatagen_profiler.model import load_profile


pytestmark = [pytest.mark.spark, pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")]


def test_profile_parquet_end_to_end(spark: SparkSession, tmp_path: Path) -> None:
    source = tmp_path / "input.parquet"
    output = tmp_path / "profile.json"
    spark.createDataFrame([(1, "a"), (2, None)], "id long, label string").write.parquet(str(source))
    assert main(["profile", "--table", f"items={source}", "--format", "parquet", "-o", str(output)]) == 0
    profile = load_profile(json.loads(output.read_text(encoding="utf-8")), strict=True)
    assert profile.tables[0].name == "items"
    assert profile.tables[0].row_count == 2
