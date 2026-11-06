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
    assert profile.source is not None
    assert profile.source.name == "items"
    assert profile.source.format == "parquet"
    assert profile.source.location is None
    assert str(source) not in output.read_text(encoding="utf-8")


def test_profile_reports_spark_read_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "profile.json"
    missing = tmp_path / "missing.parquet"
    result = main(["profile", "--table", f"missing={missing}", "--format", "parquet", "-o", str(output)])
    assert result == 1
    assert "dbldatagen-profile profile:" in capsys.readouterr().err
    assert not output.exists()
