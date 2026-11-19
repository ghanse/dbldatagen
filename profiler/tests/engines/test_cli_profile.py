"""End-to-end CLI tests for Spark profiling."""

import json
from datetime import date
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from dbldatagen_profiler.cli import main
from dbldatagen_profiler.model import PiiClass, load_profile
from dbldatagen_profiler.pii import PiiAction, PiiPolicy, PiiRule, dump_policy_json


pytestmark = [pytest.mark.spark, pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")]


def test_profile_parquet_end_to_end(spark: SparkSession, tmp_path: Path) -> None:
    source = tmp_path / "input.parquet"
    output = tmp_path / "profile.json"
    policy_output = tmp_path / "profile-policy.json"
    policy = tmp_path / "policy.json"
    policy.write_text(
        dump_policy_json(PiiPolicy(pii_class={PiiClass.DATE_OF_BIRTH: PiiRule(action=PiiAction.DROP)})),
        encoding="utf-8",
    )
    spark.createDataFrame(
        [(index, date(1980 + index, 1, 2), f"user{index}@example.com") for index in range(10)],
        "id long, date_of_birth date, email string",
    ).write.parquet(str(source))
    assert main(["profile", "--table", f"items={source}", "--format", "parquet", "-o", str(output)]) == 0
    assert (
        main(
            [
                "profile",
                "--table",
                f"items={source}",
                "--format",
                "parquet",
                "--policy",
                str(policy),
                "-o",
                str(policy_output),
            ]
        )
        == 0
    )
    profile = load_profile(json.loads(output.read_text(encoding="utf-8")), strict=True)
    policy_profile = load_profile(json.loads(policy_output.read_text(encoding="utf-8")), strict=True)
    assert profile.tables[0].name == "items"
    assert profile.tables[0].row_count == 10
    assert profile.source is not None
    assert profile.source.name == "items"
    assert profile.source.format == "parquet"
    assert profile.source.location is None
    assert str(source) not in output.read_text(encoding="utf-8")
    dob = next(column for column in profile.tables[0].columns if column.name == "date_of_birth")
    protected_dob = next(column for column in policy_profile.tables[0].columns if column.name == "date_of_birth")
    assert dob.temporal is not None and dob.temporal.min is not None and dob.temporal.max is not None
    assert protected_dob.temporal is not None
    assert protected_dob.temporal.min is None and protected_dob.temporal.max is None


def test_profile_reports_spark_read_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "profile.json"
    missing = tmp_path / "missing.parquet"
    result = main(["profile", "--table", f"missing={missing}", "--format", "parquet", "-o", str(output)])
    assert result == 1
    assert "dbldatagen-profile profile:" in capsys.readouterr().err
    assert not output.exists()


def test_profile_policy_rejects_unknown_column_rule(
    spark: SparkSession, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "input.parquet"
    output = tmp_path / "profile.json"
    policy_path = tmp_path / "policy.json"
    spark.createDataFrame([("Boston",)], "city string").write.parquet(str(source))
    policy_path.write_text(
        dump_policy_json(PiiPolicy(columns={"items.ctiy": PiiRule(action=PiiAction.SUPPRESS)})),
        encoding="utf-8",
    )
    result = main(
        [
            "profile",
            "--table",
            f"items={source}",
            "--policy",
            str(policy_path),
            "-o",
            str(output),
        ]
    )
    assert result == 1
    assert "match no input column" in capsys.readouterr().err
    assert not output.exists()
