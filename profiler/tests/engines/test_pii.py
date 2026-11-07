import json
import re
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from dbldatagen_profiler.cli import main
from dbldatagen_profiler.engines.spark import profile_dataframe
from dbldatagen_profiler.model import EngineInfo, PrivacyPolicy, Profile, Relationship, TableProfile, dump_profile
from dbldatagen_profiler.pii import SubstitutionOptions, detect, substitute


pytestmark = [pytest.mark.spark, pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")]


def test_detects_validators_and_negatives_deterministically(spark: SparkSession) -> None:
    rows = [(f"user{i}@example.com", "4532015112830366", f"SKU-{i:016d}", "1234567890123456") for i in range(20)]
    frame = spark.createDataFrame(rows, "email string, card_number string, sku string, random_digits string")
    first = detect({"t": frame.repartition(1)}, sample_rows=20)["t"]
    second = detect({"t": frame.repartition(4)}, sample_rows=20)["t"]
    assert first == second
    assert first["email"].pii_class.value == "email"
    assert first["card_number"].pii_class.value == "credit_card"
    assert "sku" not in first
    assert "random_digits" not in first
    assert all(
        set(item.model_dump()) == {"method", "match_rate"} for value in first.values() for item in value.evidence
    )


def test_substitution_preserves_relationships_and_card_validity(spark: SparkSession) -> None:
    parents = spark.createDataFrame(
        [("a@example.com", "4532015112830366"), ("b@example.com", "4532015112830366")],
        "email string, card string",
    )
    children = spark.createDataFrame([("a@example.com",), ("a@example.com",), ("b@example.com",)], "email string")
    semantics = detect({"parents": parents, "children": children})
    parent_profile = TableProfile(
        name="parents",
        row_count=2,
        columns=[
            _column("email", semantics["parents"]["email"]),
            _column("card", semantics["parents"]["card"]),
        ],
    )
    child_profile = TableProfile(
        name="children", row_count=3, columns=[_column("email", semantics["children"]["email"])]
    )
    profile = Profile(
        spec_version="0.2.0",
        created_at="2026-11-07T12:00:00Z",
        engine=EngineInfo(name="test", version="1"),
        privacy=PrivacyPolicy(),
        tables=[parent_profile, child_profile],
        relationships=[
            Relationship(
                child_table="children",
                child_columns=["email"],
                parent_table="parents",
                parent_columns=["email"],
                coverage=1.0,
            )
        ],
    )
    result = substitute({"parents": parents, "children": children}, profile, key=b"test-key")
    parent_values = {row.email for row in result.tables["parents"].select("email").collect()}
    child_values = [row.email for row in result.tables["children"].select("email").collect()]
    assert set(child_values) == parent_values
    assert len(set(child_values)) == 2
    assert all(value not in {"a@example.com", "b@example.com"} for value in child_values)
    row = result.tables["parents"].first()
    assert row is not None
    card = row.card
    assert _luhn(card)
    assert result.report["columns"]


def test_generalization_and_collision_hook(spark: SparkSession) -> None:
    frame = spark.createDataFrame([("12345",), ("12346",)], "postal_code string")
    semantic = detect({"t": frame})["t"]["postal_code"]
    profile = Profile(
        spec_version="0.2.0",
        created_at="2026-11-07T12:00:00Z",
        engine=EngineInfo(name="test", version="1"),
        tables=[TableProfile(name="t", row_count=2, columns=[_column("postal_code", semantic)])],
    )
    result = substitute(
        {"t": frame}, profile, key=b"key", options=SubstitutionOptions(generalize={"t.postal_code": "postal_prefix"})
    )
    assert {row.postal_code for row in result.tables["t"].collect()} == {"123**"}


def test_forced_collisions_are_resolved(spark: SparkSession) -> None:
    frame = spark.createDataFrame([(str(value),) for value in range(100, 108)], "account_id string")
    semantic = {
        "pii_class": "account_id",
        "sensitivity": "direct_identifier",
        "confidence": 1.0,
        "evidence": [{"method": "column_name", "match_rate": 1.0}],
    }
    profile = Profile(
        spec_version="0.2.0",
        created_at="2026-11-07T12:00:00Z",
        engine=EngineInfo(name="test", version="1"),
        tables=[TableProfile(name="t", row_count=8, columns=[_column("account_id", semantic)])],
    )
    result = substitute({"t": frame}, profile, key=b"collision-key", options=SubstitutionOptions(surrogate_bits=3))
    output = [row.account_id for row in result.tables["t"].collect()]
    assert len(set(output)) == 8
    assert not set(output) & {str(value) for value in range(100, 108)}
    assert result.report["columns"][0]["collisions_resolved"] > 0


def test_cli_detect_and_substitute_end_to_end(spark: SparkSession, tmp_path: Path) -> None:
    source = tmp_path / "source"
    semantic_path = tmp_path / "semantic.json"
    profile_path = tmp_path / "profile.json"
    key_path = tmp_path / "key.bin"
    output = tmp_path / "output"
    frame = spark.createDataFrame([("a@example.com",), ("b@example.com",)], "email string")
    frame.write.parquet(str(source))
    assert main(["pii", "detect", "--table", f"customers={source}", "-o", str(semantic_path)]) == 0
    assert json.loads(semantic_path.read_text())["customers"]["email"]["pii_class"] == "email"
    profile_path.write_text(dump_profile(profile_dataframe("customers", frame)), encoding="utf-8")
    key_path.write_bytes(b"cli-test-key")
    assert (
        main(
            [
                "pii",
                "substitute",
                "--table",
                f"customers={source}",
                "--profile",
                str(profile_path),
                "--key-file",
                str(key_path),
                "-o",
                str(output),
            ]
        )
        == 0
    )
    values = {row.email for row in spark.read.parquet(str(output / "customers")).collect()}
    assert values.isdisjoint({"a@example.com", "b@example.com"})
    assert json.loads((output / "report.json").read_text())["columns"][0]["values_substituted"] == 2


def _column(name: str, semantic: object) -> dict[str, object]:
    return {
        "name": name,
        "logical_type": "string",
        "source_type": "string",
        "nullable": True,
        "null_fraction": 0.0,
        "semantic": semantic,
    }


def _luhn(value: str) -> bool:
    digits = list(map(int, re.sub(r"\D", "", value)))
    total = 0
    for index, digit in enumerate(reversed(digits)):
        number = digit * 2 if index % 2 else digit
        total += number - 9 if number > 9 else number
    return total % 10 == 0
