"""Tests for incremental profiling with Spark Structured Streaming."""

import json
import math
import os
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from pyspark.errors import StreamingQueryException
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from dbldatagen_profiler.cli import main
from dbldatagen_profiler.engines.spark import profile_dataframe
from dbldatagen_profiler.engines.streaming import (
    DISTINCT_RELATIVE_ERROR,
    StreamingProfiler,
    TableState,
    empty_state,
    merge_states,
    profile_states,
    state_for_dataframe,
)
from dbldatagen_profiler.engines.streaming import profiler as profiler_module
from dbldatagen_profiler.engines.streaming.profiler import PII_WARMUP_ROWS
from dbldatagen_profiler.engines.streaming.state import analyze_sample
from dbldatagen_profiler.model import (
    PiiClass,
    PrivacyPolicy,
    Profile,
    ValueTreatment,
    load_profile,
    profile_json_schema,
)


pytestmark = [pytest.mark.spark, pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")]

SCHEMA = (
    "id int, big long, dec decimal(38,0), d double, email string, s string, b boolean, dt date, ts timestamp, "
    "st struct<a:int,b:string>, arr array<int>, m map<string,int>"
)
INT64_MAX = 2**63 - 1
DECIMAL38_MAX = Decimal("9" * 38)
DECIMAL38_MIN = Decimal("-" + "9" * 38)


def _rows(count: int = 150) -> list[tuple[Any, ...]]:
    return [
        (
            i,
            INT64_MAX if i == 0 else -(2**63) if i == 1 else i * 7,
            DECIMAL38_MAX if i == 2 else DECIMAL38_MIN if i == 3 else Decimal(i),
            i / 3 if i % 5 else None,
            f"user{i}@example.com",
            f"s{i % 7}" if i % 3 else "",
            None if i % 11 == 0 else i % 2 == 0,
            date(2020, 1, 1 + i % 28),
            datetime(2021, 1, 1, 0, 0, i % 60, i * 13, tzinfo=timezone.utc),
            {"a": i, "b": f"x{i % 4}"} if i % 9 else None,
            [i, None, i % 3] if i % 4 else None,
            {f"k{i % 3}": i},
        )
        for i in range(count)
    ]


def _write_splits(df: DataFrame, path: Path, splits: int) -> None:
    """Writes `df` as `splits` parquet files whose modification times follow the split order."""
    for split in range(splits):
        before = set(path.glob("*.parquet")) if path.exists() else set()
        df.where(F.col("id") % splits == split).coalesce(1).write.mode("append").parquet(str(path))
        for file in set(path.glob("*.parquet")) - before:
            os.utime(file, (1_700_000_000 + split, 1_700_000_000 + split))


def _stream(spark: SparkSession, path: Path, schema: T.StructType) -> DataFrame:
    return spark.readStream.schema(schema).option("maxFilesPerTrigger", 1).parquet(str(path))


def _run(
    spark: SparkSession,
    path: Path,
    checkpoint: Path,
    *,
    privacy: PrivacyPolicy | None = None,
    pii_warmup_rows: int = PII_WARMUP_ROWS,
) -> Profile:
    profiler = StreamingProfiler(
        {"t": _stream(spark, path, spark.read.parquet(str(path)).schema)},
        checkpoint_dir=checkpoint,
        privacy=privacy,
        pii_warmup_rows=pii_warmup_rows,
    )
    profiler.start()
    profiler.await_termination()
    profiler.stop()
    return profiler.profile()


def _comparable(profile: Profile) -> dict[str, Any]:
    data = profile.model_dump(mode="json")
    data.pop("created_at")
    data.pop("engine")
    return data


def _assert_equivalent(batch: object, stream: object, path: str = "") -> None:
    """Asserts exact equality except for float moments (rounding) and distinct counts (sketch bound)."""
    if isinstance(batch, dict):
        assert isinstance(stream, dict), path
        assert set(batch) == set(stream), path
        for key in batch:
            if key == "distinct_count":
                expected, actual = batch[key]["value"], stream[key]["value"]
                assert stream[key]["exact"] is False
                assert abs(actual - expected) <= max(1.0, 3 * DISTINCT_RELATIVE_ERROR * expected), path
            elif path.endswith("numeric") and key in {"mean", "stddev"} and batch[key] is not None:
                scale = max(1.0, abs(batch["mean"]), batch["stddev"] or 0.0)
                assert math.isclose(batch[key], stream[key], rel_tol=1e-9, abs_tol=1e-9 * scale), f"{path}.{key}"
            elif path.endswith("collection_length") and key == "stddev" and batch[key] is not None:
                assert math.isclose(batch[key], stream[key], rel_tol=1e-9), f"{path}.{key}"
            else:
                _assert_equivalent(batch[key], stream[key], f"{path}.{key}")
    elif isinstance(batch, list):
        assert isinstance(stream, list), path
        assert len(batch) == len(stream), path
        for index, (left, right) in enumerate(zip(batch, stream, strict=True)):
            _assert_equivalent(left, right, f"{path}[{index}]")
    else:
        assert batch == stream, path


@pytest.mark.parametrize("splits", [1, 3, 10])
def test_stream_matches_batch_profile(spark: SparkSession, tmp_path: Path, splits: int) -> None:
    source = tmp_path / "source"
    _write_splits(spark.createDataFrame(_rows(), SCHEMA), source, splits)
    batch = profile_dataframe("t", spark.read.parquet(str(source)))
    stream = _run(spark, source, tmp_path / "checkpoint")
    _assert_equivalent(_comparable(batch), _comparable(stream))
    columns = {column.name: column for column in stream.tables[0].columns}
    assert stream.tables[0].row_count == 150
    assert columns["big"].numeric is not None and columns["dec"].numeric is not None
    assert (columns["big"].numeric.min, columns["big"].numeric.max) == (str(-(2**63)), str(INT64_MAX))
    assert (columns["dec"].numeric.min, columns["dec"].numeric.max) == (str(DECIMAL38_MIN), str(DECIMAL38_MAX))
    assert columns["email"].semantic is not None and columns["email"].semantic.pii_class == PiiClass.EMAIL
    assert [child.value_count for child in columns["arr"].children or []] == [336]


def test_merge_is_associative_and_commutative(spark: SparkSession) -> None:
    df = spark.createDataFrame(_rows(), SCHEMA)
    a, b, c = (state_for_dataframe(df.where(F.col("id") % 3 == k)) for k in range(3))

    def snapshot(state: TableState) -> dict[str, Any]:
        return _comparable(profile_states({"t": state}, pii={"t": analyze_sample(state)}))

    reference = snapshot(merge_states(merge_states(a, b), c))
    for state in (
        merge_states(a, merge_states(b, c)),
        merge_states(c, merge_states(b, a)),
        merge_states(empty_state(df.schema), merge_states(b, merge_states(c, a))),
    ):
        _assert_equivalent(reference, snapshot(state))
        assert snapshot(state)["tables"][0]["columns"][4]["distinct_count"] == {"value": 150, "exact": False}
    assert [entry[:2] for entry in merge_states(a, b).sample] == [entry[:2] for entry in merge_states(b, a).sample]


@pytest.mark.parametrize("persisted", [False, True])
def test_restart_after_crash_gives_same_profile(
    spark: SparkSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, persisted: bool
) -> None:
    source = tmp_path / "source"
    _write_splits(spark.createDataFrame(_rows(60), SCHEMA), source, 3)
    expected = _run(spark, source, tmp_path / "reference")
    write = profiler_module._write_checkpoint

    def crash(directory: Path, checkpoint: profiler_module.Checkpoint) -> None:
        if persisted:
            write(directory, checkpoint)
        if checkpoint.batch_id == 1:
            raise RuntimeError("simulated crash")
        if not persisted:
            write(directory, checkpoint)

    monkeypatch.setattr(profiler_module, "_write_checkpoint", crash)
    with pytest.raises(StreamingQueryException):
        _run(spark, source, tmp_path / "checkpoint")
    monkeypatch.undo()
    restarted = _run(spark, source, tmp_path / "checkpoint")
    assert restarted.tables[0].row_count == 60
    assert _comparable(restarted) == _comparable(expected)


def test_mid_stream_snapshots_validate_against_json_schema(spark: SparkSession, tmp_path: Path) -> None:
    source, checkpoint = tmp_path / "source", tmp_path / "checkpoint"
    df = spark.createDataFrame(_rows(), SCHEMA)
    _write_splits(df.where("id < 50"), source, 1)
    schema = spark.read.parquet(str(source)).schema
    empty = StreamingProfiler({"t": _stream(spark, source, schema)}, checkpoint_dir=checkpoint).profile()
    first = _run(spark, source, checkpoint)
    _write_splits(df.where("id >= 50"), source, 1)
    final = _run(spark, source, checkpoint)
    for snapshot in (empty, first, final):
        jsonschema.validate(json.loads(snapshot.model_dump_json()), profile_json_schema())
    assert [snapshot.tables[0].row_count for snapshot in (empty, first, final)] == [0, 50, 150]


def test_pii_labels_freeze_after_warmup(spark: SparkSession, tmp_path: Path) -> None:
    source = tmp_path / "source"
    emails = [(i, f"person{i}@example.com") for i in range(30)]
    words = [(i, f"hello world {i % 5}") for i in range(30, 130)]
    df = spark.createDataFrame(emails + words, "id int, contact string")
    _write_splits(df.withColumn("id", F.when(F.col("id") < 30, 0).otherwise(1)), source, 2)
    frozen = _run(spark, source, tmp_path / "frozen", pii_warmup_rows=20)
    unfrozen = _run(spark, source, tmp_path / "unfrozen")
    frozen_label, unfrozen_label = (profile.tables[0].columns[1].semantic for profile in (frozen, unfrozen))
    assert frozen_label is not None and frozen_label.pii_class == PiiClass.EMAIL
    assert unfrozen_label is None
    assert frozen.tables[0].row_count == unfrozen.tables[0].row_count == 130
    assert not (tmp_path / "frozen" / "t" / "state.pkl").read_bytes().count(b"person0@example.com")


def test_snapshot_follows_suppressed_extremes(spark: SparkSession, tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_splits(spark.createDataFrame(_rows(40), SCHEMA), source, 2)
    profile = _run(spark, source, tmp_path / "checkpoint", privacy=PrivacyPolicy(extremes=ValueTreatment.SUPPRESSED))
    columns = {column.name: column for column in profile.tables[0].columns}
    assert columns["big"].numeric is not None and columns["big"].numeric.min is None
    assert columns["ts"].temporal is not None and columns["ts"].temporal.max is None
    load_profile(profile.model_dump(mode="json"), strict=True)


def test_rejects_pii_policy_and_mismatched_checkpoint(spark: SparkSession, tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_splits(spark.createDataFrame([(1, "a")], "id int, label string"), source, 1)
    schema = spark.read.parquet(str(source)).schema
    with pytest.raises(NotImplementedError):
        StreamingProfiler({"t": _stream(spark, source, schema)}, checkpoint_dir=tmp_path / "c", pii_policy={})
    _run(spark, source, tmp_path / "c")
    other = spark.readStream.schema("id long").parquet(str(source))
    with pytest.raises(ValueError, match="does not match"):
        StreamingProfiler({"t": other}, checkpoint_dir=tmp_path / "c")


def test_stream_cli_end_to_end(spark: SparkSession, tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "profile.json"
    _write_splits(spark.createDataFrame(_rows(40), SCHEMA), source, 2)
    arguments = ["stream", "--table", f"t={source}", "--format", "parquet", "--checkpoint", str(tmp_path / "c")]
    assert main([*arguments, "--trigger", "availableNow", "-o", str(output)]) == 0
    profile = load_profile(json.loads(output.read_text(encoding="utf-8")), strict=True)
    assert profile.engine.name == "pyspark-streaming"
    assert profile.tables[0].row_count == 40
    assert profile.source is not None and profile.source.format == "parquet"
    assert main([*arguments, "-o", str(output)]) == 0
    assert load_profile(output.read_text(encoding="utf-8")).tables[0].row_count == 40


def test_stream_cli_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "missing"
    arguments = ["stream", "--table", f"t={missing}", "--checkpoint", str(tmp_path / "c"), "-o", str(tmp_path / "p")]
    assert main(arguments) == 1
    assert "dbldatagen-profile stream:" in capsys.readouterr().err
