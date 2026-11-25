"""Tests for incremental profiling with Spark Structured Streaming."""

import json
import math
import os
import stat
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import jsonschema
import pyspark
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
from dbldatagen_profiler.engines.streaming import checkpoint as checkpoint_module
from dbldatagen_profiler.engines.streaming import profiler as profiler_module
from dbldatagen_profiler.engines.streaming.checkpoint import (
    Checkpoint,
    add_batch,
    decode_checkpoint,
    encode_checkpoint,
    read_checkpoint,
)
from dbldatagen_profiler.engines.streaming.profiler import PII_WARMUP_ROWS, STATE_FILE
from dbldatagen_profiler.engines.streaming.state import ColumnState, analyze_sample
from dbldatagen_profiler.model import (
    EvidenceMethod,
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


def _write_files(frames: list[DataFrame], path: Path) -> None:
    """Writes each frame as one parquet file, with modification times in list order after any existing files."""
    offset = len(list(path.glob("*.parquet"))) if path.exists() else 0
    for index, frame in enumerate(frames):
        before = set(path.glob("*.parquet")) if path.exists() else set()
        frame.coalesce(1).write.mode("append").parquet(str(path))
        moment = 1_700_000_000 + offset + index
        for file in set(path.glob("*.parquet")) - before:
            os.utime(file, (moment, moment))


def _write_splits(df: DataFrame, path: Path, splits: int) -> None:
    """Writes `df` as `splits` parquet files, interleaving rows by id."""
    _write_files([df.where(F.col("id") % splits == split) for split in range(splits)], path)


def _stream(spark: SparkSession, path: Path, schema: T.StructType, max_files: int = 1) -> DataFrame:
    return spark.readStream.schema(schema).option("maxFilesPerTrigger", max_files).parquet(str(path))


def _run(
    spark: SparkSession,
    path: Path,
    checkpoint: Path,
    *,
    privacy: PrivacyPolicy | None = None,
    pii_warmup_rows: int = PII_WARMUP_ROWS,
    max_files: int = 1,
) -> Profile:
    profiler = StreamingProfiler(
        {"t": _stream(spark, path, spark.read.parquet(str(path)).schema, max_files)},
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
    # The streaming state does not retain cross-column moments, so dependencies
    # are intentionally outside the documented batch/stream equivalence set.
    for table in data["tables"]:
        table.pop("dependencies")
    return data


def _assert_equivalent(batch: object, stream: object, path: str = "") -> None:
    """Asserts exact equality except for float moments (rounding) and distinct counts (sketch bound)."""
    # Streaming checkpoints do not retain a mergeable percentile sketch or exact value counts.
    # Quantiles and histograms are batch-profiler fields; all shared statistics remain equivalent.
    if path.endswith(("numeric.quantiles", "temporal.quantiles", "numeric.histogram")):
        assert stream is None
        return
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


def _distinct_counts(data: object) -> list[object]:
    if isinstance(data, dict):
        own = [data["distinct_count"]] if "distinct_count" in data else []
        return own + [
            count for key, value in data.items() if key != "distinct_count" for count in _distinct_counts(value)
        ]
    if isinstance(data, list):
        return [count for item in data for count in _distinct_counts(item)]
    return []


@pytest.mark.parametrize("max_files", [10, 4, 1])
def test_stream_matches_batch_profile(spark: SparkSession, tmp_path: Path, max_files: int) -> None:
    source = tmp_path / "source"
    _write_splits(spark.createDataFrame(_rows(), SCHEMA), source, 10)
    batch = profile_dataframe("t", spark.read.parquet(str(source)))
    stream = _run(spark, source, tmp_path / "checkpoint", max_files=max_files)
    checkpoint = read_checkpoint(tmp_path / "checkpoint" / "t" / STATE_FILE)
    assert checkpoint.batches == ((0, math.ceil(10 / max_files) - 1),)
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
        candidate = snapshot(state)
        _assert_equivalent(reference, candidate)
        assert _distinct_counts(candidate) == _distinct_counts(reference)
    assert [key for key, _ in merge_states(a, b).sample] == [key for key, _ in merge_states(b, a).sample]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_distinct_estimates_are_within_the_sketch_bound(spark: SparkSession, seed: int) -> None:
    values = spark.range(200_000).select(
        F.xxhash64(F.col("id"), F.lit(seed)).alias("hashed"),
        F.concat(F.lit("v"), F.col("id").cast("string"), F.lit(seed)).alias("text"),
    )
    parts = [
        state_for_dataframe(values.where(F.pmod(F.col("hashed"), F.lit(3)) == k), collect_sample=False)
        for k in range(3)
    ]
    merged = merge_states(merge_states(parts[0], parts[1]), parts[2])
    exact = values.agg(F.countDistinct("hashed"), F.countDistinct("text")).first()
    assert exact is not None
    for column, truth in zip(merged.columns, exact, strict=True):
        assert truth > 199_000
        assert abs(column.distinct - truth) <= 3 * DISTINCT_RELATIVE_ERROR * truth
    other_order = merge_states(parts[2], merge_states(parts[1], parts[0]))
    assert [column.distinct for column in other_order.columns] == [column.distinct for column in merged.columns]


@pytest.mark.parametrize("persisted", [False, True])
def test_restart_after_crash_gives_same_profile(
    spark: SparkSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, persisted: bool
) -> None:
    source = tmp_path / "source"
    _write_splits(spark.createDataFrame(_rows(60), SCHEMA), source, 3)
    expected = _run(spark, source, tmp_path / "reference")
    write = profiler_module.write_checkpoint

    def crash(path: Path, checkpoint: Checkpoint) -> None:
        if persisted:
            write(path, checkpoint)
        if checkpoint.batches[-1][1] == 1:
            raise RuntimeError("simulated crash")
        if not persisted:
            write(path, checkpoint)

    monkeypatch.setattr(profiler_module, "write_checkpoint", crash)
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


def _contacts(spark: SparkSession, path: Path) -> None:
    """Writes 10 files of 20 rows: 100 email rows, then 100 rows of words."""
    rows = [(i, f"person{i}@example.com" if i < 100 else f"hello world {i % 5}") for i in range(200)]
    df = spark.createDataFrame(rows, "id int, contact string")
    _write_files([df.where((F.col("id") >= start) & (F.col("id") < start + 20)) for start in range(0, 200, 20)], path)


def test_pii_labels_freeze_after_warmup(spark: SparkSession, tmp_path: Path) -> None:
    source = tmp_path / "source"
    _contacts(spark, source)
    frozen = _run(spark, source, tmp_path / "frozen", pii_warmup_rows=100)
    unfrozen = _run(spark, source, tmp_path / "unfrozen")
    frozen_label, unfrozen_label = (profile.tables[0].columns[1].semantic for profile in (frozen, unfrozen))
    assert frozen_label is not None and frozen_label.pii_class == PiiClass.EMAIL
    assert unfrozen_label is None
    assert frozen.tables[0].row_count == unfrozen.tables[0].row_count == 200
    text = (tmp_path / "frozen" / "t" / STATE_FILE).read_text(encoding="utf-8")
    assert "person0@example.com" not in text
    assert json.loads(text)["sample"] == []


def test_frozen_pii_labels_do_not_depend_on_batch_boundaries(spark: SparkSession, tmp_path: Path) -> None:
    source = tmp_path / "source"
    _contacts(spark, source)
    frozen = []
    for max_files in (10, 4, 1):
        directory = tmp_path / f"checkpoint{max_files}"
        _run(spark, source, directory, pii_warmup_rows=110, max_files=max_files)
        frozen.append(json.loads((directory / "t" / STATE_FILE).read_text(encoding="utf-8"))["pii"])
    assert frozen[0] is not None
    assert frozen[0]["semantics"]["contact"]["pii_class"] == "email"
    assert frozen[0]["semantics"]["contact"]["evidence"][-1]["match_rate"] == pytest.approx(100 / 110)
    assert frozen[1] == frozen[0]
    assert frozen[2] == frozen[0]


def _write_earlier_file(frame: DataFrame, path: Path) -> Path:
    """Writes `frame` as one parquet file whose name and modification time sort before every existing file."""
    before = set(path.glob("*.parquet"))
    frame.coalesce(1).write.mode("append").parquet(str(path))
    (written,) = set(path.glob("*.parquet")) - before
    earlier = written.rename(path / "a0.parquet")
    moment = min(file.stat().st_mtime for file in before) - 1
    os.utime(earlier, (moment, moment))
    assert earlier.name < min(file.name for file in before)
    return earlier


@pytest.mark.parametrize(("warmup", "frozen_first"), [(40, True), (45, False)])
def test_warmup_set_is_arrival_order(spark: SparkSession, tmp_path: Path, warmup: int, frozen_first: bool) -> None:
    source, checkpoint = tmp_path / "source", tmp_path / "checkpoint"
    rows = [(i, f"person{i}@example.com") for i in range(40)] + [(i, f"hello world {i % 5}") for i in range(40, 60)]
    df = spark.createDataFrame(rows, "id int, contact string")
    _write_files([df.where("id < 20"), df.where("id >= 20 and id < 40")], source)
    first = _run(spark, source, checkpoint, pii_warmup_rows=warmup, max_files=10)
    state = json.loads((checkpoint / "t" / STATE_FILE).read_text(encoding="utf-8"))
    assert (state["pii"] is not None) == frozen_first
    _write_earlier_file(df.where("id >= 40"), source)
    final = _run(spark, source, checkpoint, pii_warmup_rows=warmup, max_files=10)
    state = json.loads((checkpoint / "t" / STATE_FILE).read_text(encoding="utf-8"))
    assert state["batches"] == [[0, 1]] and state["pii"] is not None and state["sample"] == []
    label = final.tables[0].columns[1].semantic
    assert label is not None and label.pii_class == PiiClass.EMAIL
    if frozen_first:
        assert label == first.tables[0].columns[1].semantic
    assert state["pii"]["semantics"]["contact"]["evidence"][-1]["match_rate"] == pytest.approx(min(1.0, 40 / warmup))
    assert first.tables[0].row_count == 40 and final.tables[0].row_count == 60
    contact = final.tables[0].columns[1]
    assert contact.string is not None and contact.string.length is not None
    assert contact.string.length.min == len("hello world 0")


def test_batch_ids_are_merged_once_in_any_order(spark: SparkSession, tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_splits(spark.createDataFrame(_rows(30), SCHEMA), source, 1)
    schema = spark.read.parquet(str(source)).schema
    df = spark.createDataFrame(_rows(30), SCHEMA)
    profiler = StreamingProfiler({"t": _stream(spark, source, schema)}, checkpoint_dir=tmp_path / "c")
    for batch_id, part in ((2, "id < 10"), (0, "id >= 10 and id < 20"), (5, "id >= 20"), (2, "id < 10")):
        profiler._process("t", df.where(part), batch_id)
    assert profiler.profile().tables[0].row_count == 30
    assert read_checkpoint(tmp_path / "c" / "t" / STATE_FILE).batches == ((0, 0), (2, 2), (5, 5))
    profiler._process("t", df.where("id < 0"), 1)
    restarted = StreamingProfiler({"t": _stream(spark, source, schema)}, checkpoint_dir=tmp_path / "c")
    restarted._process("t", df.where("id >= 20"), 5)
    restarted._process("t", df.where("id < 10"), 1)
    assert restarted.profile().tables[0].row_count == 30
    assert read_checkpoint(tmp_path / "c" / "t" / STATE_FILE).batches == ((0, 2), (5, 5))


def test_add_batch_joins_ranges() -> None:
    assert add_batch((), 3) == ((3, 3),)
    assert add_batch(((0, 1), (3, 3)), 2) == ((0, 3),)
    assert add_batch(((0, 1),), 5) == ((0, 1), (5, 5))
    assert add_batch(((0, 4),), 2) == ((0, 4),)


def test_streaming_requires_pyspark_3_5(spark: SparkSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    _write_splits(spark.createDataFrame([(1, "a")], "id int, label string"), source, 1)
    stream = _stream(spark, source, spark.read.parquet(str(source)).schema)
    monkeypatch.setattr(pyspark, "__version__", "3.4.1")
    with pytest.raises(RuntimeError, match=r"StreamingProfiler requires pyspark >= 3.5; found 3.4.1"):
        StreamingProfiler({"t": stream}, checkpoint_dir=tmp_path / "c")
    with pytest.raises(RuntimeError, match=r"found 3\.4\.1"):
        state_for_dataframe(spark.read.parquet(str(source)))
    assert profile_dataframe("t", spark.read.parquet(str(source))).tables[0].row_count == 1


def test_checkpoint_write_syncs_directory(spark: SparkSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synced: list[bool] = []
    real_fsync = os.fsync

    def record(descriptor: int) -> None:
        synced.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
        real_fsync(descriptor)

    monkeypatch.setattr(checkpoint_module.os, "fsync", record)
    state = empty_state(spark.createDataFrame([], "id int").schema)
    checkpoint_module.write_checkpoint(tmp_path / "t" / STATE_FILE, Checkpoint(((0, 0),), state))
    assert synced == [False, True]
    assert not (tmp_path / "t" / f"{STATE_FILE}.tmp").exists()


def test_checkpoint_json_round_trips(spark: SparkSession) -> None:
    schema = SCHEMA + ", bin binary, ntz timestamp_ntz, span interval day to second, f float"
    rows = [
        (*row, bytes([i % 7, 0]), datetime(2022, 2, 2, 1, 2, 3, i), timedelta(days=i, microseconds=7), float("nan"))
        for i, row in enumerate(_rows(25))
    ]
    state = state_for_dataframe(spark.createDataFrame(rows, schema))
    analysis = analyze_sample(state)
    data = json.loads(json.dumps(encode_checkpoint(Checkpoint(((0, 4),), state, analysis))))
    decoded = decode_checkpoint(data)
    assert decoded.batches == ((0, 4),) and decoded.pii == analysis
    assert decoded.state.schema == state.schema and len(decoded.state.sample) == 25
    assert [key for key, _ in decoded.state.sample] == [key for key, _ in state.sample]
    assert [row[:-1] for _, row in decoded.state.sample] == [row[:-1] for _, row in state.sample]
    assert encode_checkpoint(decoded) == data
    float_column = decoded.state.columns[-1]
    assert isinstance(float_column, ColumnState) and math.isnan(float_column.max)
    assert decoded.state.columns[:-1] == state.columns[:-1]
    assert data["columns"][1]["min"] == str(-(2**63)) and data["columns"][2]["max"] == str(DECIMAL38_MAX)


def test_checkpoint_written_before_evidence_enum_decodes() -> None:
    path = Path(__file__).parent / "fixtures" / "checkpoint_v1_pre_evidence_enum.json"
    checkpoint = read_checkpoint(path)
    assert checkpoint.pii is not None
    semantics, _ = checkpoint.pii
    assert {name: [item.method for item in semantic.evidence] for name, semantic in semantics.items()} == {
        "email": [EvidenceMethod.COLUMN_NAME, EvidenceMethod.EMAIL],
        "account_id": [EvidenceMethod.COLUMN_NAME, EvidenceMethod.SAMPLE_UNIQUENESS],
        "birth_date": [EvidenceMethod.COLUMN_NAME, EvidenceMethod.PLAUSIBLE_AGE],
    }
    assert encode_checkpoint(checkpoint)["pii"] == json.loads(path.read_text(encoding="utf-8"))["pii"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "not valid JSON"),
        ('{"version": 99}', "unsupported version 99"),
        ("[1, 2]", "unsupported version None"),
        ('{"version": 1, "batches": []}', "is invalid"),
        ("__import__('os').system('touch pwned')", "not valid JSON"),
    ],
)
def test_tampered_checkpoint_fails_cleanly(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / STATE_FILE
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_checkpoint(path)


def test_tampered_checkpoint_values_fail_cleanly(spark: SparkSession, tmp_path: Path) -> None:
    state = state_for_dataframe(spark.createDataFrame(_rows(5), SCHEMA))
    data = encode_checkpoint(Checkpoint(((0, 0),), state))
    for tamper in (
        lambda d: d["columns"][0].update(count="5"),
        lambda d: d["columns"][4].update(sketch="!!!"),
        lambda d: d["columns"][1].update(min=12),
        lambda d: d["columns"].pop(),
        lambda d: d["sample"][0]["row"].update(dt="not a date"),
        lambda d: d.update(schema={"type": "struct"}),
    ):
        copy = json.loads(json.dumps(data))
        tamper(copy)
        path = tmp_path / STATE_FILE
        path.write_text(json.dumps(copy), encoding="utf-8")
        with pytest.raises(ValueError, match="is invalid"):
            read_checkpoint(path)


def test_snapshot_below_min_value_count_has_no_protected_literals(spark: SparkSession) -> None:
    df = spark.createDataFrame([("gold", 1), ("gold", 2), ("silver", 3)], "tier string, n int")
    profile = profile_states({"t": state_for_dataframe(df)}, pii={})
    table = profile.tables[0]
    assert table.row_count < profile.privacy.min_value_count
    assert table.check_constraints is None
    for column in table.columns:
        assert column.string is None or (column.string.top_values is None and column.string.patterns is None)
    load_profile(profile.model_dump(mode="json"), strict=True)


def test_empty_batches_and_all_null_columns_match_batch(spark: SparkSession, tmp_path: Path) -> None:
    schema = "id int, n long, d double, s string, b boolean, ts timestamp, arr array<int>, st struct<x:int>"
    df = spark.createDataFrame([(i, None, None, None, None, None, None, None) for i in range(12)], schema)
    states = [state_for_dataframe(df.where("id < 0")), state_for_dataframe(df), state_for_dataframe(df.where("id < 0"))]
    merged = merge_states(merge_states(states[0], states[1]), states[2])
    stream = profile_states({"t": merged}, pii={"t": analyze_sample(merged)})
    _assert_equivalent(_comparable(profile_dataframe("t", df)), _comparable(stream))
    empty = profile_states({"t": states[0]}, pii={"t": analyze_sample(states[0])})
    _assert_equivalent(_comparable(profile_dataframe("t", df.where("id < 0"))), _comparable(empty))
    source = tmp_path / "source"
    _write_splits(df, source, 1)
    profiler = StreamingProfiler(
        {"t": _stream(spark, source, spark.read.parquet(str(source)).schema)}, checkpoint_dir=tmp_path / "c"
    )
    profiler._process("t", spark.read.parquet(str(source)).where("id < 0"), 0)
    assert profiler.profile().tables[0].row_count == 0
    profiler._process("t", spark.read.parquet(str(source)), 1)
    assert profiler.profile().tables[0].columns[1].null_fraction == 1.0


def test_stream_matches_batch_in_non_utc_session(spark: SparkSession, tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_splits(spark.createDataFrame(_rows(40), SCHEMA), source, 2)
    previous = spark.conf.get("spark.sql.session.timeZone")
    spark.conf.set("spark.sql.session.timeZone", "America/Los_Angeles")
    try:
        stream = _run(spark, source, tmp_path / "checkpoint")
        batch = profile_dataframe("t", spark.read.parquet(str(source)))
        assert spark.conf.get("spark.sql.session.timeZone") == "America/Los_Angeles"
    finally:
        spark.conf.set("spark.sql.session.timeZone", previous or "UTC")
    _assert_equivalent(_comparable(batch), _comparable(stream))
    ts = next(column for column in stream.tables[0].columns if column.name == "ts")
    assert ts.temporal is not None and ts.temporal.min == "2021-01-01T00:00:00Z"


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
