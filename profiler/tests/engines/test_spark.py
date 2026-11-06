"""Tests for the baseline PySpark profiling engine."""

import time
from datetime import date, datetime, timezone
from decimal import Decimal

import jsonschema
import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql import types as T

from dbldatagen_profiler.engines.spark import profile_dataframe, profile_tables
from dbldatagen_profiler.engines.spark.engine import _logical_type
from dbldatagen_profiler.model import LogicalType, PrivacyPolicy, ValueTreatment, profile_json_schema


pytestmark = [pytest.mark.spark, pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")]


@pytest.mark.parametrize(
    ("spark_type", "logical_type"),
    [
        (T.NullType(), LogicalType.NULL),
        (T.BooleanType(), LogicalType.BOOLEAN),
        (T.ByteType(), LogicalType.INT8),
        (T.ShortType(), LogicalType.INT16),
        (T.IntegerType(), LogicalType.INT32),
        (T.LongType(), LogicalType.INT64),
        (T.FloatType(), LogicalType.FLOAT32),
        (T.DoubleType(), LogicalType.FLOAT64),
        (T.DecimalType(10, 2), LogicalType.DECIMAL),
        (T.StringType(), LogicalType.STRING),
        (T.CharType(4), LogicalType.STRING),
        (T.VarcharType(4), LogicalType.STRING),
        (T.BinaryType(), LogicalType.BINARY),
        (T.DateType(), LogicalType.DATE),
        (T.TimeType(), LogicalType.TIME),
        (T.TimestampType(), LogicalType.TIMESTAMP),
        (T.TimestampNTZType(), LogicalType.TIMESTAMP_NTZ),
        (T.CalendarIntervalType(), LogicalType.INTERVAL),
        (T.DayTimeIntervalType(), LogicalType.INTERVAL),
        (T.YearMonthIntervalType(), LogicalType.INTERVAL),
        (T.StructType([T.StructField("x", T.IntegerType())]), LogicalType.STRUCT),
        (T.ArrayType(T.IntegerType()), LogicalType.ARRAY),
        (T.MapType(T.StringType(), T.IntegerType()), LogicalType.MAP),
        (T.VariantType(), LogicalType.VARIANT),
        (T.GeometryType(4326), LogicalType.VARIANT),
        (T.GeographyType(4326), LogicalType.VARIANT),
    ],
)
def test_maps_spark_types(spark_type: T.DataType, logical_type: LogicalType) -> None:
    assert _logical_type(spark_type) == logical_type


def test_maps_every_type_family_and_validates_schema(spark: SparkSession) -> None:
    schema = T.StructType(
        [
            T.StructField("void", T.NullType()),
            T.StructField("boolean", T.BooleanType()),
            T.StructField("byte", T.ByteType()),
            T.StructField("short", T.ShortType()),
            T.StructField("integer", T.IntegerType()),
            T.StructField("long", T.LongType()),
            T.StructField("float", T.FloatType()),
            T.StructField("double", T.DoubleType()),
            T.StructField("decimal", T.DecimalType(10, 2)),
            T.StructField("string", T.StringType()),
            T.StructField("binary", T.BinaryType()),
            T.StructField("date", T.DateType()),
            T.StructField("timestamp", T.TimestampType()),
            T.StructField("interval", T.DayTimeIntervalType()),
            T.StructField("struct", T.StructType([T.StructField("x", T.IntegerType())])),
            T.StructField("array", T.ArrayType(T.StringType())),
            T.StructField("map", T.MapType(T.StringType(), T.LongType())),
        ]
    )
    row = (
        None,
        True,
        1,
        2,
        3,
        4,
        1.5,
        2.5,
        Decimal("3.25"),
        "abc",
        b"x",
        date(2026, 1, 2),
        datetime(2026, 1, 2, 3, 4, 5),
        None,
        Row(x=1),
        ["a", None],
        {"a": 1},
    )
    profile = profile_dataframe("types", spark.createDataFrame([row], schema))
    assert [column.logical_type.value for column in profile.tables[0].columns] == [
        "null",
        "boolean",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "decimal",
        "string",
        "binary",
        "date",
        "timestamp",
        "interval",
        "struct",
        "array",
        "map",
    ]
    jsonschema.validate(profile.model_dump(mode="json", exclude_none=True), profile_json_schema())


def test_profiles_flat_statistics(spark: SparkSession) -> None:
    schema = "number long, text string, moment timestamp, flag boolean, all_null int, constant double"
    rows = [
        (1, "a", datetime(2026, 1, 1, tzinfo=timezone.utc), True, None, 2.0),
        (2, "abcd", datetime(2026, 1, 3, tzinfo=timezone.utc), False, None, 2.0),
        (None, None, None, None, None, None),
    ]
    table = profile_dataframe("stats", spark.createDataFrame(rows, schema)).tables[0]
    columns = {column.name: column for column in table.columns}
    number = columns["number"]
    text = columns["text"]
    moment = columns["moment"]
    flag = columns["flag"]
    assert number.distinct_count is not None and number.numeric is not None
    assert text.string is not None and text.string.length is not None
    assert moment.temporal is not None and flag.boolean is not None
    assert columns["all_null"].numeric is not None and columns["constant"].numeric is not None
    assert table.row_count == 3
    assert number.null_fraction == pytest.approx(1 / 3)
    assert number.distinct_count.value == 2
    assert number.numeric.min == "1"
    assert number.numeric.max == "2"
    assert number.numeric.mean == pytest.approx(1.5)
    assert number.numeric.stddev == pytest.approx(2**-0.5)
    assert text.string.length.model_dump(exclude_none=True) == {"min": 1, "max": 4, "mean": 2.5}
    assert moment.temporal.min == "2026-01-01T00:00:00Z"
    assert flag.boolean.true_fraction == 0.5
    assert columns["all_null"].numeric.model_dump(exclude_none=True) == {}
    assert columns["constant"].numeric.stddev == 0.0


def test_profiles_empty_table(spark: SparkSession) -> None:
    table = profile_dataframe("empty", spark.createDataFrame([], "value long")).tables[0]
    assert table.row_count == 0
    assert table.columns[0].null_fraction == 0.0
    assert table.columns[0].distinct_count is not None
    assert table.columns[0].distinct_count.value == 0


def test_timestamp_extremes_are_utc_when_host_is_not(spark: SparkSession, monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        with monkeypatch.context() as patch:
            patch.setenv("TZ", "America/New_York")
            time.tzset()
            frame = spark.sql("SELECT TIMESTAMP '2026-01-02 03:04:05' AS moment")
            column = profile_dataframe("timestamps", frame).tables[0].columns[0]
    finally:
        time.tzset()
    assert column.temporal is not None
    assert column.temporal.min == "2026-01-02T03:04:05Z"
    assert column.temporal.max == "2026-01-02T03:04:05Z"


def test_profile_restores_session_time_zone(spark: SparkSession) -> None:
    key = "spark.sql.session.timeZone"
    previous = spark.conf.get(key, None)
    spark.conf.set(key, "America/Los_Angeles")
    try:
        frame = spark.sql("SELECT TIMESTAMP '2026-01-02 03:04:05' AS moment")
        column = profile_dataframe("timestamps", frame).tables[0].columns[0]
        assert spark.conf.get(key) == "America/Los_Angeles"
        assert column.temporal is not None
        assert column.temporal.min == "2026-01-02T11:04:05Z"
        assert column.temporal.max == "2026-01-02T11:04:05Z"
    finally:
        if previous is None:
            spark.conf.unset(key)
        else:
            spark.conf.set(key, previous)


def test_non_finite_float_extremes_are_omitted(spark: SparkSession) -> None:
    column = (
        profile_dataframe("non_finite", spark.createDataFrame([(float("inf"),)], "value double")).tables[0].columns[0]
    )
    assert column.numeric is not None
    assert column.numeric.min is None
    assert column.numeric.max is None
    assert "min" not in column.numeric.model_dump(exclude_none=True)
    assert "max" not in column.numeric.model_dump(exclude_none=True)


def test_lossless_decimal_and_int64_extremes(spark: SparkSession) -> None:
    schema = T.StructType([T.StructField("integer", T.LongType()), T.StructField("decimal", T.DecimalType(38, 18))])
    rows = [
        (-(2**63), Decimal("-99999999999999999999.999999999999999999")),
        (2**63 - 1, Decimal("99999999999999999999.999999999999999999")),
    ]
    columns = profile_dataframe("lossless", spark.createDataFrame(rows, schema)).tables[0].columns
    assert columns[0].numeric is not None and columns[1].numeric is not None
    assert columns[0].numeric.min == "-9223372036854775808"
    assert columns[0].numeric.max == "9223372036854775807"
    assert columns[1].numeric.min == "-99999999999999999999.999999999999999999"
    assert columns[1].numeric.max == "99999999999999999999.999999999999999999"


def test_large_numeric_aggregates_do_not_overflow_in_ansi_mode(spark: SparkSession) -> None:
    key = "spark.sql.ansi.enabled"
    previous = spark.conf.get(key, None)
    spark.conf.set(key, "true")
    try:
        frame = spark.range(100_000).selectExpr(
            "IF(id % 2 = 0, 9999999999999999999999999999.9999999999, "
            "8999999999999999999999999999.9999999999)::DECIMAL(38, 10) AS scaled",
            "IF(id % 2 = 0, 99999999999999999999999999999999999999, "
            "89999999999999999999999999999999999999)::DECIMAL(38, 0) AS unscaled",
            f"IF(id % 2 = 0, {-(2**63)}, {2**63 - 1})::BIGINT AS integer",
        )

        profile = profile_dataframe("large_numeric", frame)
        columns = {column.name: column for column in profile.tables[0].columns}
        scaled = columns["scaled"].numeric
        unscaled = columns["unscaled"].numeric
        integer = columns["integer"].numeric

        assert scaled is not None and unscaled is not None and integer is not None
        assert scaled.min == "8999999999999999999999999999.9999999999"
        assert scaled.max == "9999999999999999999999999999.9999999999"
        assert scaled.mean == pytest.approx(9.5e27)
        assert scaled.stddev == pytest.approx(5e26, rel=1e-4)
        assert unscaled.min == "89999999999999999999999999999999999999"
        assert unscaled.max == "99999999999999999999999999999999999999"
        assert unscaled.mean == pytest.approx(9.5e37)
        assert unscaled.stddev == pytest.approx(5e36, rel=1e-4)
        assert integer.min == "-9223372036854775808"
        assert integer.max == "9223372036854775807"
        assert integer.mean == pytest.approx(0.0, abs=1024.0)
        assert integer.stddev == pytest.approx(2**63, rel=1e-4)
    finally:
        if previous is None:
            spark.conf.unset(key)
        else:
            spark.conf.set(key, previous)


def test_suppressed_extremes_produce_valid_profile(spark: SparkSession) -> None:
    policy = PrivacyPolicy(extremes=ValueTreatment.SUPPRESSED)
    profile = profile_dataframe(
        "private", spark.createDataFrame([(1, date(2026, 1, 1))], "n int, d date"), privacy=policy
    )
    assert profile.tables[0].columns[0].numeric is not None
    assert profile.tables[0].columns[1].temporal is not None
    assert profile.tables[0].columns[0].numeric.min is None
    assert profile.tables[0].columns[1].temporal.min is None


def test_nested_columns_have_value_counts(spark: SparkSession) -> None:
    schema = "record struct<x:int>, items array<struct<name:string>>, lookup map<string,long>"
    rows = [(Row(x=1), [Row(name="a"), Row(name=None)], {"a": 1}), (None, [], None)]
    columns = profile_dataframe("nested", spark.createDataFrame(rows, schema)).tables[0].columns
    assert columns[0].children is not None and columns[1].children is not None and columns[2].children is not None
    assert columns[1].children[0].children is not None
    assert columns[0].children[0].value_count == 2
    assert columns[1].children[0].value_count == 2
    assert columns[1].children[0].children[0].value_count == 2
    assert columns[2].children[0].value_count == 1
    assert columns[2].children[1].value_count == 1


def test_profiles_multiple_tables(spark: SparkSession) -> None:
    profile = profile_tables(
        {"first": spark.createDataFrame([(1,)], "id int"), "second": spark.createDataFrame([(2,)], "id int")}
    )
    assert [table.name for table in profile.tables] == ["first", "second"]
