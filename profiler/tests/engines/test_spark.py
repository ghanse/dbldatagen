"""Tests for the baseline PySpark profiling engine."""

import time
from datetime import date, datetime, timezone
from decimal import Decimal

import jsonschema
import pytest
from pyspark.sql import Column, DataFrame, Row, SparkSession
from pyspark.sql import types as T

from dbldatagen_profiler.engines.spark import engine as spark_engine
from dbldatagen_profiler.engines.spark import profile_dataframe, profile_tables
from dbldatagen_profiler.engines.spark.engine import (
    _comparison_pairs,
    _histogram,
    _logical_type,
    _numeric_stats,
    _pearson_pairs,
)
from dbldatagen_profiler.model import (
    ComparisonConstraint,
    LogicalType,
    PrivacyPolicy,
    ValueTreatment,
    profile_json_schema,
)


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
    assert number.numeric.quantiles is None
    assert text.string.length.model_dump(exclude_none=True) == {"min": 1, "max": 4, "mean": 2.5}
    assert moment.temporal.min == "2026-01-01T00:00:00Z"
    assert moment.temporal.quantiles is None
    assert flag.boolean.true_fraction == 0.5
    assert columns["all_null"].numeric.model_dump(exclude_none=True) == {}
    assert columns["constant"].numeric.stddev == 0.0


def test_profiles_pearson_dependencies_for_numeric_and_temporal_pairs(spark: SparkSession) -> None:
    rows = [
        (
            float(value),
            value * 2.0 + (value % 3),
            date(2026, 1, value),
            datetime(2026, 1, value, tzinfo=timezone.utc),
        )
        for value in range(1, 21)
        for _ in range(2)
    ]
    table = profile_dataframe(
        "dependencies",
        spark.createDataFrame(rows, "x double, y double, day date, moment timestamp"),
        detect_pii=False,
    ).tables[0]

    assert table.dependencies is not None
    actual = {(dependency.columns, dependency.method.value): dependency.value for dependency in table.dependencies}
    assert actual[(("x", "y"), "pearson")] == pytest.approx(0.9968, rel=1e-3)
    assert actual[(("day", "moment"), "pearson")] == pytest.approx(1.0)


def test_pearson_dependencies_require_privacy_min_value_count_finite_pairs(spark: SparkSession) -> None:
    frame = spark.createDataFrame(
        [(1.0, 2.0), (2.0, 4.0), (float("inf"), 6.0), (float("nan"), 8.0), (None, 10.0)],
        "x double, y double",
    )

    table = profile_dataframe("private", frame, detect_pii=False).tables[0]

    assert table.dependencies is None
    assert "dependencies" not in table.model_dump(mode="json", exclude_none=True)


def test_pearson_dependencies_honor_custom_privacy_support(spark: SparkSession) -> None:
    frame = spark.createDataFrame([(float(x), float(x * x)) for x in range(3) for _ in range(2)], "x double, y double")

    table = profile_dataframe("supported", frame, privacy=PrivacyPolicy(min_value_count=3), detect_pii=False).tables[0]

    assert table.dependencies is not None
    assert table.dependencies[0].columns == ("x", "y")


def test_pearson_dependencies_use_a_separate_aggregate(spark: SparkSession, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    collect = spark_engine._collect_aggregate

    def recording_collect(df: DataFrame, expressions: list[Column]) -> dict[str, object]:
        calls.append([str(expression) for expression in expressions])
        return collect(df, expressions)

    monkeypatch.setattr(spark_engine, "_collect_aggregate", recording_collect)
    frame = spark.createDataFrame([(float(x), float(x * x)) for x in range(20) for _ in range(2)], "x double, y double")

    profile_dataframe("separate", frame, detect_pii=False)

    assert len(calls) == 2
    assert all("p0_" not in expression for expression in calls[0])
    assert len(calls[1]) == 5
    assert sum("p0_" in expression for expression in calls[1]) == 2
    assert sum("c0_" in expression for expression in calls[1]) == 3


def test_pearson_dependencies_keep_exact_and_skip_key_like_pairs(spark: SparkSession) -> None:
    rows = [(value, index, value * 3, value * 6) for index, value in enumerate(list(range(10)) * 2)]
    frame = spark.createDataFrame(rows, "order_id long, unique_key long, exact long, exact_copy long")

    table = profile_dataframe("guarded", frame, detect_pii=False).tables[0]

    assert table.dependencies is not None
    assert [dependency.columns for dependency in table.dependencies] == [("exact", "exact_copy")]
    assert table.dependencies[0].value == pytest.approx(1.0)


def test_profiles_strict_and_nonstrict_comparison_constraints(spark: SparkSession) -> None:
    rows = [(value, value + 1, value) for value in range(20)]
    frame = spark.createDataFrame(rows, "left long, strict_right long, weak_right long")

    table = profile_dataframe("ordered", frame, detect_pii=False).tables[0]

    assert table.check_constraints is not None
    assert [constraint.model_dump() for constraint in table.check_constraints] == [
        {
            "satisfied_fraction": 1.0,
            "kind": "comparison",
            "left": "left",
            "operator": "<",
            "right": "strict_right",
        },
        {
            "satisfied_fraction": 1.0,
            "kind": "comparison",
            "left": "left",
            "operator": "<=",
            "right": "weak_right",
        },
    ]


def test_profiles_reverse_strict_and_nonstrict_comparison_constraints(spark: SparkSession) -> None:
    rows = [(value + 1, value + value % 2, value) for value in range(20)]
    frame = spark.createDataFrame(rows, "strict_left long, weak_left long, right long")

    table = profile_dataframe("reverse_ordered", frame, detect_pii=False).tables[0]

    assert table.check_constraints is not None
    actual = {
        (constraint.left, constraint.operator, constraint.right)
        for constraint in table.check_constraints
        if isinstance(constraint, ComparisonConstraint)
    }
    assert ("right", "<", "strict_left") in actual
    assert ("right", "<=", "weak_left") in actual


def test_comparison_constraints_require_support_and_skip_violations_keys_and_cross_family(
    spark: SparkSession,
) -> None:
    rows = [
        (
            index,
            index + 1 if index < 9 else None,
            float(index + 2),
            index + 1 if index < 9 else -1,
            index + 3,
        )
        for index in range(10)
    ]
    frame = spark.createDataFrame(rows, "left long, low_support long, fractional double, violated long, order_id long")

    table = profile_dataframe("guarded", frame, detect_pii=False).tables[0]

    assert table.check_constraints is None


def test_comparison_constraints_exclude_nan_from_support(spark: SparkSession) -> None:
    rows = [(float(value), float(value + 1)) for value in range(10)] + [(float("nan"), 0.0)]
    frame = spark.createDataFrame(rows, "left double, right double")

    table = profile_dataframe("nan", frame, detect_pii=False).tables[0]

    assert table.check_constraints is not None
    constraint = table.check_constraints[0]
    assert isinstance(constraint, ComparisonConstraint)
    assert (constraint.left, constraint.operator, constraint.right) == ("left", "<", "right")


def test_comparison_pairs_require_matching_supported_temporal_kinds(spark: SparkSession) -> None:
    schema = T.StructType(
        [
            T.StructField("date_a", T.DateType()),
            T.StructField("date_b", T.DateType()),
            T.StructField("timestamp_a", T.TimestampType()),
            T.StructField("timestamp_b", T.TimestampType()),
            T.StructField("ntz_a", T.TimestampNTZType()),
            T.StructField("ntz_b", T.TimestampNTZType()),
            T.StructField("time_a", T.TimeType()),
            T.StructField("time_b", T.TimeType()),
        ]
    )

    pairs = _comparison_pairs(spark.createDataFrame([], schema))

    assert {(left, right) for left, right, _, _ in pairs} == {
        ("date_a", "date_b"),
        ("timestamp_a", "timestamp_b"),
        ("ntz_a", "ntz_b"),
    }


def test_comparison_constraints_honor_custom_privacy_support(spark: SparkSession) -> None:
    frame = spark.createDataFrame([(value, value + 1) for value in range(3)], "left long, right long")

    table = profile_dataframe("supported", frame, privacy=PrivacyPolicy(min_value_count=3), detect_pii=False).tables[0]

    assert table.check_constraints is not None
    assert isinstance(table.check_constraints[0], ComparisonConstraint)
    assert table.check_constraints[0].operator == "<"


def test_pearson_pair_selection_is_deterministic_and_bounded_for_wide_tables(spark: SparkSession) -> None:
    frame = spark.createDataFrame(
        [tuple(float(value + row % 3) for value in range(200)) for row in range(10)],
        ", ".join(f"c{index:03d} double" for index in range(200)),
    )

    pairs = _pearson_pairs(frame)

    assert len(pairs) == 64
    assert [(left, right) for left, right, _, _ in pairs[:3]] == [
        ("c000", "c001"),
        ("c001", "c002"),
        ("c002", "c003"),
    ]
    assert (pairs[-1][0], pairs[-1][1]) == ("c063", "c064")

    comparison_pairs = _comparison_pairs(frame)
    assert len(comparison_pairs) == 64
    assert [(left, right) for left, right, _, _ in comparison_pairs[:3]] == [
        ("c000", "c001"),
        ("c001", "c002"),
        ("c002", "c003"),
    ]
    assert (comparison_pairs[-1][0], comparison_pairs[-1][1]) == ("c063", "c064")


def test_profiles_empty_table(spark: SparkSession) -> None:
    table = profile_dataframe("empty", spark.createDataFrame([], "value long")).tables[0]
    assert table.row_count == 0
    assert table.columns[0].null_fraction == 0.0
    assert table.columns[0].distinct_count is not None
    assert table.columns[0].distinct_count.value == 0


CORE_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
FULL_GRID = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]


@pytest.mark.parametrize(
    ("rows", "grid"),
    [
        (60, CORE_GRID),
        (199, CORE_GRID),
        (200, [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]),
        (999, [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]),
        (1000, FULL_GRID),
    ],
)
def test_tail_quantiles_need_k_rows_beyond_them(spark: SparkSession, rows: int, grid: list[float]) -> None:
    frame = spark.range(rows).selectExpr("id * 1.5 AS number", "date_add(DATE '2026-01-01', CAST(id AS INT)) AS day")
    columns = profile_dataframe("quantiles", frame).tables[0].columns

    assert columns[0].numeric is not None and columns[1].temporal is not None
    assert [item.q for item in columns[0].numeric.quantiles or []] == grid
    assert [item.q for item in columns[1].temporal.quantiles or []] == grid


def test_low_cardinality_integers_get_k_anonymous_histogram(spark: SparkSession) -> None:
    counts = {0: 3, 1: 4, 2: 40, 3: 25, 4: 12, 6: 9, 9: 1}
    frame = spark.createDataFrame([(value,) for value, count in counts.items() for _ in range(count)], "n int")
    stats = profile_dataframe("histogram", frame).tables[0].columns[0].numeric

    assert stats is not None and stats.histogram is not None
    assert stats.quantiles is None
    assert stats.histogram.edges == [0, 3, 4, 6, 9]
    assert stats.histogram.counts == [47, 25, 12, 10]
    assert not stats.histogram.generalized_outer_edges


def test_high_cardinality_and_non_integer_columns_keep_quantiles(spark: SparkSession) -> None:
    frame = spark.range(1000).selectExpr("id % 65 AS wide", "CAST(id % 3 AS DOUBLE) AS real")
    columns = profile_dataframe("quantiles", frame).tables[0].columns

    assert all(column.numeric is not None and column.numeric.histogram is None for column in columns)
    assert all(column.numeric is not None and column.numeric.quantiles for column in columns)


def test_nested_integer_children_get_histograms(spark: SparkSession) -> None:
    frame = spark.range(100).selectExpr(
        "array(CAST(id % 4 AS INT)) AS items",
        "named_struct('a', CAST(id % 3 AS INT), 'b', CAST(id % 5 AS INT)) AS pair",
    )
    columns = profile_dataframe("nested", frame).tables[0].columns

    assert columns[0].children is not None
    histogram = columns[0].children[0].numeric.histogram if columns[0].children[0].numeric else None
    assert histogram is not None and histogram.edges == [0, 1, 2, 3, 3]
    assert histogram.counts == [25, 25, 25, 25]
    assert columns[1].children is not None
    assert [child.numeric.histogram is not None for child in columns[1].children if child.numeric] == [True, True]


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([(5, 9), (6, 1)], None),
        ([(5, 10), (6, 9)], None),
        ([(5, 10), (6, 10)], ([5, 6, 6], [10, 10])),
        ([(1, 1), (2, 30), (3, 30), (4, 1)], ([1, 3, 4], [31, 31])),
    ],
)
def test_histogram_bins_hold_at_least_k_rows(
    counts: list[tuple[int, int]], expected: tuple[list[int], list[int]] | None
) -> None:
    histogram = _histogram(counts, T.IntegerType(), PrivacyPolicy())

    assert (histogram and (histogram.edges, histogram.counts)) == expected


def test_suppressed_extremes_widen_histogram_outer_edges() -> None:
    policy = PrivacyPolicy(extremes=ValueTreatment.SUPPRESSED)
    histogram = _histogram([(23, 10), (40, 10), (70, 10)], T.IntegerType(), policy)
    on_step = _histogram([(20, 10), (25, 10), (30, 10)], T.IntegerType(), policy)

    assert histogram is not None and on_step is not None
    assert histogram.generalized_outer_edges and histogram.edges == [20, 40, 70, 80]
    assert on_step.edges == [10, 25, 30, 40]
    assert _histogram([(-128, 10), (0, 10)], T.ByteType(), policy) is None
    assert _histogram([(0, 10), (2**63 - 1, 10)], T.LongType(), policy) is None


def test_int64_histogram_edges_are_lossless() -> None:
    values = {"c0_non_null": 20, "c0_value_counts": [(-(2**63), 10), (2**63 - 1, 10)]}
    histogram = _numeric_stats(T.LongType(), values, "c0", PrivacyPolicy()).histogram

    assert histogram is not None
    assert histogram.edges == [str(-(2**63)), str(2**63 - 1), str(2**63 - 1)]


def test_timestamp_fraction_is_canonical_six_digits(spark: SparkSession) -> None:
    """Spark cast-to-string drops trailing zeros (5 digits here); the engine must pad to micros."""
    frame = spark.sql(
        """
        SELECT
          TIMESTAMP '2024-02-04 16:27:43.79144' AS ts,
          TIMESTAMP_NTZ '2024-02-04 16:27:43.79144' AS ntz,
          TIMESTAMP '2024-02-04 16:27:43' AS whole
        """
    )
    columns = {column.name: column for column in profile_dataframe("fractions", frame).tables[0].columns}
    assert columns["ts"].temporal is not None
    assert columns["ntz"].temporal is not None
    assert columns["whole"].temporal is not None
    assert columns["ts"].temporal.min == "2024-02-04T16:27:43.791440Z"
    assert columns["ts"].temporal.max == "2024-02-04T16:27:43.791440Z"
    assert columns["ntz"].temporal.min == "2024-02-04T16:27:43.791440"
    assert columns["ntz"].temporal.max == "2024-02-04T16:27:43.791440"
    assert columns["whole"].temporal.min == "2024-02-04T16:27:43Z"
    assert columns["whole"].temporal.max == "2024-02-04T16:27:43Z"


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


def test_lossless_decimal_and_int64_extremes_and_quantiles(spark: SparkSession) -> None:
    schema = T.StructType([T.StructField("integer", T.LongType()), T.StructField("decimal", T.DecimalType(38, 18))])
    rows = [
        (-(2**63), Decimal("-99999999999999999999.999999999999999999")),
        (2**63 - 1, Decimal("99999999999999999999.999999999999999999")),
    ] * 30
    columns = profile_dataframe("lossless", spark.createDataFrame(rows, schema)).tables[0].columns
    assert columns[0].numeric is not None and columns[1].numeric is not None
    assert columns[0].numeric.min == "-9223372036854775808"
    assert columns[0].numeric.max == "9223372036854775807"
    assert columns[1].numeric.min == "-99999999999999999999.999999999999999999"
    assert columns[1].numeric.max == "99999999999999999999.999999999999999999"
    assert columns[0].numeric.quantiles is None
    assert columns[0].numeric.histogram is not None
    assert columns[0].numeric.histogram.edges == ["-9223372036854775808", "9223372036854775807", "9223372036854775807"]


def test_lossless_decimal_quantiles(spark: SparkSession) -> None:
    values = [Decimal("1.250000000000000000"), Decimal("9.750000000000000000")]
    frame = spark.createDataFrame([(value,) for value in values] * 30, "value decimal(38,18)")
    stats = profile_dataframe("lossless", frame).tables[0].columns[0].numeric

    assert stats is not None
    quantiles = stats.quantiles or []
    assert [type(item.value) for item in quantiles] == [str] * 7
    assert [item.value for item in quantiles] == [
        "1.250000000000000000",
        "1.250000000000000000",
        "1.250000000000000000",
        "1.250000000000000000",
        "9.750000000000000000",
        "9.750000000000000000",
        "9.750000000000000000",
    ]


def test_decimal_quantile_interiors_never_round_trip_through_float(spark: SparkSession) -> None:
    values = [
        Decimal("1.0000000000"),
        Decimal("2.0000000001"),
        Decimal("3.0000000001"),
        Decimal("4.0000000001"),
        Decimal("5.0000000001"),
        Decimal("6.0000000001"),
        Decimal("9.0000000000"),
    ]
    frame = spark.createDataFrame([(value,) for value in values for _ in range(10)], "value decimal(38,10)")
    stats = profile_dataframe("lossless", frame).tables[0].columns[0].numeric

    assert stats is not None
    assert [item.value for item in stats.quantiles or []] == [
        "1.0000000000",
        "1.0000000000",
        "2.0000000001",
        "4.0000000001",
        "6.0000000001",
        "9.0000000000",
        "9.0000000000",
    ]


def test_non_decimal_quantile_result_omits_all_decimal_quantiles() -> None:
    low = Decimal("1.0000000000")
    high = Decimal("9.0000000000")
    values = {
        "c0_non_null": 60,
        "c0_min": low,
        "c0_max": high,
        "c0_quantiles": [low, low, low, low, 3.0000000001, Decimal("5.0000000000"), high, high, high, high, high],
    }

    assert _numeric_stats(T.DecimalType(38, 10), values, "c0", PrivacyPolicy()).quantiles is None
    values["c0_quantiles"] = [low, low, low, low, Decimal("3.000000001"), low, high, high, high, high, high]
    assert _numeric_stats(T.DecimalType(38, 10), values, "c0", PrivacyPolicy()).quantiles is None


def test_non_integral_quantile_result_omits_all_int64_quantiles() -> None:
    low = 2**63 - 100
    high = 2**63 - 1
    values = {
        "c0_non_null": 60,
        "c0_min": low,
        "c0_max": high,
        "c0_quantiles": [low, low, low, low, float(low + 25), low + 50, high, high, high, high, high],
    }

    assert _numeric_stats(T.LongType(), values, "c0", PrivacyPolicy()).quantiles is None


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


@pytest.mark.parametrize("row_count", [1, 59, 60])
def test_suppressed_extremes_never_appear_as_quantiles(spark: SparkSession, row_count: int) -> None:
    policy = PrivacyPolicy(extremes=ValueTreatment.SUPPRESSED)
    profile = profile_dataframe(
        "private",
        spark.createDataFrame([(1, date(2026, 1, 1))] * row_count, "n int, d date"),
        privacy=policy,
    )
    assert profile.tables[0].columns[0].numeric is not None
    assert profile.tables[0].columns[1].temporal is not None
    assert profile.tables[0].columns[0].numeric.min is None
    assert profile.tables[0].columns[1].temporal.min is None
    assert profile.tables[0].columns[0].numeric.quantiles is None
    assert profile.tables[0].columns[1].temporal.quantiles is None


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
