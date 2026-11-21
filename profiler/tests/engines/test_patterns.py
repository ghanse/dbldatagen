"""Tests for privacy-safe string patterns."""

import re

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import types as T

from dbldatagen_profiler.engines.spark.engine import profile_dataframe
from dbldatagen_profiler.engines.spark.patterns import COARSE, FINE, MAX_PATTERNS, Shape, patterns_for
from dbldatagen_profiler.model import Pattern, PrivacyPolicy, Profile, dump_profile
from dbldatagen_profiler.model.profile import _is_generalized_pattern


pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")

DIALECT = re.compile(
    r"(?:(?:\[\^?[^\]]+\]|\\[dDwWsS.^$*+?{}\[\]\\|()]|\.|[^\\\[\]{}()|*+?^$.\s])(?:\{\d+(?:,\d+)?\})?)+"
)


def _strings(profile: Profile, name: str) -> list[Pattern]:
    column = next(c for c in profile.tables[0].columns if c.name == name)
    assert column.string is not None
    return column.string.patterns or []


def test_fine_shape_becomes_class_quantifier_pattern() -> None:
    shapes = [Shape(FINE, "A-9", (3, 1, 4), (3, 1, 4), 40)]

    assert patterns_for(shapes, 50, 10, literal_free=False) == [Pattern(regex=r"[A-Z]{3}-\d{4}", coverage=0.8)]


def test_literals_are_escaped_and_spaces_use_a_class() -> None:
    shapes = [Shape(FINE, "a.a a", (4, 1, 4, 1, 3), (8, 1, 8, 1, 5), 20)]

    (pattern,) = patterns_for(shapes, 20, 10, literal_free=False) or []

    assert pattern.regex == r"[a-z]{4,8}\.[a-z]{4,8}[ ][a-z]{3,5}"
    assert DIALECT.fullmatch(pattern.regex)


def test_literal_free_patterns_carry_no_literal_characters() -> None:
    shapes = [
        Shape(FINE, "a.a9@a.a", (4, 1, 4, 2, 1, 4, 1, 3), (6, 1, 6, 2, 1, 7, 1, 4), 30),
        Shape(FINE, "a_a9@a.a", (4, 1, 4, 2, 1, 4, 1, 3), (6, 1, 6, 2, 1, 7, 1, 4), 30),
    ]

    emitted = patterns_for(shapes, 60, 10, literal_free=True) or []

    assert [p.regex for p in emitted] == [
        r"[a-z]{4,6}[^\w\s][a-z]{4,6}\d{2}[^\w\s][a-z]{4,7}[^\w\s][a-z]{3,4}",
        r"[a-z]{4,6}\w[a-z]{4,6}\d{2}[^\w\s][a-z]{4,7}[^\w\s][a-z]{3,4}",
    ]
    assert all(_is_generalized_pattern(p.regex) and DIALECT.fullmatch(p.regex) for p in emitted)


def test_literal_free_shapes_with_whitespace_are_skipped() -> None:
    shapes = [Shape(FINE, "a a a", (5, 1, 5, 1, 5), (9, 1, 9, 1, 9), 50)]

    assert patterns_for(shapes, 50, 10, literal_free=True) is None


def test_literal_free_adjacent_punctuation_joins_into_one_run() -> None:
    shapes = [Shape(FINE, "9.-9", (6, 1, 1, 6), (6, 1, 1, 6), 50)]

    assert [p.regex for p in patterns_for(shapes, 50, 10, literal_free=True) or []] == [r"\d{6}[^\w\s]{2}\d{6}"]


def test_shapes_that_map_to_one_pattern_merge() -> None:
    shapes = [
        Shape(FINE, "9-9", (3, 1, 4), (3, 1, 4), 6),
        Shape(FINE, "9.9", (3, 1, 5), (3, 1, 5), 6),
    ]

    (pattern,) = patterns_for(shapes, 12, 10, literal_free=True) or []

    assert (pattern.regex, pattern.coverage) == (r"\d{3}[^\w\s]\d{4,5}", 1.0)


@pytest.mark.parametrize(
    "shape",
    [
        Shape(FINE, "9", (5,), (5,), 100),
        Shape(FINE, "A", (2,), (2,), 100),
        Shape(FINE, "Aa", (1, 3), (1, 9), 100),
    ],
)
def test_small_languages_are_not_emitted(shape: Shape) -> None:
    assert patterns_for([shape], 100, 10, literal_free=False) is None


def test_language_threshold_follows_k_when_k_is_larger() -> None:
    shape = Shape(FINE, "9", (7,), (7,), 10**9)

    assert patterns_for([shape], 10**9, 10**7, literal_free=False) is not None
    assert patterns_for([shape], 10**9, 10**8, literal_free=False) is None


def test_shapes_below_k_rows_are_not_emitted() -> None:
    shapes = [Shape(FINE, "A-9", (3, 1, 4), (3, 1, 4), 9)]

    assert patterns_for(shapes, 100, 10, literal_free=False) is None


def test_coarse_level_wins_only_when_it_covers_more_rows() -> None:
    hex_shapes = [Shape(FINE, "9a9", (2, 1, 3), (2, 1, 3), 3)] * 4 + [
        Shape(COARSE, "x", (32,), (32,), 40, frozenset({"digit", "lower_af"}))
    ]
    tie = [
        Shape(FINE, "A-9", (3, 1, 4), (3, 1, 4), 40),
        Shape(COARSE, "x-x", (3, 1, 4), (3, 1, 4), 40, frozenset({"digit", "upper_gz"})),
    ]

    assert [p.regex for p in patterns_for(hex_shapes, 40, 10, literal_free=True) or []] == ["[0-9a-f]{32}"]
    assert [p.regex for p in patterns_for(tie, 40, 10, literal_free=False) or []] == [r"[A-Z]{3}-\d{4}"]


def test_at_most_max_patterns_are_kept_by_coverage() -> None:
    shapes = [Shape(FINE, f"A{c}9", (3, 1, 4), (3, 1, 4), 10 + i) for i, c in enumerate("-.#:;~=+")]

    emitted = patterns_for(shapes, 200, 10, literal_free=False) or []

    assert len(emitted) == MAX_PATTERNS
    assert [round(p.coverage * 200) for p in emitted] == [17, 16, 15, 14, 13]


@pytest.mark.spark
def test_profile_emits_patterns_that_match_their_coverage(spark: SparkSession) -> None:
    values = [f"{chr(65 + i % 26)}{chr(65 + i // 26 % 26)}Q-{i:04d}" for i in range(60)] + ["rare"] * 5
    df = spark.createDataFrame([(v,) for v in values], "sku string")

    profile = profile_dataframe("t", df, privacy=PrivacyPolicy(min_value_count=10))
    (pattern,) = _strings(profile, "sku")

    assert pattern.regex == r"[A-Z]{3}-\d{4}"
    assert pattern.coverage == pytest.approx(60 / 65)
    assert sum(1 for v in values if re.fullmatch(pattern.regex, v)) == 60


@pytest.mark.spark
def test_detected_identifiers_get_literal_free_patterns(spark: SparkSession) -> None:
    rows = [(f"user{i:03d}.name@example.com", f"{i:010d}") for i in range(100)]
    df = spark.createDataFrame(rows, "email string, account_number string")

    profile = profile_dataframe("t", df, privacy=PrivacyPolicy(min_value_count=10))
    emitted = _strings(profile, "email")

    assert emitted
    assert all(_is_generalized_pattern(p.regex) and DIALECT.fullmatch(p.regex) for p in emitted)
    assert all("@" not in p.regex and "\\." not in p.regex for p in emitted)


@pytest.mark.spark
def test_struct_fields_get_patterns_and_profiles_are_deterministic(spark: SparkSession) -> None:
    schema = T.StructType([T.StructField("address", T.StructType([T.StructField("unit", T.StringType())]))])
    df = spark.createDataFrame([((f"{i:04d}-{i % 7:03d}",),) for i in range(500)] + [(None,)] * 20, schema)

    first = profile_dataframe("t", df, privacy=PrivacyPolicy(min_value_count=10))
    second = profile_dataframe("t", df, privacy=PrivacyPolicy(min_value_count=10))
    (child,) = first.tables[0].columns[0].children or []

    assert child.string is not None
    assert [(p.regex, p.coverage) for p in child.string.patterns or []] == [(r"\d{4}-\d{3}", 1.0)]
    strip = re.compile(r'"created_at":\s*"[^"]+"')
    assert strip.sub("", dump_profile(first)) == strip.sub("", dump_profile(second))


@pytest.mark.spark
def test_patterns_are_literal_free_without_pii_detection(spark: SparkSession) -> None:
    df = spark.createDataFrame([(f"ABC-{i:04d}",) for i in range(100)], "sku string")

    with_detection = profile_dataframe("t", df, privacy=PrivacyPolicy(min_value_count=10))
    without_detection = profile_dataframe("t", df, privacy=PrivacyPolicy(min_value_count=10), detect_pii=False)

    assert [p.regex for p in _strings(with_detection, "sku")] == [r"[A-Z]{3}-\d{4}"]
    assert [p.regex for p in _strings(without_detection, "sku")] == [r"[A-Z]{3}[^\w\s]\d{4}"]


@pytest.mark.spark
def test_dotted_field_names_are_not_read_as_nested_fields(spark: SparkSession) -> None:
    struct = T.StructType
    schema = struct(
        [
            T.StructField("a.b", T.StringType()),
            T.StructField("a", struct([T.StructField("b", T.StringType())])),
            T.StructField("s", struct([T.StructField("x.y", T.StringType())])),
        ]
    )
    rows = [(f"ABC-{i:04d}", (f"{i:04d}-{i % 7:03d}",), (f"abcd-{i:04d}",)) for i in range(100)]
    df = spark.createDataFrame(rows, schema)

    profile = profile_dataframe("t", df, privacy=PrivacyPolicy(min_value_count=10))
    dotted, nested, outer = profile.tables[0].columns
    (nested_b,) = nested.children or []
    (outer_xy,) = outer.children or []

    assert [p.regex for p in _strings(profile, "a.b")] == [r"[A-Z]{3}[^\w\s]\d{4}"]
    assert nested_b.string is not None and outer_xy.string is not None
    assert [p.regex for p in nested_b.string.patterns or []] == [r"\d{4}-\d{3}"]
    assert [p.regex for p in outer_xy.string.patterns or []] == [r"[a-z]{4}[^\w\s]\d{4}"]
    assert dotted.name == "a.b" and outer_xy.name == "x.y"
