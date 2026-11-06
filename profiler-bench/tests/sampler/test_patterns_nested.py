from __future__ import annotations

import re

import numpy as np
import pytest
from pyspark.sql import SparkSession

from dbldatagen_profiler.model import LengthStats, LogicalType, NumericStats, Pattern, StringStats, TableProfile
from profiler_bench.sampler import sample
from profiler_bench.sampler.regex import generate_regex

from .profiles import column, profile


@pytest.mark.parametrize(
    "pattern",
    [
        r"literal",
        r"[A-Z][0-9a-f]",
        r"\d\w\s",
        r"a{2,4}",
        r"colou?r",
        r"ab*",
        r"ab+",
        r"(cat|dog)",
        r"(ab|cd){1,3}",
        r"[^0-9]{2}",
    ],
)
def test_regex_supported_subset_fullmatches(pattern: str) -> None:
    rng = np.random.default_rng(11)

    assert all(re.fullmatch(pattern, generate_regex(pattern, rng, max_length=20)) for _ in range(100))


@pytest.mark.parametrize("pattern", [r"a*", r"b+", r"(ab|c)*", r"\d+"])
def test_unbounded_repeats_respect_length_bound(pattern: str) -> None:
    rng = np.random.default_rng(12)
    values = [generate_regex(pattern, rng, max_length=5) for _ in range(100)]

    assert all(len(value) <= 5 and re.fullmatch(pattern, value) for value in values)


def test_pattern_coverage_weights_are_recovered(spark: SparkSession) -> None:
    stats = StringStats(
        length=LengthStats(min=2, max=4),
        patterns=[Pattern(regex=r"[0-9]{4}", coverage=0.7), Pattern(regex=r"[A-Z]{2}", coverage=0.3)],
    )
    values = (
        sample(
            profile(
                TableProfile(name="t", row_count=4_000, columns=[column("code", LogicalType.STRING, string=stats)])
            ),
            spark,
            seed=13,
        )
        .tables["t"]
        .collect()
    )
    strings = [row.code for row in values]

    assert sum(bool(re.fullmatch(r"[0-9]{4}", value)) for value in strings) / len(strings) == pytest.approx(
        0.7, abs=0.03
    )
    assert sum(bool(re.fullmatch(r"[A-Z]{2}", value)) for value in strings) / len(strings) == pytest.approx(
        0.3, abs=0.03
    )
    assert all(2 <= len(value) <= 4 for value in strings)


def test_nested_lengths_and_child_marginals(spark: SparkSession) -> None:
    array = column(
        "items",
        LogicalType.ARRAY,
        children=[column("element", LogicalType.INT32, numeric=NumericStats(min=5, max=5))],
        collection_length=LengthStats(min=3, max=3, mean=3, stddev=0),
    )
    mapping = column(
        "lookup",
        LogicalType.MAP,
        children=[
            column("key", LogicalType.STRING, string=StringStats(length=LengthStats(min=8, max=8))),
            column("value", LogicalType.FLOAT64, numeric=NumericStats(min=2.5, max=2.5)),
        ],
        collection_length=LengthStats(min=2, max=2, mean=2, stddev=0),
    )
    rows = (
        sample(profile(TableProfile(name="t", row_count=500, columns=[array, mapping])), spark, seed=14)
        .tables["t"]
        .collect()
    )

    assert all(len(row.items) == 3 and set(row.items) == {5} for row in rows)
    assert all(len(row.lookup) == 2 and set(row.lookup.values()) == {2.5} for row in rows)
