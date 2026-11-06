"""Messy edge cases: all-null and constant columns, type extremes, special floats, blanks, and unicode."""

from __future__ import annotations

import sys
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from profiler_bench.catalog import gen
from profiler_bench.catalog.gen import DIGITS, Rng
from profiler_bench.catalog.manifest import Column, Distribution, Family, Format, RangeCheck, Table
from profiler_bench.catalog.registry import Build, Dataset


INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1
INT64_PLANTED = {"min": 0.01, "max": 0.01, "zero": 0.01}
DECIMAL_SCALED = (38, 10)
DECIMAL_SCALED_MAX = "9" * 28 + "." + "9" * 10
DECIMAL_WHOLE_MAX = "9" * 38
DOUBLE_MAX = sys.float_info.max
DOUBLE_TINY = 5e-324
DOUBLE_EXPONENT = 300
SPECIAL_SHARES = {"nan": 0.02, "inf": 0.01, "-inf": 0.01}
SPECIAL_NULL_RATE = 0.05
BLANK_NULL_RATE = 0.1
BLANK_EMPTY_RATE = 0.2
UNICODE = ["café", "naïve", "東京", "Привет", "مرحبا", "😀", "é", "zero\u200bwidth", "Ω≈ç√", "ß"]
BOOL_NULL_RATE = 0.3
BOOL_TRUE_SHARE = 0.25
DATE_RANGE = ("0001-01-01", "9999-12-31")
BLOB_MAX_LENGTH = 16
CONSTANT_STRING, CONSTANT_INT = "X", 7


def _digit_strings(rng: np.random.Generator, n: int, width: int, low: int) -> pa.Array:
    """Returns zero-padded digit strings of `width` with `low..width` significant digits, uniformly."""
    codes = np.frombuffer(DIGITS.encode(), dtype=np.uint8)[rng.integers(0, 10, (n, width))]
    significant = rng.integers(low, width + 1, n)
    codes[np.arange(width)[None, :] < (width - significant)[:, None]] = ord("0")
    return gen.fixed_strings(codes)


def _signs(rng: np.random.Generator, n: int) -> pa.Array:
    return gen.take(["", "-"], rng.integers(0, 2, n))


def _plant(array: pa.Array, values: list[object]) -> pa.Array:
    head = pa.array(values, type=array.type)
    return pa.concat_arrays([head, array.slice(len(values))])


def build(n: int, rng: Rng) -> Build:
    """Generates the `edge_cases` table with `n` rows and the zero-row `empty` table."""
    int_digits = DECIMAL_SCALED[0] - DECIMAL_SCALED[1]
    scaled = gen.join(
        _signs(rng("dec38_10.sign"), n),
        _digit_strings(rng("dec38_10.int"), n, int_digits, 1),
        ".",
        gen.random_strings(rng("dec38_10.frac"), n, [DIGITS] * DECIMAL_SCALED[1]),
    ).cast(pa.decimal128(*DECIMAL_SCALED))
    whole = gen.join(_signs(rng("dec38_0.sign"), n), _digit_strings(rng("dec38_0.int"), n, 38, 1)).cast(
        pa.decimal128(38, 0)
    )

    int64 = rng("int64").integers(INT64_MIN, INT64_MAX, n, dtype=np.int64, endpoint=True)
    kind = rng("int64.kind").random(n)
    edges = np.cumsum(list(INT64_PLANTED.values()))
    int64 = np.select([kind < edges[0], kind < edges[1], kind < edges[2]], [INT64_MIN, INT64_MAX, 0], int64)

    special = gen.choice(rng("special.kind"), n, [*SPECIAL_SHARES.values(), 1 - sum(SPECIAL_SHARES.values())])
    specials = np.select(
        [special == 0, special == 1, special == 2], [np.nan, np.inf, -np.inf], rng("special").standard_normal(n)
    )

    magnitude = 10.0 ** rng("wide_range").uniform(-DOUBLE_EXPONENT, DOUBLE_EXPONENT, n)
    wide_range = np.where(rng("wide_range.sign").random(n) < 0.5, -magnitude, magnitude)

    blank_kind = rng("blank.kind").random(n)
    vocabulary = gen.words(rng("blank.vocabulary"), 200)
    blank = pc.if_else(
        pa.array(blank_kind < BLANK_EMPTY_RATE),
        "",
        gen.take(vocabulary, rng("blank.words").integers(0, len(vocabulary), n)),
    )
    padding = rng("padded.kind").integers(0, 3, n)
    word = gen.take(vocabulary, rng("padded.words").integers(0, len(vocabulary), n))
    spaces = gen.take([" ", "  ", "   "], rng("padded.spaces").integers(0, 3, n))
    padded = pc.if_else(
        pa.array(padding == 0),
        word,
        pc.if_else(pa.array(padding == 1), gen.join(spaces, word), gen.join(word, spaces)),
    )

    blob_lengths = rng("blob.length").integers(0, BLOB_MAX_LENGTH + 1, n)
    blob_bytes = rng("blob").integers(0, 256, int(blob_lengths.sum()), dtype=np.uint8).tobytes()
    blob = pa.Array.from_buffers(
        pa.binary(), n, [None, gen.list_offsets(blob_lengths).buffers()[1], pa.py_buffer(blob_bytes)]
    )

    table = pa.table(
        {
            "id": pa.array(np.arange(1, n + 1), pa.int64()),
            "all_null_string": pa.nulls(n, pa.string()),
            "all_null_int": pa.nulls(n, pa.int32()),
            "all_null_double": pa.nulls(n, pa.float64()),
            "constant_string": pa.array([CONSTANT_STRING] * n, pa.string()),
            "constant_int": pa.array(np.full(n, CONSTANT_INT), pa.int32()),
            "dec38_10": _plant(scaled, [Decimal(DECIMAL_SCALED_MAX), Decimal("-" + DECIMAL_SCALED_MAX)]),
            "dec38_0": _plant(whole, [Decimal(DECIMAL_WHOLE_MAX), Decimal("-" + DECIMAL_WHOLE_MAX)]),
            "int64_extremes": _plant(pa.array(int64, pa.int64()), [INT64_MIN, INT64_MAX]),
            "int8_full": pa.array(rng("int8").integers(-128, 128, n), pa.int8()),
            "special_floats": pa.array(specials, mask=gen.null_mask(rng("special.nulls"), n, SPECIAL_NULL_RATE)),
            "wide_range": _plant(pa.array(wide_range), [DOUBLE_MAX, -DOUBLE_MAX, DOUBLE_TINY]),
            "blank": gen.with_nulls(blank, blank_kind > 1 - BLANK_NULL_RATE),
            "padded": padded,
            "unicode": gen.take(UNICODE, rng("unicode").integers(0, len(UNICODE), n)),
            "flag": pa.array(
                rng("flag").random(n) < BOOL_TRUE_SHARE, mask=gen.null_mask(rng("flag.nulls"), n, BOOL_NULL_RATE)
            ),
            "date_extremes": _plant(
                gen.dates(rng("date").integers(gen.days(DATE_RANGE[0]), gen.days(DATE_RANGE[1]) + 1, n)),
                [np.datetime64(DATE_RANGE[0]).item(), np.datetime64(DATE_RANGE[1]).item()],
            ),
            "blob": blob,
        }
    )
    empty = pa.table({"id": pa.array([], pa.int64()), "name": pa.array([], pa.string())})
    return Build({"edge_cases": table, "empty": empty}, [_truth(n), _empty_truth()], [])


def _dist(family: Family, **params: object) -> Distribution:
    return Distribution(family=family, params=params)


def _truth(n: int) -> Table:
    non_null_blank = 1 - BLANK_NULL_RATE
    return Table(
        name="edge_cases",
        description="Columns that break naive profilers and samplers.",
        primary_key=("id",),
        columns=(
            Column(name="id", unique=True, distinct=n, distribution=_dist(Family.SEQUENCE, start=1, step=1)),
            Column(name="all_null_string", null_rate=1.0, distinct=0),
            Column(name="all_null_int", null_rate=1.0, distinct=0),
            Column(name="all_null_double", null_rate=1.0, distinct=0),
            Column(name="constant_string", distinct=1, distribution=_dist(Family.CONSTANT, value=CONSTANT_STRING)),
            Column(name="constant_int", distinct=1, distribution=_dist(Family.CONSTANT, value=CONSTANT_INT)),
            Column(
                name="dec38_10",
                min_value="-" + DECIMAL_SCALED_MAX,
                max_value=DECIMAL_SCALED_MAX,
                note="Random sign, 1..28 significant integer digits, 10 random fraction digits; extremes planted.",
            ),
            Column(
                name="dec38_0",
                min_value="-" + DECIMAL_WHOLE_MAX,
                max_value=DECIMAL_WHOLE_MAX,
                note="Random sign and 1..38 significant digits; extremes planted.",
            ),
            Column(
                name="int64_extremes",
                min_value=INT64_MIN,
                max_value=INT64_MAX,
                note=f"Uniform over int64 with {INT64_PLANTED} shares planted at min, max, and zero.",
            ),
            Column(name="int8_full", distribution=_dist(Family.DISCRETE_UNIFORM, low=-128, high=127)),
            Column(
                name="special_floats",
                null_rate=SPECIAL_NULL_RATE,
                special_values=SPECIAL_SHARES,
                distribution=_dist(Family.NORMAL, mean=0.0, std=1.0),
                note="Finite values are standard normal.",
            ),
            Column(
                name="wide_range",
                min_value=-DOUBLE_MAX,
                max_value=DOUBLE_MAX,
                note=f"Random sign times 10**U(-{DOUBLE_EXPONENT}, {DOUBLE_EXPONENT}); max, -max, and 5e-324 planted.",
            ),
            Column(
                name="blank",
                null_rate=BLANK_NULL_RATE,
                formats=(
                    Format(regex="", share=BLANK_EMPTY_RATE / non_null_blank),
                    Format(regex="[a-z]+", share=1 - BLANK_EMPTY_RATE / non_null_blank),
                ),
                note="Empty strings and nulls are different values.",
            ),
            Column(
                name="padded",
                formats=(
                    Format(regex="[a-z]+", share=1 / 3),
                    Format(regex=" {1,3}[a-z]+", share=1 / 3),
                    Format(regex="[a-z]+ {1,3}", share=1 / 3),
                ),
            ),
            Column(
                name="unicode",
                max_distinct=len(UNICODE),
                distribution=_dist(Family.CATEGORICAL, values=UNICODE, probabilities=[1 / len(UNICODE)] * len(UNICODE)),
                note="Accents, CJK, Cyrillic, Arabic, emoji, a combining mark, and a zero-width space.",
            ),
            Column(
                name="flag",
                null_rate=BOOL_NULL_RATE,
                distribution=_dist(
                    Family.CATEGORICAL, values=[True, False], probabilities=[BOOL_TRUE_SHARE, 1 - BOOL_TRUE_SHARE]
                ),
            ),
            Column(
                name="date_extremes",
                min_value=DATE_RANGE[0],
                max_value=DATE_RANGE[1],
                distribution=_dist(Family.UNIFORM, low=DATE_RANGE[0], high=DATE_RANGE[1]),
            ),
            Column(name="blob", length=_dist(Family.DISCRETE_UNIFORM, low=0, high=BLOB_MAX_LENGTH)),
        ),
        checks=(
            RangeCheck(column="int8_full", min=-128, max=127),
            RangeCheck(column="date_extremes", min=DATE_RANGE[0], max=DATE_RANGE[1]),
        ),
    )


def _empty_truth() -> Table:
    return Table(
        name="empty",
        description="A table with a schema and no rows.",
        primary_key=("id",),
        columns=(Column(name="id"), Column(name="name")),
    )


DATASET = Dataset(
    name="edge",
    description="All-null and constant columns, decimal(38) and int64 extremes, NaN/inf, blanks vs nulls, unicode.",
    build=build,
)
