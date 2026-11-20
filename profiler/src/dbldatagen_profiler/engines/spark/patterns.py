"""Privacy-safe string patterns derived from character-class signatures.

Each value is mapped to a signature: ASCII letters and digits become class tokens and every other character stays as
itself. Runs of one token collapse into a shape (for example `AAA-9999` becomes `A-9`), and one batched Spark job per
table counts each shape and the shortest and longest run at each position. A shape becomes a pattern such as
`[A-Z]{3}-\\d{4}` when it covers at least `k` rows and the pattern's language is too large to single out a value.

Two signature levels are counted: a fine one that keeps upper case, lower case, and digits apart, and a coarse one that
merges them into one alphanumeric class (so mixed tokens such as hex digests form one shape). Per column, the level
whose patterns cover more rows wins; ties go to the fine level.

Literal-free columns (direct identifiers, secrets, free text, or policy-protected columns) never carry a literal
character: punctuation becomes `[^\\w\\s]`, `_` becomes `\\w`, other non-ASCII characters become `.`, and shapes that
contain whitespace are skipped. On other columns, literal characters are kept only as whole runs of a shape, so every
literal occurs in at least `k` rows.
"""

from __future__ import annotations

import math
import string
from collections.abc import Sequence
from dataclasses import dataclass

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from dbldatagen_profiler.model import ColumnProfile, Pattern


MAX_PATTERNS = 5
MAX_RUNS = 24
MAX_VALUE_LENGTH = 256
MIN_LANGUAGE = 1_000_000
FINE = 0
COARSE = 1

_UPPER, _LOWER, _DIGIT, _ALNUM = "A", "a", "9", "x"
_CLASS_SIZES = {_UPPER: 26, _LOWER: 26, _DIGIT: 10, _ALNUM: 10}
_ESCAPED = frozenset(".^$*+?{}[]\\|()")
_PUNCTUATION = frozenset(string.punctuation) - {"_"}
_FLAGS = ("digit", "upper_af", "upper_gz", "lower_af", "lower_gz")
_FLAG_PATTERNS = ("[0-9]", "[A-F]", "[G-Z]", "[a-f]", "[g-z]")


@dataclass(frozen=True)
class Shape:
    """Counted shape of one column at one signature level.

    Attributes:
        level: `FINE` or `COARSE`.
        tokens: One token per run: a class token or a literal character.
        low: Shortest run at each position.
        high: Longest run at each position.
        count: Rows with this shape.
        flags: Character classes seen in the shape's rows, used to size the coarse alphanumeric class.
    """

    level: int
    tokens: str
    low: tuple[int, ...]
    high: tuple[int, ...]
    count: int
    flags: frozenset[str] = frozenset()


def string_paths(schema: T.StructType) -> list[tuple[str, Column]]:
    """Returns the dotted path and Spark expression of every string column reachable through structs only."""
    found: list[tuple[str, Column]] = []

    def walk(fields: Sequence[T.StructField], prefix: str, parent: Column | None) -> None:
        for field in fields:
            expr = F.col("`" + field.name.replace("`", "``") + "`") if parent is None else parent.getField(field.name)
            path = f"{prefix}{field.name}"
            if isinstance(field.dataType, T.StringType):
                found.append((path, expr))
            elif isinstance(field.dataType, T.StructType):
                walk(field.dataType.fields, f"{path}.", expr)

    walk(schema.fields, "", None)
    return found


def collect_shapes(df: DataFrame, exprs: Sequence[Column], k: int) -> dict[int, list[Shape]]:
    """Counts the shapes of every expression in one Spark job; returns them by expression index.

    Rows longer than `MAX_VALUE_LENGTH`, with more than `MAX_RUNS` runs, or whose own shape has fewer than
    `MIN_LANGUAGE` strings are not counted. Only shapes with at least `k` rows are returned, at most
    `4 * MAX_PATTERNS` per expression and level.
    """
    if not exprs:
        return {}
    values = (
        df.select(
            F.explode(
                F.array(*[F.struct(F.lit(i).alias("c"), e.cast("string").alias("v")) for i, e in enumerate(exprs)])
            ).alias("s")
        )
        .select("s.c", "s.v")
        .where(F.col("v").isNotNull() & (F.length("v") <= MAX_VALUE_LENGTH))
    )
    upper, lower, digits = string.ascii_uppercase, string.ascii_lowercase, string.digits
    signatures = F.array(
        F.struct(
            F.lit(FINE).alias("level"),
            F.translate("v", upper + lower + digits, _UPPER * 26 + _LOWER * 26 + _DIGIT * 10).alias("sig"),
        ),
        F.struct(F.lit(COARSE).alias("level"), F.translate("v", upper + lower + digits, _ALNUM * 62).alias("sig")),
    )
    runs = F.expr(r"regexp_extract_all(sig, '(?s)(A+|a+|9+|x+|(.)\\2*)', 0)")
    shaped = (
        values.select("c", "v", F.explode(signatures).alias("s"))
        .select("c", "v", "s.level", "s.sig")
        .select("c", "v", "level", runs.alias("runs"))
        .select(
            "c",
            "v",
            "level",
            F.array_join(F.transform("runs", lambda r: F.substring(r, 1, 1)), "").alias("shape"),
            F.transform("runs", F.length).alias("lens"),
            F.aggregate("runs", F.lit(0.0), lambda acc, r: acc + F.length(r) * _size_log(r)).alias("language"),
        )
        .where((F.size("lens") <= MAX_RUNS) & (F.col("language") >= math.log10(max(k, MIN_LANGUAGE))))
    )
    aggregates = [F.count(F.lit(1)).alias("n")]
    aggregates += [F.min(F.get("lens", p)).alias(f"lo{p}") for p in range(MAX_RUNS)]
    aggregates += [F.max(F.get("lens", p)).alias(f"hi{p}") for p in range(MAX_RUNS)]
    aggregates += [F.max(F.col("v").rlike(r)).alias(f) for f, r in zip(_FLAGS, _FLAG_PATTERNS, strict=True)]
    rank = F.row_number().over(Window.partitionBy("c", "level").orderBy(F.desc("n"), "shape"))
    grouped = (
        shaped.groupBy("c", "level", "shape")
        .agg(*aggregates)
        .where(F.col("n") >= k)
        .withColumn("rank", rank)
        .where(F.col("rank") <= 4 * MAX_PATTERNS)
    )
    result: dict[int, list[Shape]] = {}
    for row in grouped.collect():
        width = len(row["shape"])
        result.setdefault(row["c"], []).append(
            Shape(
                level=row["level"],
                tokens=row["shape"],
                low=tuple(row[f"lo{p}"] for p in range(width)),
                high=tuple(row[f"hi{p}"] for p in range(width)),
                count=row["n"],
                flags=frozenset(f for f in _FLAGS if row[f]),
            )
        )
    return result


def _size_log(run: Column) -> Column:
    """Returns log10 of the smallest class size of a run's token; literal runs count as one string."""
    token = F.substring(run, 1, 1)
    return (
        F.when(token.isin(_UPPER, _LOWER), F.lit(math.log10(26)))
        .when(token.isin(_DIGIT, _ALNUM), F.lit(1.0))
        .otherwise(F.lit(0.0))
    )


def patterns_for(shapes: Sequence[Shape], non_null: int, k: int, *, literal_free: bool) -> list[Pattern] | None:
    """Returns up to `MAX_PATTERNS` privacy-safe patterns with coverage from the shapes of one column.

    Args:
        shapes: Counted shapes of the column at both levels.
        non_null: Non-null values of the column.
        k: Fewest rows a pattern must match.
        literal_free: Whether the column may not carry literal characters.

    Returns:
        Patterns ordered by coverage, or None when no shape qualifies.
    """
    best: list[tuple[str, int]] = []
    for level in (FINE, COARSE):
        merged: dict[tuple[str, ...], tuple[int, list[tuple[str, int, int, int]]]] = {}
        for shape in (s for s in shapes if s.level == level):
            atoms = _atoms(shape, literal_free=literal_free)
            if atoms is None:
                continue
            key = tuple(a for a, _, _, _ in atoms)
            if key in merged:
                count, previous = merged[key]
                atoms = [
                    (a, size, min(lo, plo), max(hi, phi))
                    for (a, size, lo, hi), (_, _, plo, phi) in zip(atoms, previous, strict=True)
                ]
                merged[key] = (count + shape.count, atoms)
            else:
                merged[key] = (shape.count, atoms)
        candidates = sorted(
            ((_serialize(atoms), count) for count, atoms in merged.values() if count >= k and _large(atoms, k)),
            key=lambda item: (-item[1], item[0]),
        )[:MAX_PATTERNS]
        if sum(c for _, c in candidates) > sum(c for _, c in best):
            best = candidates
    if not best or non_null <= 0:
        return None
    return [Pattern(regex=regex, coverage=count / non_null) for regex, count in best]


def with_patterns(columns: Sequence[ColumnProfile], found: dict[str, list[Pattern]]) -> list[ColumnProfile]:
    """Returns the columns with `string.patterns` set from `found`, keyed by dotted path."""

    def apply(column: ColumnProfile, path: str) -> ColumnProfile:
        updates: dict[str, object] = {}
        if path in found and column.string is not None:
            updates["string"] = column.string.model_copy(update={"patterns": found[path]})
        if column.children is not None:
            updates["children"] = [apply(child, f"{path}.{child.name}") for child in column.children]
        return column.model_copy(update=updates) if updates else column

    return [apply(column, column.name) for column in columns]


def column_at(columns: Sequence[ColumnProfile], path: str) -> ColumnProfile:
    """Returns the column profile at a dotted struct path."""
    head, *rest = path.split(".")
    column = next(c for c in columns if c.name == head)
    for name in rest:
        column = next(c for c in column.children or [] if c.name == name)
    return column


def non_null_count(column: ColumnProfile, row_count: int) -> int:
    """Returns the non-null values of a column from its value count and null fraction."""
    total = column.value_count if column.value_count is not None else row_count
    return round(total * (1.0 - column.null_fraction))


def _atoms(shape: Shape, *, literal_free: bool) -> list[tuple[str, int, int, int]] | None:
    """Returns `(atom, class size, low, high)` per run with equal adjacent atoms joined, or None if ineligible."""
    atoms: list[tuple[str, int, int, int]] = []
    for token, low, high in zip(shape.tokens, shape.low, shape.high, strict=True):
        atom = _atom(token, shape.flags, literal_free=literal_free)
        if atom is None:
            return None
        text, size = atom
        if atoms and atoms[-1][0] == text:
            _, _, plo, phi = atoms[-1]
            atoms[-1] = (text, size, plo + low, phi + high)
        else:
            atoms.append((text, size, low, high))
    return atoms


def _atom(token: str, flags: frozenset[str], *, literal_free: bool) -> tuple[str, int] | None:  # noqa: PLR0911
    """Returns the regex atom and the smallest class size of one signature token, or None when it is ineligible."""
    if token == _UPPER:
        return "[A-Z]", 26
    if token == _LOWER:
        return "[a-z]", 26
    if token == _DIGIT:
        return r"\d", 10
    if token == _ALNUM:
        return _alnum_class(flags)
    if literal_free:
        if token.isspace():
            return None
        if token == "_":
            return r"\w", 1
        if token in _PUNCTUATION:
            return r"[^\w\s]", 1
        return ".", 1
    if token == " ":
        return "[ ]", 1
    if token.isspace() or not token.isprintable():
        return (r"\s", 1) if token.isspace() else None
    return (f"\\{token}" if token in _ESCAPED else token), 1


def _alnum_class(flags: frozenset[str]) -> tuple[str, int]:
    """Returns the narrowest fixed alphanumeric class covering the seen character classes, with its size."""
    parts = [("0-9", 10)] if "digit" in flags else []
    if "upper_gz" in flags:
        parts.append(("A-Z", 26))
    elif "upper_af" in flags:
        parts.append(("A-F", 6))
    if "lower_gz" in flags:
        parts.append(("a-z", 26))
    elif "lower_af" in flags:
        parts.append(("a-f", 6))
    if parts == [("0-9", 10)]:
        return r"\d", 10
    return "[" + "".join(p for p, _ in parts) + "]", sum(n for _, n in parts)


def _large(atoms: Sequence[tuple[str, int, int, int]], k: int) -> bool:
    """Returns whether the pattern's shortest matches alone number at least max(k, `MIN_LANGUAGE`)."""
    return sum(lo * math.log10(size) for _, size, lo, _ in atoms) >= math.log10(max(k, MIN_LANGUAGE))


def _serialize(atoms: Sequence[tuple[str, int, int, int]]) -> str:
    return "".join(atom + _quantifier(lo, hi) for atom, _, lo, hi in atoms)


def _quantifier(low: int, high: int) -> str:
    if low == high:
        return "" if low == 1 else f"{{{low}}}"
    return f"{{{low},{high}}}"
