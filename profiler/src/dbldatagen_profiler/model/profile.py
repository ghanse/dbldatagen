"""Pydantic models for the dbldatagen profile spec.

A `Profile` describes one or more tables and the relationships between them
without holding raw rows. It is the contract between profiling engines and
consumers such as data generators and the benchmark's reference sampler. Every
block except the basic schema, row count, and null information is optional so
engines can fill it in over time.

Literal values (statistics, histogram edges, and constraint bounds or values)
are typed by the column's logical type: `decimal`, `int64`, and `uint64` values
are canonical decimal strings so they survive JSON without loss, other numeric
values are JSON numbers, temporal values are ISO-8601 strings, `string` values
are strings, and `boolean` values are booleans.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from itertools import pairwise
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationInfo, model_validator


SPEC_VERSION = "0.2.0"
SUPPORTED_MAJOR_VERSION = int(SPEC_VERSION.split(".", maxsplit=1)[0])
STRICT_CONTEXT_KEY = "strict"
DEFAULT_MIN_VALUE_COUNT = 10

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Count = Annotated[int, Field(ge=0)]
Scalar = bool | int | float | str

_INTEGER_RE = re.compile(r"^-?(0|[1-9]\d*)$")
_UNSIGNED_RE = re.compile(r"^(0|[1-9]\d*)$")
_DECIMAL_RE = re.compile(r"^-?(0|[1-9]\d*)(\.\d+)?$")
_DATE = r"\d{4}-\d{2}-\d{2}"
_TIME = r"\d{2}:\d{2}:\d{2}(\.\d{1,9})?"
# Fractional seconds before an optional UTC offset / Z. Spec allows 1-9 digits; Python 3.10's
# fromisoformat only accepts 0, 3, or 6, so parsers normalize through this pattern.
_ISO_FRACTION_RE = re.compile(r"(\.\d{1,9})(?=(Z|[+-]\d{2}:\d{2})?$)")
_ISO_FRACTION_ANY_RE = re.compile(r"(\.\d+)(?=(Z|[+-]\d{2}:\d{2})?$)")


class _SpecModel(BaseModel):
    """Base model for every spec type.

    Unknown fields are ignored so that readers accept profiles written against a
    newer minor spec version. Validation with the strict context flag set rejects
    unknown fields instead; producers and tests use it to check their output.
    """

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_fields(cls, data: object, info: ValidationInfo) -> object:
        if isinstance(data, dict) and info.context and info.context.get(STRICT_CONTEXT_KEY):
            unknown = sorted(set(data) - set(cls.model_fields))
            if unknown:
                raise ValueError(f"{cls.__name__} has unknown fields {unknown}")
        return data


class LogicalType(str, Enum):
    """Engine-neutral logical column type covering the Spark and Arrow type families."""

    NULL = "null"
    BOOLEAN = "boolean"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    DECIMAL = "decimal"
    STRING = "string"
    BINARY = "binary"
    DATE = "date"
    TIME = "time"
    TIMESTAMP = "timestamp"
    TIMESTAMP_NTZ = "timestamp_ntz"
    INTERVAL = "interval"
    STRUCT = "struct"
    ARRAY = "array"
    MAP = "map"
    VARIANT = "variant"


EXACT_INTEGER_TYPES = frozenset(
    {
        LogicalType.INT8,
        LogicalType.INT16,
        LogicalType.INT32,
        LogicalType.UINT8,
        LogicalType.UINT16,
        LogicalType.UINT32,
    }
)
FLOAT_TYPES = frozenset({LogicalType.FLOAT32, LogicalType.FLOAT64})
STRING_ENCODED_TYPES = frozenset({LogicalType.INT64, LogicalType.UINT64, LogicalType.DECIMAL})
NUMERIC_TYPES = EXACT_INTEGER_TYPES | FLOAT_TYPES | STRING_ENCODED_TYPES
TEMPORAL_TYPES = frozenset({LogicalType.DATE, LogicalType.TIME, LogicalType.TIMESTAMP, LogicalType.TIMESTAMP_NTZ})
COLLECTION_TYPES = frozenset({LogicalType.ARRAY, LogicalType.MAP})
UNSIGNED_TYPES = frozenset({LogicalType.UINT8, LogicalType.UINT16, LogicalType.UINT32, LogicalType.UINT64})
MAX_UTC_OFFSET = timedelta(hours=18)

_STRING_LITERAL_RES = {
    LogicalType.DECIMAL: _DECIMAL_RE,
    LogicalType.INT64: _INTEGER_RE,
    LogicalType.UINT64: _UNSIGNED_RE,
    LogicalType.DATE: re.compile(rf"^{_DATE}$"),
    LogicalType.TIME: re.compile(rf"^{_TIME}$"),
    LogicalType.TIMESTAMP: re.compile(rf"^{_DATE}T{_TIME}(Z|[+-]\d{{2}}:\d{{2}})$"),
    LogicalType.TIMESTAMP_NTZ: re.compile(rf"^{_DATE}T{_TIME}$"),
    LogicalType.STRING: re.compile(r".*", re.DOTALL),
}
_JSON_LITERAL_TYPES: dict[LogicalType, tuple[type, ...]] = {
    **dict.fromkeys(EXACT_INTEGER_TYPES, (int,)),
    **dict.fromkeys(FLOAT_TYPES, (int, float)),
    LogicalType.BOOLEAN: (bool,),
}


def is_valid_literal(value: object, column: ColumnProfile) -> bool:
    """Returns whether `value` is a correctly encoded literal for the column's logical type.

    Args:
        value: Literal value taken from parsed JSON.
        column: Column the value belongs to.

    Returns:
        True when `value` follows the spec's encoding rule for the column's logical
        type, or when the spec defines no rule for the type.
    """
    logical_type = column.logical_type
    pattern = _STRING_LITERAL_RES.get(logical_type)
    if pattern is not None:
        return isinstance(value, str) and pattern.fullmatch(value) is not None and _is_in_domain(value, column)
    types = _JSON_LITERAL_TYPES.get(logical_type)
    if types is None:
        return True
    is_bool = isinstance(value, bool)
    if logical_type in UNSIGNED_TYPES:
        return isinstance(value, int) and not is_bool and value >= 0
    return isinstance(value, types) and is_bool == (logical_type == LogicalType.BOOLEAN)


def _is_in_domain(value: str, column: ColumnProfile) -> bool:
    """Returns whether a well-formed string literal fits the column's decimal precision or is a real date or time.

    Args:
        value: String literal that already matches the format for the column's logical type.
        column: Column the value belongs to.

    Returns:
        True when a decimal fits `precision` and `scale`, a temporal value parses to a real date or time with a UTC
        offset within 18 hours, or the type has no further rule.
    """
    if column.logical_type == LogicalType.DECIMAL and column.precision is not None and column.scale is not None:
        whole, _, fraction = value.lstrip("-").partition(".")
        whole_digits = 0 if whole == "0" else len(whole)
        return len(fraction) <= column.scale and whole_digits <= column.precision - column.scale
    if column.logical_type in TEMPORAL_TYPES:
        try:
            parsed = parse_temporal(value, column.logical_type)
        except ValueError:
            return False
        offset = parsed.utcoffset() if isinstance(parsed, datetime) else None
        return offset is None or abs(offset) <= MAX_UTC_OFFSET
    return True


def canonicalize_iso_fraction(value: str) -> str:
    """Returns `value` with fractional seconds in the producer canonical form.

    When a sub-second component is present and non-zero, it is normalized to exactly
    six digits (microseconds; longer fractions are truncated). An all-zero fraction is
    omitted. This matches Spark's microsecond timestamps and avoids Python 3.10's
    `fromisoformat` rejecting odd-length fractions such as Spark's cast-to-string output.
    """
    match = _ISO_FRACTION_ANY_RE.search(value)
    if match is None:
        return value
    digits = match.group(1)[1:]
    if int(digits) == 0:
        return value[: match.start()] + value[match.end() :]
    normalized = (digits + "000000")[:6]
    return f"{value[: match.start()]}.{normalized}{value[match.end() :]}"


def _normalize_iso_fraction_for_parse(value: str) -> str:
    """Pads or truncates fractional seconds to six digits for Python 3.10 `fromisoformat`."""
    match = _ISO_FRACTION_RE.search(value)
    if match is None:
        return value
    digits = match.group(1)[1:]
    if len(digits) == 6:
        return value
    digits = digits.ljust(6, "0") if len(digits) < 6 else digits[:6]
    return f"{value[: match.start(1)]}.{digits}{value[match.end(1) :]}"


def parse_temporal(value: str, logical_type: LogicalType) -> date | time | datetime:
    """Parses an ISO-8601 date, time, or timestamp literal from a profile.

    Accepts 1-9 fractional digits (the spec maximum) on Python 3.10 by normalizing to
    microseconds before calling `fromisoformat`. `Z` is treated as UTC. Sub-microsecond
    digits beyond six are truncated.

    Raises:
        ValueError: If `value` is not a real date, time, or timestamp for `logical_type`.
    """
    if logical_type == LogicalType.DATE:
        return date.fromisoformat(value)
    if logical_type == LogicalType.TIME:
        return time.fromisoformat(_normalize_iso_fraction_for_parse(value))
    normalized = _normalize_iso_fraction_for_parse(value.replace("Z", "+00:00"))
    return datetime.fromisoformat(normalized)


def _to_decimal(value: Scalar) -> Decimal:
    """Converts a JSON number or canonical decimal string to a `Decimal`.

    Raises:
        ValueError: If `value` is a string that is not a canonical decimal.
    """
    if isinstance(value, bool) or (isinstance(value, str) and _DECIMAL_RE.fullmatch(value) is None):
        raise ValueError(f"{value!r} is not a number or canonical decimal string")
    return Decimal(str(value))


class Cardinality(str, Enum):
    """Bucket for the number of distinct non-null values in a column.

    Buckets are checked in this order: `unique` when every non-null value is
    distinct, `constant` for exactly one distinct value, `low` for at most 100
    distinct values, `high` when distinct values exceed 10% of non-null rows, and
    `medium` otherwise.
    """

    CONSTANT = "constant"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNIQUE = "unique"


class DistinctCount(_SpecModel):
    """Number of distinct non-null values in a column.

    Attributes:
        value: Distinct value count.
        exact: Whether the count is exact (true) or an approximation such as HyperLogLog (false).
    """

    value: Count
    exact: bool


class Quantile(_SpecModel):
    """Value at a given quantile of the non-null values.

    Attributes:
        q: Quantile in [0, 1], e.g. 0.5 for the median.
        value: Value at the quantile, encoded by the column's logical type.
    """

    q: Fraction
    value: Scalar


class Histogram(_SpecModel):
    """Histogram over the non-null values of a numeric column.

    Bin `i` holds values in `[edges[i], edges[i + 1])`; the last bin also holds
    `edges[-1]`. The edges span every profiled value, so no value falls outside
    the bins and the counts add up to the number of non-null values profiled.

    Attributes:
        edges: Non-decreasing bin edges, encoded by the column's logical type; one more than the number of bins.
        counts: Number of values in each bin.
        sampled: Whether the counts come from a sample rather than every row (default false).
        generalized_outer_edges: Whether the first and last edges were rounded or widened rather than being the exact
            data extremes (default false). Must be true when `privacy.extremes` is "suppressed".
    """

    edges: list[Scalar] = Field(min_length=2)
    counts: list[Count]
    sampled: bool = False
    generalized_outer_edges: bool = False

    @model_validator(mode="after")
    def _check_shape(self) -> Histogram:
        if len(self.counts) != len(self.edges) - 1:
            raise ValueError("Histogram.counts must have exactly one fewer entry than Histogram.edges")
        edges = [_to_decimal(edge) for edge in self.edges]
        if any(left > right for left, right in pairwise(edges)):
            raise ValueError("Histogram.edges must be non-decreasing")
        return self


class NumericStats(_SpecModel):
    """Distribution statistics for an integer, floating-point, or decimal column.

    Attributes:
        min: Optional smallest value, encoded by the column's logical type.
        max: Optional largest value, encoded by the column's logical type.
        mean: Optional arithmetic mean, as an approximate JSON number.
        stddev: Optional sample standard deviation, as an approximate JSON number.
        quantiles: Optional list of quantiles.
        histogram: Optional histogram.
    """

    min: Scalar | None = None
    max: Scalar | None = None
    mean: float | None = None
    stddev: Annotated[float, Field(ge=0.0)] | None = None
    quantiles: list[Quantile] | None = None
    histogram: Histogram | None = None


class LengthStats(_SpecModel):
    """Distribution of lengths: characters for strings, entries for arrays and maps.

    Attributes:
        min: Optional shortest length.
        max: Optional longest length.
        mean: Optional mean length.
        stddev: Optional sample standard deviation of the length.
        quantiles: Optional list of length quantiles.
    """

    min: Count | None = None
    max: Count | None = None
    mean: Annotated[float, Field(ge=0.0)] | None = None
    stddev: Annotated[float, Field(ge=0.0)] | None = None
    quantiles: list[Quantile] | None = None


class Pattern(_SpecModel):
    """Regular expression that matches part of a string column.

    Attributes:
        regex: Regular expression that a matching value satisfies in full.
        coverage: Fraction of non-null values that match the expression; must cover at least
            `privacy.min_value_count` rows.
    """

    regex: str
    coverage: Fraction


class ValueFrequency(_SpecModel):
    """Frequent value and its share of the column.

    Attributes:
        value: The value.
        frequency: Fraction of non-null values equal to `value`.
    """

    value: str
    frequency: Fraction


class StringStats(_SpecModel):
    """Statistics for a string column.

    Attributes:
        length: Optional distribution of value lengths.
        patterns: Optional regular expressions with the fraction of values each one matches.
        top_values: Optional most frequent values; each value must occur in at
            least `privacy.min_value_count` rows.
    """

    length: LengthStats | None = None
    patterns: list[Pattern] | None = None
    top_values: list[ValueFrequency] | None = None


class TemporalStats(_SpecModel):
    """Statistics for a date, time, or timestamp column.

    Values are ISO-8601 strings: `YYYY-MM-DD` for `date`, `HH:MM:SS[.f]` for
    `time`, `YYYY-MM-DDTHH:MM:SS[.f]` followed by `Z` or `+HH:MM` for
    `timestamp`, and the same without an offset for `timestamp_ntz`. Fractions
    have up to nine digits. Producers emit exactly six fractional digits when the
    value has a non-zero sub-second component and omit the fraction otherwise.

    Attributes:
        min: Optional earliest value.
        max: Optional latest value.
        quantiles: Optional list of quantiles.
    """

    min: str | None = None
    max: str | None = None
    quantiles: list[Quantile] | None = None


class BooleanStats(_SpecModel):
    """Statistics for a boolean column.

    Attributes:
        true_fraction: Fraction of non-null values that are true.
    """

    true_fraction: Fraction


class PiiClass(str, Enum):
    """Semantic class of personally identifying data."""

    EMAIL = "email"
    PHONE = "phone"
    PERSON_NAME = "person_name"
    STREET_ADDRESS = "street_address"
    CITY = "city"
    POSTAL_CODE = "postal_code"
    NATIONAL_ID = "national_id"
    CREDIT_CARD = "credit_card"
    IBAN = "iban"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    ACCOUNT_ID = "account_id"
    SECRET_TOKEN = "secret_token"
    FREE_TEXT_PII = "free_text_pii"
    OTHER = "other"


class Sensitivity(str, Enum):
    """Privacy sensitivity taxonomy shared with the benchmark catalog."""

    DIRECT_IDENTIFIER = "direct_identifier"
    QUASI_IDENTIFIER = "quasi_identifier"
    SECRET = "secret"
    FREE_TEXT = "free_text"


class SemanticEvidence(_SpecModel):
    """Detection method and its sampled match rate, without source values."""

    method: str = Field(min_length=1)
    match_rate: Fraction

    @model_validator(mode="before")
    @classmethod
    def _forbid_value_fields(cls, data: object) -> object:
        if isinstance(data, dict) and set(data) != {"method", "match_rate"}:
            raise ValueError("semantic evidence contains only method and match_rate")
        return data


class Semantic(_SpecModel):
    """Detected semantic meaning and sensitivity of a column."""

    pii_class: PiiClass
    sensitivity: Sensitivity
    confidence: Fraction
    evidence: list[SemanticEvidence] = Field(default_factory=list)


class ColumnProfile(_SpecModel):
    """Profile of a single column, or of a field nested inside a complex column.

    Nested types use `children`: a struct lists its fields, an array has one child
    named "element", and a map has two children named "key" and "value". Child
    statistics are computed over all nested values, not over rows.

    Attributes:
        name: Column or field name.
        logical_type: Engine-neutral logical type.
        source_type: Type name as reported by the source system, e.g. "decimal(10,2)".
        nullable: Whether the schema allows nulls.
        null_fraction: Fraction of values that are null.
        precision: Total digits; required for `decimal` and absent otherwise.
        scale: Digits after the decimal point; required for `decimal` and absent otherwise.
        children: Nested field profiles; required for `struct`, `array`, and `map`, absent otherwise.
        distinct_count: Optional number of distinct non-null values.
        cardinality: Optional cardinality bucket.
        numeric: Optional statistics for numeric types.
        string: Optional statistics for `string`.
        temporal: Optional statistics for `date`, `time`, `timestamp`, and `timestamp_ntz`.
        boolean: Optional statistics for `boolean`.
        collection_length: Optional distribution of entry counts for non-null `array` and `map` values.
        value_count: Optional number of values, including nulls, that a nested column's statistics cover. Only valid
            on nested columns; required on `array` and `map` children, and on their descendants, that carry
            `patterns` or `top_values`. Struct fields without it inherit their parent's count.
    """

    name: str = Field(min_length=1)
    logical_type: LogicalType
    source_type: str
    nullable: bool
    null_fraction: Fraction
    precision: Annotated[int, Field(ge=1)] | None = None
    scale: Count | None = None
    children: list[ColumnProfile] | None = None
    distinct_count: DistinctCount | None = None
    cardinality: Cardinality | None = None
    numeric: NumericStats | None = None
    string: StringStats | None = None
    temporal: TemporalStats | None = None
    boolean: BooleanStats | None = None
    collection_length: LengthStats | None = None
    value_count: Count | None = None
    semantic: Semantic | None = None

    @model_validator(mode="after")
    def _check_type_consistency(self) -> ColumnProfile:
        kind = self.logical_type
        if not self.nullable and self.null_fraction > 0:
            raise ValueError(f"column {self.name!r} is not nullable but has null_fraction={self.null_fraction}")
        if (kind == LogicalType.DECIMAL) != (self.precision is not None and self.scale is not None):
            raise ValueError(f"column {self.name!r}: precision and scale are required for decimal and only decimal")
        if self.precision is not None and self.scale is not None and self.scale > self.precision:
            raise ValueError(f"column {self.name!r}: scale must not exceed precision")
        _check_children(self)
        allowed = {
            "numeric": kind in NUMERIC_TYPES,
            "string": kind == LogicalType.STRING,
            "temporal": kind in TEMPORAL_TYPES,
            "boolean": kind == LogicalType.BOOLEAN,
            "collection_length": kind in COLLECTION_TYPES,
        }
        for block, ok in allowed.items():
            if getattr(self, block) is not None and not ok:
                raise ValueError(f"column {self.name!r}: {block} stats are not valid for logical_type={kind.value}")
        check_literals(self, self.literals(), "statistics")
        return self

    def literals(self) -> list[Scalar]:
        """Returns the literal values held in the numeric and temporal statistics."""
        stats = self.numeric or self.temporal
        if stats is None:
            return []
        values = [stats.min, stats.max, *(q.value for q in stats.quantiles or [])]
        if self.numeric is not None and self.numeric.histogram is not None:
            values.extend(self.numeric.histogram.edges)
        return [value for value in values if value is not None]


def check_literals(column: ColumnProfile, values: list[Scalar], where: str) -> None:
    """Checks that literal values follow the encoding rule for the column's logical type.

    Args:
        column: Column the values belong to.
        values: Literal values to check.
        where: Name of the block holding the values, used in the error message.

    Raises:
        ValueError: If any value is not encoded correctly.
    """
    invalid = [value for value in values if not is_valid_literal(value, column)]
    if invalid:
        raise ValueError(
            f"column {column.name!r}: {where} values {invalid} are not valid {column.logical_type.value} literals"
        )


def _check_children(column: ColumnProfile) -> None:
    """Checks that `children` matches the column's logical type.

    Args:
        column: Column to check.

    Raises:
        ValueError: If the children are missing, unexpected, or misnamed.
    """
    names = [child.name for child in column.children or []]
    expected = {LogicalType.ARRAY: ["element"], LogicalType.MAP: ["key", "value"]}.get(column.logical_type)
    if column.logical_type == LogicalType.STRUCT:
        if not names or len(set(names)) != len(names):
            raise ValueError(f"column {column.name!r}: struct needs at least one child, with unique names")
    elif expected is not None:
        if names != expected:
            raise ValueError(f"column {column.name!r}: {column.logical_type.value} children must be named {expected}")
    elif column.children is not None:
        raise ValueError(f"column {column.name!r}: children are only valid for struct, array, and map")


def counted_columns(column: ColumnProfile, value_count: int | None) -> Iterator[tuple[ColumnProfile, int | None]]:
    """Yields a column and every column nested below it, each with the number of values its statistics cover.

    Args:
        column: Column to start from.
        value_count: Number of values `column` covers: the table row count for a top-level column.

    Yields:
        `(column, value_count)` pairs. A child's own `value_count` wins; otherwise struct fields inherit their
        parent's count and array or map children get None.
    """
    yield column, value_count
    inherited = value_count if column.logical_type == LogicalType.STRUCT else None
    for child in column.children or []:
        yield from counted_columns(child, inherited if child.value_count is None else child.value_count)


class KeyCandidate(_SpecModel):
    """Set of columns that is likely unique across the table.

    Attributes:
        columns: Distinct column names that form the key, in order.
        confidence: Confidence that the columns form a key, in [0, 1].
    """

    columns: list[str] = Field(min_length=1)
    confidence: Fraction

    @model_validator(mode="after")
    def _check_distinct_columns(self) -> KeyCandidate:
        if len(set(self.columns)) != len(self.columns):
            raise ValueError(f"key candidate repeats a column: {self.columns}")
        return self


class _Constraint(_SpecModel):
    """Common fields of every check constraint.

    Attributes:
        satisfied_fraction: Fraction of rows that satisfy the constraint.
    """

    satisfied_fraction: Fraction

    def column_refs(self) -> list[str]:
        """Returns the column names the constraint refers to."""
        return []


class RangeConstraint(_Constraint):
    """Bounds a column's values to an inclusive range.

    Attributes:
        kind: Discriminator; always "range".
        column: Constrained column.
        min: Optional inclusive lower bound, encoded by the column's logical type.
        max: Optional inclusive upper bound, encoded by the column's logical type.
        satisfied_fraction: Fraction of rows that satisfy the constraint.
    """

    kind: Literal["range"] = "range"
    column: str
    min: Scalar | None = None
    max: Scalar | None = None

    def column_refs(self) -> list[str]:
        """Returns the column names the constraint refers to."""
        return [self.column]


class EnumerationConstraint(_Constraint):
    """Limits a column to a fixed set of values.

    Attributes:
        kind: Discriminator; always "enumeration".
        column: Constrained column.
        values: Allowed values, encoded by the column's logical type.
        min_value_count: Number of rows holding the least frequent listed value; must be at least
            `privacy.min_value_count`.
        satisfied_fraction: Fraction of rows that satisfy the constraint.
    """

    kind: Literal["enumeration"] = "enumeration"
    column: str
    values: list[Scalar] = Field(min_length=1)
    min_value_count: Count

    def column_refs(self) -> list[str]:
        """Returns the column names the constraint refers to."""
        return [self.column]


class ComparisonConstraint(_Constraint):
    """Compares two columns of the same row, as `left <operator> right`.

    Attributes:
        kind: Discriminator; always "comparison".
        left: Left-hand column.
        operator: Comparison operator.
        right: Right-hand column.
        satisfied_fraction: Fraction of rows that satisfy the constraint.
    """

    kind: Literal["comparison"] = "comparison"
    left: str
    operator: Literal["<", "<=", "=", "!=", ">=", ">"]
    right: str

    def column_refs(self) -> list[str]:
        """Returns the column names the constraint refers to."""
        return [self.left, self.right]


class ExpressionConstraint(_Constraint):
    """Boolean SQL expression that rows satisfy.

    Attributes:
        kind: Discriminator; always "expression".
        expression: Boolean SQL expression over the table's columns.
        dialect: SQL dialect of `expression` (default "spark").
        satisfied_fraction: Fraction of rows that satisfy the constraint.
    """

    kind: Literal["expression"] = "expression"
    expression: str = Field(min_length=1)
    dialect: str = "spark"


CheckConstraint = Annotated[
    RangeConstraint | EnumerationConstraint | ComparisonConstraint | ExpressionConstraint,
    Field(discriminator="kind"),
]


class DependenceMethod(str, Enum):
    """Measure used to quantify the dependence between two columns."""

    PEARSON = "pearson"
    SPEARMAN = "spearman"
    CRAMERS_V = "cramers_v"
    CORRELATION_RATIO = "correlation_ratio"


class Dependency(_SpecModel):
    """Measured dependence between two columns of a table.

    Attributes:
        columns: The two column names.
        method: Dependence measure.
        value: Measured value; in [-1, 1] for `pearson` and `spearman` and in [0, 1] otherwise.
    """

    columns: tuple[str, str]
    method: DependenceMethod
    value: float

    @model_validator(mode="after")
    def _check_value_range(self) -> Dependency:
        signed = self.method in (DependenceMethod.PEARSON, DependenceMethod.SPEARMAN)
        low = -1.0 if signed else 0.0
        if not low <= self.value <= 1.0:
            raise ValueError(f"Dependency.value={self.value} is out of range for method={self.method.value}")
        return self


class QuasiIdentifierSet(_SpecModel):
    """Sampled uniqueness and k-anonymity estimate for a column combination."""

    columns: list[str] = Field(min_length=2, max_length=3)
    unique_fraction: Fraction
    k_anonymity: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def _check_columns(self) -> QuasiIdentifierSet:
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("quasi-identifier set repeats a column")
        return self


class TableProfile(_SpecModel):
    """Profile of a single table.

    Attributes:
        name: Table name, unique within the profile.
        row_count: Number of rows profiled.
        columns: Top-level column profiles, in schema order; at least one.
        primary_key: Optional best primary key candidate.
        unique_keys: Optional other unique key candidates.
        check_constraints: Optional row-level constraints found in the data.
        dependencies: Optional pairwise dependence measures between columns.
    """

    name: str = Field(min_length=1)
    row_count: Count
    columns: list[ColumnProfile] = Field(min_length=1)
    primary_key: KeyCandidate | None = None
    unique_keys: list[KeyCandidate] | None = None
    check_constraints: list[CheckConstraint] | None = None
    dependencies: list[Dependency] | None = None
    quasi_identifier_sets: list[QuasiIdentifierSet] | None = None

    @model_validator(mode="after")
    def _check_column_refs(self) -> TableProfile:
        columns = {column.name: column for column in self.columns}
        if len(columns) != len(self.columns):
            raise ValueError(f"table {self.name!r} has duplicate column names")
        keys = [self.primary_key] if self.primary_key else []
        refs = [
            *(name for key in [*keys, *(self.unique_keys or [])] for name in key.columns),
            *(name for constraint in self.check_constraints or [] for name in constraint.column_refs()),
            *(name for dependency in self.dependencies or [] for name in dependency.columns),
            *(name for item in self.quasi_identifier_sets or [] for name in item.columns),
        ]
        unknown = sorted(set(refs) - set(columns))
        if unknown:
            raise ValueError(f"table {self.name!r} refers to unknown columns {unknown}")
        for constraint in self.check_constraints or []:
            if isinstance(constraint, RangeConstraint):
                bounds = [bound for bound in (constraint.min, constraint.max) if bound is not None]
                check_literals(columns[constraint.column], bounds, "range constraint")
            elif isinstance(constraint, EnumerationConstraint):
                check_literals(columns[constraint.column], constraint.values, "enumeration constraint")
        return self


class FanOutStats(_SpecModel):
    """Distribution of the number of child rows per parent key.

    Every parent key counts, including keys with no child rows. Child rows whose
    key matches no parent are left out; `Relationship.coverage` accounts for them.
    Quantile values are integer row counts.

    Attributes:
        min: Fewest child rows for a parent key.
        max: Most child rows for a parent key.
        mean: Mean child rows per parent key.
        stddev: Optional sample standard deviation.
        quantiles: Optional list of quantiles.
    """

    min: Count
    max: Count
    mean: Annotated[float, Field(ge=0.0)]
    stddev: Annotated[float, Field(ge=0.0)] | None = None
    quantiles: list[Quantile] | None = None

    @model_validator(mode="after")
    def _check_integer_quantiles(self) -> FanOutStats:
        if any(isinstance(q.value, bool) or not isinstance(q.value, int) for q in self.quantiles or []):
            raise ValueError("FanOutStats.quantiles values must be integer row counts")
        return self


class Relationship(_SpecModel):
    """Foreign key from child columns to parent key columns.

    Attributes:
        child_table: Table that holds the foreign key.
        child_columns: Foreign key columns, in order.
        parent_table: Referenced table.
        parent_columns: Referenced key columns, matched to `child_columns` by position.
        coverage: Fraction of non-null child keys that match a parent key.
        confidence: Optional confidence that the relationship is a true foreign key, in [0, 1].
        fan_out: Optional distribution of child rows per parent key.
    """

    child_table: str
    child_columns: list[str] = Field(min_length=1)
    parent_table: str
    parent_columns: list[str] = Field(min_length=1)
    coverage: Fraction
    confidence: Fraction | None = None
    fan_out: FanOutStats | None = None

    @model_validator(mode="after")
    def _check_arity(self) -> Relationship:
        if len(self.child_columns) != len(self.parent_columns):
            raise ValueError("Relationship.child_columns and parent_columns must have the same length")
        return self


class ValueTreatment(str, Enum):
    """How a producer handled a class of raw values."""

    EXACT = "exact"
    GENERALIZED = "generalized"
    SUPPRESSED = "suppressed"


class PrivacyPolicy(_SpecModel):
    """Privacy policy that the producer applied when writing the profile.

    Attributes:
        min_value_count: Fewest rows a literal value must occur in before it may appear in `top_values` or an
            enumeration constraint, and fewest rows a string pattern must match (default 10). Literals embedded in
            regexes are checked by the benchmark's privacy guard, not by this model.
        extremes: Treatment of data extremes (default "exact"). When "suppressed", numeric and temporal `min` and
            `max`, quantiles at q=0 and q=1, and range constraint bounds may not appear, and histograms must set
            `generalized_outer_edges`.
        enumerations: Treatment of `top_values` and enumeration constraints (default "exact"). When "suppressed",
            none of them may appear.
        allow_expression_constraints: Whether SQL expression constraints, which may embed literal values, may
            appear (default false).
    """

    min_value_count: Annotated[int, Field(ge=1)] = DEFAULT_MIN_VALUE_COUNT
    extremes: ValueTreatment = ValueTreatment.EXACT
    enumerations: ValueTreatment = ValueTreatment.EXACT
    allow_expression_constraints: bool = False


class EngineInfo(_SpecModel):
    """Engine that produced the profile.

    Attributes:
        name: Engine name, e.g. "pyspark".
        version: Engine version.
    """

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class SamplingInfo(_SpecModel):
    """How rows were sampled before profiling.

    Attributes:
        fraction: Optional fraction of rows sampled.
        max_rows: Optional cap on the number of rows sampled.
        seed: Optional random seed.
    """

    fraction: Fraction | None = None
    max_rows: Count | None = None
    seed: int | None = None


class SourceInfo(_SpecModel):
    """Metadata about the profiled source. Holds no data values.

    Attributes:
        name: Optional dataset name.
        format: Optional storage format, e.g. "delta" or "parquet".
        location: Optional path, URI, or catalog name of the source.
        sampling: Optional sampling applied before profiling.
    """

    name: str | None = None
    format: str | None = None
    location: str | None = None
    sampling: SamplingInfo | None = None


class Profile(_SpecModel):
    """Profile of one or more tables and the relationships between them.

    Attributes:
        spec_version: Version of this spec that the profile follows, as "major.minor.patch". The major version must
            match the reader's.
        created_at: Time the profile was created, with a UTC offset.
        engine: Engine that produced the profile.
        privacy: Privacy policy the producer applied (default: the `PrivacyPolicy` defaults).
        source: Optional metadata about the profiled source.
        tables: Table profiles.
        relationships: Optional foreign key relationships between tables.
    """

    spec_version: str = Field(pattern=rf"^{SUPPORTED_MAJOR_VERSION}\.\d+\.\d+$")
    created_at: AwareDatetime
    engine: EngineInfo
    privacy: PrivacyPolicy = Field(default_factory=PrivacyPolicy)
    source: SourceInfo | None = None
    tables: list[TableProfile] = Field(min_length=1)
    relationships: list[Relationship] | None = None

    @model_validator(mode="after")
    def _check_relationship_refs(self) -> Profile:
        columns = {table.name: {column.name for column in table.columns} for table in self.tables}
        if len(columns) != len(self.tables):
            raise ValueError("profile has duplicate table names")
        for rel in self.relationships or []:
            for table, refs in ((rel.child_table, rel.child_columns), (rel.parent_table, rel.parent_columns)):
                if table not in columns:
                    raise ValueError(f"relationship refers to unknown table {table!r}")
                unknown = sorted(set(refs) - columns[table])
                if unknown:
                    raise ValueError(f"relationship refers to unknown columns {unknown} in table {table!r}")
        return self

    @model_validator(mode="after")
    def _check_privacy(self) -> Profile:
        for table in self.tables:
            _check_table_privacy(table, self.privacy)
        return self


def _check_table_privacy(table: TableProfile, policy: PrivacyPolicy) -> None:
    """Checks that a table's literal values follow the privacy policy.

    Args:
        table: Table to check.
        policy: Privacy policy the producer declared.

    Raises:
        ValueError: If a literal value or pattern covers fewer than `policy.min_value_count` rows, or a suppressed or
            disallowed value class is present.
    """
    k = policy.min_value_count
    suppress_extremes = policy.extremes == ValueTreatment.SUPPRESSED
    suppress_enumerations = policy.enumerations == ValueTreatment.SUPPRESSED
    if any(column.value_count is not None for column in table.columns):
        raise ValueError(f"table {table.name!r}: value_count is only valid on nested columns")
    for top_level in table.columns:
        for column, value_count in counted_columns(top_level, table.row_count):
            _check_column_privacy(column, value_count, policy)
    for constraint in table.check_constraints or []:
        if isinstance(constraint, RangeConstraint) and suppress_extremes:
            raise ValueError(f"table {table.name!r}: range constraint present but privacy.extremes is suppressed")
        if isinstance(constraint, ExpressionConstraint) and not policy.allow_expression_constraints:
            raise ValueError(f"table {table.name!r}: expression constraint present but not allowed by privacy policy")
        if isinstance(constraint, EnumerationConstraint):
            if suppress_enumerations:
                raise ValueError(f"table {table.name!r}: enumeration present but privacy.enumerations is suppressed")
            if constraint.min_value_count < k:
                raise ValueError(
                    f"table {table.name!r}: enumeration on {constraint.column!r} has values in fewer than {k} rows"
                )


def _check_column_privacy(column: ColumnProfile, value_count: int | None, policy: PrivacyPolicy) -> None:
    """Checks that a column's statistics follow the privacy policy.

    Args:
        column: Column to check.
        value_count: Number of values the column's statistics cover, or None when it is unknown.
        policy: Privacy policy the producer declared.

    Raises:
        ValueError: If a value or pattern covers fewer than `policy.min_value_count` values, literal-bearing
            statistics have no known value count, or a suppressed value class is present.
    """
    k = policy.min_value_count
    if column.semantic is not None and column.semantic.sensitivity in {
        Sensitivity.DIRECT_IDENTIFIER,
        Sensitivity.SECRET,
    }:
        string = column.string or StringStats()
        if column.literals() or string.top_values:
            raise ValueError(f"column {column.name!r}: direct identifiers and secrets cannot carry literal statistics")
        if string.patterns and any(not _is_generalized_pattern(pattern.regex) for pattern in string.patterns):
            raise ValueError(f"column {column.name!r}: sensitive patterns must contain character classes only")
    stats = column.numeric or column.temporal
    if policy.extremes == ValueTreatment.SUPPRESSED and stats is not None:
        histogram = column.numeric.histogram if column.numeric else None
        if stats.min is not None or stats.max is not None:
            raise ValueError(f"column {column.name!r}: min/max present but privacy.extremes is suppressed")
        if any(q.q in (0.0, 1.0) for q in stats.quantiles or []):
            raise ValueError(f"column {column.name!r}: q=0 or q=1 quantile present but privacy.extremes is suppressed")
        if histogram is not None and not histogram.generalized_outer_edges:
            raise ValueError(
                f"column {column.name!r}: histogram outer edges must be generalized when extremes are suppressed"
            )
    string = column.string or StringStats()
    if string.top_values and policy.enumerations == ValueTreatment.SUPPRESSED:
        raise ValueError(f"column {column.name!r}: top_values present but privacy.enumerations is suppressed")
    if value_count is None:
        if string.top_values or string.patterns:
            raise ValueError(f"column {column.name!r}: nested patterns or top_values require value_count")
        return
    non_null_rows = value_count * (1.0 - column.null_fraction)
    rare = [entry.value for entry in string.top_values or [] if round(entry.frequency * non_null_rows) < k]
    rare += [pattern.regex for pattern in string.patterns or [] if round(pattern.coverage * non_null_rows) < k]
    if rare:
        raise ValueError(f"column {column.name!r}: top_values or patterns cover fewer than {k} rows: {rare}")


def _is_generalized_pattern(regex: str) -> bool:
    """Returns whether a regex consists only of character classes and structural operators."""
    stripped = re.sub(r"\\[dDsSwW]|\[[^]]+\]|\{\d+(,\d*)?\}|[.*+?^$|()\\-]", "", regex)
    return re.search(r"[A-Za-z0-9]", stripped) is None
