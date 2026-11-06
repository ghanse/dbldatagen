"""Pydantic models for the dbldatagen profile spec.

A `Profile` describes one or more tables and the relationships between them
without holding raw rows. It is the contract between profiling engines and
consumers such as data generators and the benchmark's reference sampler. Every
block except the basic schema, row count, and null information is optional so
engines can fill it in over time.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


SPEC_VERSION = "0.1.0"
LENIENT_CONTEXT_KEY = "lenient"

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Count = Annotated[int, Field(ge=0)]
Number = int | float
Scalar = int | float | str


def parse_version(version: str) -> tuple[int, int, int]:
    """Parses a `major.minor.patch` version string.

    Args:
        version: Version string such as "0.1.0".

    Returns:
        A `(major, minor, patch)` tuple of integers.

    Raises:
        ValueError: If `version` is not three dot-separated integers.
    """
    match = _SEMVER_RE.fullmatch(version)
    if match is None:
        raise ValueError(f"spec_version={version!r} is not a 'major.minor.patch' version")
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


class _SpecModel(BaseModel):
    """Base model for every spec type.

    Unknown fields are rejected, except when validation runs with the lenient
    context flag set, in which case they are dropped. Readers set the flag when a
    profile declares a newer minor spec version than they implement.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_unknown_fields(cls, data: object, info: ValidationInfo) -> object:
        if isinstance(data, dict) and info.context and info.context.get(LENIENT_CONTEXT_KEY):
            return {key: value for key, value in data.items() if key in cls.model_fields}
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


NUMERIC_TYPES = frozenset(
    {
        LogicalType.INT8,
        LogicalType.INT16,
        LogicalType.INT32,
        LogicalType.INT64,
        LogicalType.UINT8,
        LogicalType.UINT16,
        LogicalType.UINT32,
        LogicalType.UINT64,
        LogicalType.FLOAT32,
        LogicalType.FLOAT64,
        LogicalType.DECIMAL,
    }
)
TEMPORAL_TYPES = frozenset({LogicalType.DATE, LogicalType.TIME, LogicalType.TIMESTAMP, LogicalType.TIMESTAMP_NTZ})


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
        value: Value at the quantile; an ISO-8601 string for temporal columns.
    """

    q: Fraction
    value: Scalar


class Histogram(_SpecModel):
    """Equal- or variable-width histogram over the non-null values.

    Attributes:
        edges: Non-decreasing bin edges; one more than the number of bins.
        counts: Number of values in each bin.
    """

    edges: list[float] = Field(min_length=2)
    counts: list[Count]

    @model_validator(mode="after")
    def _check_shape(self) -> Histogram:
        if len(self.counts) != len(self.edges) - 1:
            raise ValueError("Histogram.counts must have exactly one fewer entry than Histogram.edges")
        if any(left > right for left, right in zip(self.edges, self.edges[1:], strict=False)):
            raise ValueError("Histogram.edges must be non-decreasing")
        return self


class NumericStats(_SpecModel):
    """Distribution statistics for an integer, floating-point, or decimal column.

    Attributes:
        min: Optional smallest value.
        max: Optional largest value.
        mean: Optional arithmetic mean.
        stddev: Optional sample standard deviation.
        quantiles: Optional list of quantiles.
        histogram: Optional histogram.
    """

    min: Number | None = None
    max: Number | None = None
    mean: float | None = None
    stddev: Annotated[float, Field(ge=0.0)] | None = None
    quantiles: list[Quantile] | None = None
    histogram: Histogram | None = None


class LengthStats(_SpecModel):
    """Distribution of value lengths, in characters.

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
        coverage: Fraction of non-null values that match the expression.
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
        top_values: Optional most frequent values. Engines must leave out rare values that could identify a record.
    """

    length: LengthStats | None = None
    patterns: list[Pattern] | None = None
    top_values: list[ValueFrequency] | None = None


class TemporalStats(_SpecModel):
    """Statistics for a date, time, or timestamp column.

    Values are ISO-8601 strings. `timestamp` values carry a UTC offset and
    `timestamp_ntz` values do not.

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
        }
        for block, ok in allowed.items():
            if getattr(self, block) is not None and not ok:
                raise ValueError(f"column {self.name!r}: {block} stats are not valid for logical_type={kind.value}")
        return self


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


class KeyCandidate(_SpecModel):
    """Set of columns that is likely unique across the table.

    Attributes:
        columns: Column names that form the key, in order.
        confidence: Confidence that the columns form a key, in [0, 1].
    """

    columns: list[str] = Field(min_length=1)
    confidence: Fraction


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
        min: Optional inclusive lower bound; an ISO-8601 string for temporal columns.
        max: Optional inclusive upper bound; an ISO-8601 string for temporal columns.
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
        values: Allowed values.
        satisfied_fraction: Fraction of rows that satisfy the constraint.
    """

    kind: Literal["enumeration"] = "enumeration"
    column: str
    values: list[Scalar | bool] = Field(min_length=1)

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
    """Boolean Spark SQL expression that rows satisfy.

    Attributes:
        kind: Discriminator; always "expression".
        expression: Spark SQL boolean expression over the table's columns.
        satisfied_fraction: Fraction of rows that satisfy the constraint.
    """

    kind: Literal["expression"] = "expression"
    expression: str = Field(min_length=1)


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


class TableProfile(_SpecModel):
    """Profile of a single table.

    Attributes:
        name: Table name, unique within the profile.
        row_count: Number of rows profiled.
        columns: Top-level column profiles, in schema order.
        primary_key: Optional best primary key candidate.
        unique_keys: Optional other unique key candidates.
        check_constraints: Optional row-level constraints found in the data.
        dependencies: Optional pairwise dependence measures between columns.
    """

    name: str = Field(min_length=1)
    row_count: Count
    columns: list[ColumnProfile]
    primary_key: KeyCandidate | None = None
    unique_keys: list[KeyCandidate] | None = None
    check_constraints: list[CheckConstraint] | None = None
    dependencies: list[Dependency] | None = None

    @model_validator(mode="after")
    def _check_column_refs(self) -> TableProfile:
        names = [column.name for column in self.columns]
        if len(set(names)) != len(names):
            raise ValueError(f"table {self.name!r} has duplicate column names")
        keys = [self.primary_key] if self.primary_key else []
        refs = [
            *(name for key in [*keys, *(self.unique_keys or [])] for name in key.columns),
            *(name for constraint in self.check_constraints or [] for name in constraint.column_refs()),
            *(name for dependency in self.dependencies or [] for name in dependency.columns),
        ]
        unknown = sorted(set(refs) - set(names))
        if unknown:
            raise ValueError(f"table {self.name!r} refers to unknown columns {unknown}")
        return self


class FanOutStats(_SpecModel):
    """Distribution of the number of child rows per parent key.

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


class Relationship(_SpecModel):
    """Foreign key from child columns to parent key columns.

    Attributes:
        child_table: Table that holds the foreign key.
        child_columns: Foreign key columns, in order.
        parent_table: Referenced table.
        parent_columns: Referenced key columns, matched to `child_columns` by position.
        coverage: Fraction of non-null child keys that match a parent key.
        confidence: Optional confidence that the relationship is a true foreign key, in [0, 1].
        fan_out: Optional distribution of child rows per parent key, including parents with no children.
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
        spec_version: Version of this spec that the profile follows, as "major.minor.patch".
        created_at: Time the profile was created, with a UTC offset.
        engine: Engine that produced the profile.
        source: Optional metadata about the profiled source.
        tables: Table profiles.
        relationships: Optional foreign key relationships between tables.
    """

    spec_version: str = Field(pattern=_SEMVER_RE.pattern)
    created_at: AwareDatetime
    engine: EngineInfo
    source: SourceInfo | None = None
    tables: list[TableProfile] = Field(min_length=1)
    relationships: list[Relationship] | None = None

    @field_validator("spec_version")
    @classmethod
    def _check_major_version(cls, value: str) -> str:
        if parse_version(value)[0] != parse_version(SPEC_VERSION)[0]:
            raise ValueError(f"spec_version={value!r} has a different major version than supported {SPEC_VERSION!r}")
        return value

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
