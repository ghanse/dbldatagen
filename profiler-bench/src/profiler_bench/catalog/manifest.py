"""Ground-truth manifest schema for catalog datasets.

A manifest declares the structure a dataset generator planted on purpose: keys,
foreign keys and their fan-out, check constraints, functional dependencies,
per-column distributions, string formats, pairwise dependencies, null rates, and
cardinality. It is independent of the profile spec so that the benchmark does not
measure the profiler with its own vocabulary.

Conventions:

- Column `type` is the Spark SQL type string (`DataType.simpleString()`).
- Shares, null rates, and distribution checks apply to the non-null values of the
  column. Struct fields are measured over rows whose struct is non-null, array
  elements over all elements, and map keys and values over all entries.
- Dates and timestamps in distribution parameters and range bounds are ISO-8601
  strings; timestamps are UTC (`timestamp`) or wall-clock (`timestamp_ntz`).
- Decimal bounds are decimal strings so they survive JSON without loss.
- Format regexes use Python `re` syntax and must match the whole value.
- Spark SQL check expressions assume the session time zone is UTC.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MANIFEST_VERSION = "0.1.0"

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Bound = int | float | str


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Family(str, Enum):
    """Distribution family with its required parameters.

    - `constant`: `value`.
    - `normal`: `mean`, `std`.
    - `lognormal`: `mu`, `sigma` of the underlying normal.
    - `mixture`: `components`, a list of `{weight, mean, std}` normal components.
    - `uniform`: `low`, `high` (continuous, or dates and timestamps as ISO strings).
    - `discrete_uniform`: integer `low`, `high`, both inclusive.
    - `exponential`: `scale` (the mean).
    - `pareto`: `alpha`, `scale` (the minimum).
    - `poisson`: `lam`, optional `shift` added to every draw.
    - `binomial`: `n`, `p`.
    - `negative_binomial`: `n`, `p` (numpy parameterization, mean `n(1-p)/p`).
    - `zipf`: `s`, `n_values`; value of rank `k` has probability proportional to `k**-s`.
    - `categorical`: `values`, `probabilities`.
    - `sequence`: `start`, `step`; the sorted values are `start + step * i`.
    - `seasonal`: `start`, `end` (ISO timestamps), `hour_weights` (24, UTC hour of day),
      `weekday_weights` (7, Monday first); the shares of values per hour and weekday.
    """

    CONSTANT = "constant"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    MIXTURE = "mixture"
    UNIFORM = "uniform"
    DISCRETE_UNIFORM = "discrete_uniform"
    EXPONENTIAL = "exponential"
    PARETO = "pareto"
    POISSON = "poisson"
    BINOMIAL = "binomial"
    NEGATIVE_BINOMIAL = "negative_binomial"
    ZIPF = "zipf"
    CATEGORICAL = "categorical"
    SEQUENCE = "sequence"
    SEASONAL = "seasonal"


REQUIRED_PARAMS: dict[Family, frozenset[str]] = {
    Family.CONSTANT: frozenset({"value"}),
    Family.NORMAL: frozenset({"mean", "std"}),
    Family.LOGNORMAL: frozenset({"mu", "sigma"}),
    Family.MIXTURE: frozenset({"components"}),
    Family.UNIFORM: frozenset({"low", "high"}),
    Family.DISCRETE_UNIFORM: frozenset({"low", "high"}),
    Family.EXPONENTIAL: frozenset({"scale"}),
    Family.PARETO: frozenset({"alpha", "scale"}),
    Family.POISSON: frozenset({"lam"}),
    Family.BINOMIAL: frozenset({"n", "p"}),
    Family.NEGATIVE_BINOMIAL: frozenset({"n", "p"}),
    Family.ZIPF: frozenset({"s", "n_values"}),
    Family.CATEGORICAL: frozenset({"values", "probabilities"}),
    Family.SEQUENCE: frozenset({"start", "step"}),
    Family.SEASONAL: frozenset({"start", "end", "hour_weights", "weekday_weights"}),
}
OPTIONAL_PARAMS: dict[Family, frozenset[str]] = {Family.POISSON: frozenset({"shift"})}


class Distribution(_Model):
    """Distribution a column's values, lengths, increments, or fan-out were drawn from.

    Attributes:
        family: Distribution family.
        params: Family parameters; see `Family` for the required keys.
    """

    family: Family
    params: dict[str, Any]

    @model_validator(mode="after")
    def _check_params(self) -> Distribution:
        required = REQUIRED_PARAMS[self.family]
        allowed = required | OPTIONAL_PARAMS.get(self.family, frozenset())
        keys = set(self.params)
        if not required <= keys <= allowed:
            raise ValueError(f"{self.family.value} needs params {sorted(required)}, got {sorted(keys)}")
        if self.family == Family.CATEGORICAL:
            values, probabilities = self.params["values"], self.params["probabilities"]
            if len(values) != len(probabilities) or abs(sum(probabilities) - 1.0) > 1e-6:
                raise ValueError("categorical values and probabilities must align and sum to 1")
        if self.family == Family.MIXTURE and abs(sum(c["weight"] for c in self.params["components"]) - 1.0) > 1e-6:
            raise ValueError("mixture weights must sum to 1")
        if self.family == Family.SEASONAL and (
            len(self.params["hour_weights"]) != 24 or len(self.params["weekday_weights"]) != 7
        ):
            raise ValueError("seasonal needs 24 hour weights and 7 weekday weights")
        return self


class Format(_Model):
    """String format and the share of non-null values that follow it.

    Attributes:
        regex: Reference regex in Python `re` syntax, matched against the whole value.
        share: Share of non-null values that match. Formats of one column are disjoint.
    """

    regex: str
    share: Fraction


class Column(_Model):
    """Planted structure of one column, struct field, array element, or map key or value.

    Attributes:
        name: Column or field name; `element`, `key`, or `value` for collection children.
        type: Spark SQL type string, filled in from the generated data.
        null_rate: Expected fraction of null values.
        unique: Whether non-null values are distinct.
        distinct: Exact number of distinct non-null values, when planted.
        max_distinct: Upper bound on distinct non-null values (the size of the value domain).
        distribution: Distribution of the non-null values.
        formats: Disjoint string formats with their shares.
        min_value: Exact minimum of the non-null values, when planted.
        max_value: Exact maximum of the non-null values, when planted.
        length: Distribution of the number of elements (arrays, maps) or bytes (binary) of non-null values.
        monotonic: Ordering of the values in storage order.
        increments: Distribution of the differences between consecutive values in storage order,
            in the column's unit (days for dates, seconds for timestamps).
        special_values: Shares of `nan`, `inf`, and `-inf` among non-null float values.
        sensitive: Whether the values identify individuals and must not appear in a profile.
        fields: Struct fields.
        element: Array element.
        key: Map key.
        value: Map value.
        note: Free-text description of how the column was generated.
    """

    name: str
    type: str = ""
    null_rate: Fraction = 0.0
    unique: bool = False
    distinct: int | None = Field(default=None, ge=0)
    max_distinct: int | None = Field(default=None, ge=0)
    distribution: Distribution | None = None
    formats: tuple[Format, ...] = ()
    min_value: Bound | None = None
    max_value: Bound | None = None
    length: Distribution | None = None
    monotonic: Literal["increasing", "strictly_increasing"] | None = None
    increments: Distribution | None = None
    special_values: dict[Literal["nan", "inf", "-inf"], Fraction] = {}
    sensitive: bool = False
    fields: tuple[Column, ...] = ()
    element: Column | None = None
    key: Column | None = None
    value: Column | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _check_shares(self) -> Column:
        if sum(f.share for f in self.formats) > 1.0 + 1e-6:
            raise ValueError(f"format shares of {self.name} exceed 1")
        return self

    def children(self) -> tuple[Column, ...]:
        """Returns the nested columns: struct fields, then the array element or map key and value."""
        return self.fields + tuple(c for c in (self.element, self.key, self.value) if c is not None)


class RangeCheck(_Model):
    """Inclusive range every non-null value satisfies.

    Attributes:
        kind: Constant `range`.
        column: Column name.
        min: Lower bound, or None for no bound.
        max: Upper bound, or None for no bound.
    """

    kind: Literal["range"] = "range"
    column: str
    min: Bound | None = None
    max: Bound | None = None


class EnumerationCheck(_Model):
    """Closed set of values every non-null value belongs to.

    Attributes:
        kind: Constant `enumeration`.
        column: Column name.
        values: Allowed values.
    """

    kind: Literal["enumeration"] = "enumeration"
    column: str
    values: tuple[Bound | bool, ...]


class ComparisonCheck(_Model):
    """Comparison between two columns that holds for every row where both are non-null.

    Attributes:
        kind: Constant `comparison`.
        left: Left column name.
        op: Comparison operator.
        right: Right column name.
    """

    kind: Literal["comparison"] = "comparison"
    left: str
    op: Literal["<", "<=", "=", ">=", ">"]
    right: str


class ExpressionCheck(_Model):
    """Spark SQL boolean expression that is never false (null counts as satisfied, as in SQL CHECK).

    Attributes:
        kind: Constant `expression`.
        sql: Spark SQL expression over the table's columns.
        columns: Columns the expression reads.
    """

    kind: Literal["expression"] = "expression"
    sql: str
    columns: tuple[str, ...]


Check = Annotated[RangeCheck | EnumerationCheck | ComparisonCheck | ExpressionCheck, Field(discriminator="kind")]


class FunctionalDependency(_Model):
    """Functional dependency `determinant -> dependent` over non-null values.

    Attributes:
        determinant: Columns that determine the dependent columns.
        dependent: Columns with one value per determinant value.
    """

    determinant: tuple[str, ...]
    dependent: tuple[str, ...]


class Dependency(_Model):
    """Planted pairwise dependency and its strength.

    Attributes:
        columns: The two columns. For `correlation_ratio`, the first is the conditioning column.
        measure: `pearson`, `spearman`, `cramers_v`, or `correlation_ratio` (eta of the second
            column given the first, binned into 50 quantile bins when numeric).
        value: Expected value of the measure.
        mechanism: How the dependency was generated, e.g. `gaussian copula rho=0.7`.
    """

    columns: tuple[str, str]
    measure: Literal["pearson", "spearman", "cramers_v", "correlation_ratio"]
    value: float = Field(ge=-1.0, le=1.0)
    mechanism: str


class Table(_Model):
    """Planted structure of one table.

    Attributes:
        name: Table name.
        row_count: Number of rows, filled in from the generated data.
        description: What the table models.
        primary_key: Primary key columns, if any.
        unique_keys: Other unique column sets.
        columns: Every column of the table, in order.
        checks: Check constraints that hold for every row.
        functional_dependencies: Functional dependencies between columns.
        dependencies: Pairwise dependencies with their strength.
    """

    name: str
    row_count: int = Field(default=-1, ge=-1)
    description: str = ""
    primary_key: tuple[str, ...] = ()
    unique_keys: tuple[tuple[str, ...], ...] = ()
    columns: tuple[Column, ...]
    checks: tuple[Check, ...] = ()
    functional_dependencies: tuple[FunctionalDependency, ...] = ()
    dependencies: tuple[Dependency, ...] = ()

    @model_validator(mode="after")
    def _check_references(self) -> Table:
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            raise ValueError(f"table {self.name} has duplicate columns")
        referenced = [
            *self.primary_key,
            *(c for key in self.unique_keys for c in key),
            *(c for fd in self.functional_dependencies for c in (*fd.determinant, *fd.dependent)),
            *(c for dep in self.dependencies for c in dep.columns),
            *(c for check in self.checks for c in _check_columns(check)),
        ]
        missing = sorted(set(referenced) - set(names))
        if missing:
            raise ValueError(f"table {self.name} references unknown columns {missing}")
        return self

    def column(self, name: str) -> Column:
        """Returns the column called `name`."""
        return next(c for c in self.columns if c.name == name)


def _check_columns(check: Check) -> tuple[str, ...]:
    if isinstance(check, ComparisonCheck):
        return (check.left, check.right)
    if isinstance(check, ExpressionCheck):
        return check.columns
    return (check.column,)


class ForeignKey(_Model):
    """Foreign key from child columns to parent columns.

    Attributes:
        child_table: Referencing table.
        child_columns: Referencing columns.
        parent_table: Referenced table.
        parent_columns: Referenced columns, a primary or unique key of the parent.
        coverage: Fraction of non-null child keys that match a parent key.
        fanout: Distribution of child rows per parent key, counting keys with none. For `zipf`,
            parents are ranked in random order and chosen with probability proportional to `rank**-s`.
    """

    child_table: str
    child_columns: tuple[str, ...]
    parent_table: str
    parent_columns: tuple[str, ...]
    coverage: Fraction = 1.0
    fanout: Distribution | None = None


class Manifest(_Model):
    """Ground truth for one generated dataset instance.

    Attributes:
        manifest_version: Version of this schema.
        dataset: Catalog dataset name.
        tier: Scale tier the data was generated at.
        seed: Seed the data was generated with.
        description: What the dataset exercises.
        tables: Planted structure of every table.
        foreign_keys: Foreign keys between tables.
    """

    manifest_version: str = MANIFEST_VERSION
    dataset: str
    tier: str
    seed: int
    description: str
    tables: tuple[Table, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()

    @model_validator(mode="after")
    def _check_foreign_keys(self) -> Manifest:
        tables = {t.name: t for t in self.tables}
        for fk in self.foreign_keys:
            for table, columns in ((fk.child_table, fk.child_columns), (fk.parent_table, fk.parent_columns)):
                if table not in tables or not set(columns) <= {c.name for c in tables[table].columns}:
                    raise ValueError(f"foreign key references unknown {table}{list(columns)}")
            parent = tables[fk.parent_table]
            if fk.parent_columns not in (parent.primary_key, *parent.unique_keys):
                raise ValueError(f"foreign key target {fk.parent_table}{list(fk.parent_columns)} is not a key")
        return self

    def table(self, name: str) -> Table:
        """Returns the table called `name`."""
        return next(t for t in self.tables if t.name == name)
