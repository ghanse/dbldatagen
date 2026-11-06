"""Ground-truth manifest schema for catalog datasets.

A manifest declares the structure a dataset generator planted on purpose: keys,
foreign keys and their fan-out, check constraints, functional dependencies,
per-column distributions, string formats, pairwise dependencies, null rates, and
cardinality. It is independent of the profile spec so that the benchmark does not
measure the profiler with its own vocabulary.

Declared values are GENERATING parameters: the distribution parameters, shares,
null rates, dependency strengths, and fan-out laws the generator sampled from, not
statistics of the realized sample. Use `profiler_bench.catalog.realized` for the
statistics of a materialized instance. Exact structure (keys, constraints, exact
distinct counts, planted extremes, null rates of 0 or 1) holds in every instance.

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

import math
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MANIFEST_VERSION = "0.2.0"

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Bound = int | float | str
Privacy = Literal["direct_identifier", "quasi_identifier", "secret", "free_text"]


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
    - `binomial`: `n`, `p`, optional `shift` added to every draw.
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
OPTIONAL_PARAMS: dict[Family, frozenset[str]] = {
    Family.POISSON: frozenset({"shift"}),
    Family.BINOMIAL: frozenset({"shift"}),
}


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _order_key(value: object) -> float:
    """Returns a comparable number for a numeric bound, a decimal string, or an ISO date or timestamp string."""
    if isinstance(value, str):
        try:
            return float(Decimal(value))
        except InvalidOperation:
            moment = datetime.fromisoformat(value.removesuffix("Z"))
            return (moment - datetime(1970, 1, 1)).total_seconds()
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"{value!r} is not a number, decimal string, or ISO date")


def _positive(p: dict[str, Any], *keys: str) -> list[str]:
    return [f"{k} must be a positive number" for k in keys if not (_is_number(p[k]) and p[k] > 0)]


def _numbers(p: dict[str, Any], *keys: str) -> list[str]:
    return [f"{k} must be a finite number" for k in keys if not _is_number(p[k])]


def _probability(p: dict[str, Any], key: str) -> list[str]:
    return [] if _is_number(p[key]) and 0 < p[key] < 1 else [f"{key} must be in (0, 1)"]


def _ordered(low: object, high: object, strict: bool) -> list[str]:
    try:
        lo, hi = _order_key(low), _order_key(high)
    except (TypeError, ValueError) as error:
        return [str(error)]
    return [] if lo < hi or (not strict and lo == hi) else [f"bounds {low!r}, {high!r} are not ordered"]


def _weights(values: object, length: int | None, name: str) -> list[str]:
    if not isinstance(values, list) or (length is not None and len(values) != length) or not values:
        return [f"{name} must be a non-empty list" + (f" of {length}" if length else "")]
    if not all(_is_number(v) and v >= 0 for v in values) or sum(values) <= 0:
        return [f"{name} must be non-negative with a positive sum"]
    return []


def _probabilities(values: object, name: str) -> list[str]:
    errors = _weights(values, None, name)
    if errors or not isinstance(values, list):
        return errors
    return [] if all(v <= 1 for v in values) and abs(sum(values) - 1.0) <= 1e-6 else [f"{name} must sum to 1"]


def _mixture(p: dict[str, Any]) -> list[str]:
    components = p["components"]
    if not isinstance(components, list) or not components:
        return ["components must be a non-empty list"]
    if any(not isinstance(c, dict) or set(c) != {"weight", "mean", "std"} for c in components):
        return ["each component needs exactly weight, mean, and std"]
    errors = [e for c in components for e in _numbers(c, "mean") + _positive(c, "std")]
    return errors + _probabilities([c["weight"] for c in components], "component weights")


def _categorical(p: dict[str, Any]) -> list[str]:
    values, probabilities = p["values"], p["probabilities"]
    if not isinstance(values, list) or not isinstance(probabilities, list) or len(values) != len(probabilities):
        return ["categorical values and probabilities must be aligned lists"]
    if len({repr(v) for v in values}) != len(values):
        return ["categorical values must be distinct"]
    return _probabilities(probabilities, "probabilities")


def _poisson(p: dict[str, Any]) -> list[str]:
    return _positive(p, "lam") + ([] if _is_int(p.get("shift", 0)) else ["shift must be an integer"])


def _binomial(p: dict[str, Any]) -> list[str]:
    errors = [] if _is_int(p["n"]) and p["n"] >= 1 else ["n must be a positive integer"]
    return errors + _probability(p, "p") + ([] if _is_int(p.get("shift", 0)) else ["shift must be an integer"])


def _zipf(p: dict[str, Any]) -> list[str]:
    return _positive(p, "s") + ([] if _is_int(p["n_values"]) and p["n_values"] >= 1 else ["n_values must be >= 1"])


def _discrete_uniform(p: dict[str, Any]) -> list[str]:
    if not (_is_int(p["low"]) and _is_int(p["high"])):
        return ["low and high must be integers"]
    return _ordered(p["low"], p["high"], strict=False)


def _sequence(p: dict[str, Any]) -> list[str]:
    errors = _numbers(p, "start", "step")
    return errors or ([] if p["step"] != 0 else ["step must be non-zero"])


def _seasonal(p: dict[str, Any]) -> list[str]:
    return (
        _ordered(p["start"], p["end"], strict=True)
        + _weights(p["hour_weights"], 24, "hour_weights")
        + _weights(p["weekday_weights"], 7, "weekday_weights")
    )


_DOMAINS: dict[Family, Callable[[dict[str, Any]], list[str]]] = {
    Family.CONSTANT: lambda p: [],
    Family.NORMAL: lambda p: _numbers(p, "mean") + _positive(p, "std"),
    Family.LOGNORMAL: lambda p: _numbers(p, "mu") + _positive(p, "sigma"),
    Family.MIXTURE: _mixture,
    Family.UNIFORM: lambda p: _ordered(p["low"], p["high"], strict=True),
    Family.DISCRETE_UNIFORM: _discrete_uniform,
    Family.EXPONENTIAL: lambda p: _positive(p, "scale"),
    Family.PARETO: lambda p: _positive(p, "alpha", "scale"),
    Family.POISSON: _poisson,
    Family.BINOMIAL: _binomial,
    Family.NEGATIVE_BINOMIAL: lambda p: _positive(p, "n") + _probability(p, "p"),
    Family.ZIPF: _zipf,
    Family.CATEGORICAL: _categorical,
    Family.SEQUENCE: _sequence,
    Family.SEASONAL: _seasonal,
}


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
        errors = _DOMAINS[self.family](self.params)
        if errors:
            raise ValueError(f"{self.family.value}: {'; '.join(errors)}")
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
        privacy: Privacy class of the values, or None when they are not sensitive:
            `direct_identifier` (identifies a person or record on its own, e.g. email, phone, account
            number, a record UUID), `quasi_identifier` (identifies in combination, see
            `Table.quasi_identifiers`), `secret` (credentials and tokens), or `free_text` (may hold
            anything). Values of every class except `quasi_identifier` must not appear in a profile.
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
    privacy: Privacy | None = None
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

    @property
    def sensitive(self) -> bool:
        """Returns whether the column has a privacy class."""
        return self.privacy is not None

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

    @model_validator(mode="after")
    def _check_order(self) -> RangeCheck:
        errors = [] if self.min is None or self.max is None else _ordered(self.min, self.max, strict=False)
        if errors:
            raise ValueError(f"range on {self.column}: {errors[0]}")
        return self


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


class QuasiIdentifierSet(_Model):
    """Columns that together identify rows although none does alone.

    Attributes:
        columns: The quasi-identifier columns.
        unique_fraction: Expected fraction of rows whose combination of values occurs once in the table,
            under the generating distribution.
        mechanism: How the combination was generated.
    """

    columns: tuple[str, ...]
    unique_fraction: Fraction
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
        quasi_identifiers: Planted quasi-identifier combinations.
        row_order: `generated` when rows are stored in generation order, or `shuffled` when they were
            permuted at random before writing. Storage order, which claims such as `monotonic` refer
            to, is the part files in path order and the rows in file order; Spark does not read
            multi-file tables in that order, so sort by `_metadata.file_path` first.
        files: Number of parquet files the table is written as, filled in from the tier.
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
    quasi_identifiers: tuple[QuasiIdentifierSet, ...] = ()
    row_order: Literal["generated", "shuffled"] = "generated"
    files: int = Field(default=1, ge=1)

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
            *(c for qi in self.quasi_identifiers for c in qi.columns),
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
        declared_values: Always `generating_parameters`: declared values are what the generator sampled
            from, not statistics of this instance (see `profiler_bench.catalog.realized`).
        dataset: Catalog dataset name.
        tier: Scale tier the data was generated at.
        structure_seed: Seed of the generating parameters and vocabularies. The manifest depends only on
            it and the tier; the rows also depend on a sample seed, which the manifest does not record.
            Input that uses the earlier name `seed` is accepted.
        description: What the dataset exercises.
        tables: Planted structure of every table.
        foreign_keys: Foreign keys between tables.
    """

    manifest_version: str = MANIFEST_VERSION
    declared_values: Literal["generating_parameters"] = "generating_parameters"
    dataset: str
    tier: str
    structure_seed: int
    description: str
    tables: tuple[Table, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _accept_seed(cls, data: object) -> object:
        if isinstance(data, dict) and "seed" in data and "structure_seed" not in data:
            return {("structure_seed" if k == "seed" else k): v for k, v in data.items()}
        return data

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

    @property
    def seed(self) -> int:
        """Returns the structure seed (the name used before seeds were split)."""
        return self.structure_seed

    def table(self, name: str) -> Table:
        """Returns the table called `name`."""
        return next(t for t in self.tables if t.name == name)
