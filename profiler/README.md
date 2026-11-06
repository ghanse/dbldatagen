# dbldatagen-profiler

`dbldatagen-profiler` profiles one or more datasets into a portable JSON **profile**. A profile records the
structure needed to generate realistic synthetic data (schema, null rates, distributions, string patterns, keys,
constraints, and relationships between tables) without storing raw rows. dbldatagen, or any other generator, can
read the profile to produce benchmark data.

The package is a separate distribution that lives next to the `dbldatagen` source and is a member of the repository's
uv workspace. It does not change the `dbldatagen` package.

## Layout

```
profiler/
  pyproject.toml                  distribution `dbldatagen-profiler`
  src/dbldatagen_profiler/
    model/                        profile spec as pydantic models
    engines/spark/                baseline PySpark profiling engine
    cli.py                        `dbldatagen-profile` command
  spec/
    profile.schema.json           JSON Schema generated from the model
    examples/                     hand-written example profiles
  tests/
```

## Running

From the repository root:

```bash
uv run --package dbldatagen-profiler dbldatagen-profile validate profiler/spec/examples/single_table.json
uv run --package dbldatagen-profiler dbldatagen-profile schema
make profiler-test
make profiler-lint
```

Profile one or more local datasets with PySpark:

```bash
uv run --package dbldatagen-profiler --all-extras dbldatagen-profile profile \
  --table customers=data/customers.parquet \
  --table orders=data/orders.parquet \
  --format parquet -o profile.json
```

The input format may be `parquet`, `csv`, or `delta`. PySpark is an optional extra; install
`dbldatagen-profiler[spark]` when using the engine as a library.

## The profile spec

The spec is defined by the pydantic models in `dbldatagen_profiler.model`. `profiler/spec/profile.schema.json` is
generated from them so that engines in other languages can validate their output. After changing the model,
regenerate it with:

```bash
uv run --package dbldatagen-profiler dbldatagen-profile schema -o profiler/spec/profile.schema.json
```

A test fails if the committed schema is out of date.

A profile contains:

- `spec_version`, `created_at`, `engine` (name and version), a `privacy` block, and optional `source` metadata.
  Source metadata describes where the data came from and how it was sampled; it never holds data values.
- `tables`: each table has a `row_count` and at least one column, plus optional `primary_key` and `unique_keys`
  candidates with a confidence, `check_constraints` (range, enumeration, column comparison, or SQL expression with a
  `dialect`, each with the fraction of rows that satisfy it), and pairwise `dependencies` (Pearson, Spearman,
  Cramér's V, correlation ratio).
- `columns`: each column has a `logical_type`, the `source_type` string, `nullable`, and `null_fraction`. Optional
  blocks add the distinct count, a cardinality bucket, and type-specific statistics: `numeric` (min, max, mean,
  stddev, quantiles, histogram), `string` (lengths, regex patterns with coverage, top values), `temporal`, `boolean`,
  and `collection_length` for arrays and maps. Struct, array, and map columns describe their nested fields in
  `children`.
- `relationships`: foreign keys from child columns to parent columns, with coverage, confidence, and fan-out. Fan-out
  counts child rows for every parent key, including keys with none; child keys that match no parent are left out and
  show up in `coverage` instead.

Only the basic schema, row count, and null information are required; engines can add the other blocks over time.

### Value encoding

Literal values (min, max, quantiles, histogram edges, range bounds, enumeration values) follow the column's
`logical_type`:

| Logical type | Encoding | Example |
|---|---|---|
| `decimal`, `int64`, `uint64` | canonical decimal string, so no precision is lost | `"9223372036854775807"`, `"-0.50"` |
| other integers | JSON integer | `42` |
| `float32`, `float64` | JSON number | `1.5` |
| `date` | `YYYY-MM-DD` | `"2026-11-06"` |
| `time` | `HH:MM:SS[.fffffffff]` | `"08:30:00"` |
| `timestamp` | `YYYY-MM-DDTHH:MM:SS[.f]` with `Z` or `±HH:MM` | `"2026-11-06T08:30:00Z"` |
| `timestamp_ntz` | the same without an offset | `"2026-11-06T08:30:00"` |
| `string` | JSON string | `"gold"` |
| `boolean` | JSON boolean | `true` |

`mean` and `stddev` are always approximate JSON numbers. Histogram bins are `[lo, hi)` except the last, which is
`[lo, hi]`. The edges span every profiled value, and `counts` are row counts, with `sampled: true` when they come from
a sample.

### Privacy

The `privacy` block records the policy the producer applied:

- `min_value_count` (default 10): a value may appear in `top_values` or an enumeration constraint only if at least this
  many rows hold it. The model checks `top_values` frequencies against the table's row count and null fraction, and
  each enumeration's `min_value_count` (the row count of its rarest value). String `patterns` must also cover at least
  this many values. For nested columns, counts come from the column's `value_count` (the number of nested values its
  statistics cover); struct fields inherit their parent's count. A nested array or map child that has `patterns` or
  `top_values` but no `value_count` is rejected. Literals embedded in regexes are checked by the benchmark's privacy
  guard, not by the model.
- `extremes` and `enumerations`: `exact`, `generalized`, or `suppressed`. When suppressed, min/max statistics and
  range bounds (for `extremes`), or `top_values` and enumeration constraints (for `enumerations`), must be absent.

All raw-literal fields are optional, so a strict-privacy producer can leave them out.

### Versioning

`spec_version` follows `major.minor.patch`, and the major version is fixed by the JSON Schema (`^0\.` for this
release):

- **patch**: documentation or clarification only; no change to the schema.
- **minor**: adds optional fields. Readers ignore fields they do not know, and the JSON Schema accepts them, so older
  readers keep working with newer minor versions.
- **major**: any other change, including new enum values or constraint kinds. A reader rejects a profile whose major
  version differs from its own.

Producers should validate their output strictly, which rejects any field the spec does not define:

```bash
uv run --package dbldatagen-profiler dbldatagen-profile validate --strict profile.json
```

In Python, use `dbldatagen_profiler.model.load_profile(text, strict=True)`.
