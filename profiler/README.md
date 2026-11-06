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

`dbldatagen-profile profile` is reserved for the profiling engine and is not implemented yet. PySpark is an optional
extra: install `dbldatagen-profiler[spark]` to use the Spark engine once it exists.

## The profile spec

The spec is defined by the pydantic models in `dbldatagen_profiler.model`. `profiler/spec/profile.schema.json` is
generated from them so that engines in other languages can validate their output. After changing the model,
regenerate it with:

```bash
uv run --package dbldatagen-profiler dbldatagen-profile schema -o profiler/spec/profile.schema.json
```

A test fails if the committed schema is out of date.

A profile contains:

- `spec_version`, `created_at`, `engine` (name and version), and optional `source` metadata. Source metadata
  describes where the data came from and how it was sampled; it never holds data values.
- `tables`: each table has a `row_count` and `columns`, plus optional `primary_key` and `unique_keys` candidates
  with a confidence, `check_constraints` (range, enumeration, column comparison, or SQL expression, each with the
  fraction of rows that satisfy it), and pairwise `dependencies` (Pearson, Spearman, Cramér's V, correlation ratio).
- `columns`: each column has a `logical_type`, the `source_type` string, `nullable`, and `null_fraction`. Optional
  blocks add the distinct count, a cardinality bucket, and type-specific statistics: `numeric`, `string` (lengths,
  regex patterns with coverage, top values), `temporal`, and `boolean`. Struct, array, and map columns describe their
  nested fields in `children`.
- `relationships`: foreign keys from child columns to parent columns, with coverage, confidence, and fan-out.

Only the basic schema, row count, and null information are required; engines can add the other blocks over time.

### Versioning

`spec_version` follows `major.minor.patch`:

- **patch**: documentation or clarification only; no change to the schema.
- **minor**: adds optional fields. A reader accepts a profile with a newer minor version and drops the fields it does
  not know. A profile with the reader's own or an older minor version must not contain unknown fields.
- **major**: any other change, including new enum values or constraint kinds. A reader rejects a profile whose major
  version differs from its own.

Use `dbldatagen_profiler.model.load_profile` to read profiles so these rules apply.
