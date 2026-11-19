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
    engines/streaming/            incremental profiling with Spark Structured Streaming
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

### Streaming

`dbldatagen_profiler.engines.streaming` profiles data as it arrives. Each micro-batch is reduced to a mergeable
state (one Spark aggregate per nesting level, like the batch engine), and states are merged on the driver:

```python
from dbldatagen_profiler.engines.streaming import StreamingProfiler, merge_states, profile_states, state_for_dataframe

profiler = StreamingProfiler({"orders": spark.readStream.schema(schema).parquet(path)}, checkpoint_dir="ckpt")
profiler.start()               # writeStream.foreachBatch, trigger "availableNow" (default) or e.g. "30 seconds"
snapshot = profiler.profile()  # spec-valid profile of every batch merged so far; callable at any time
profiler.await_termination()
profiler.stop()

state = merge_states(state_for_dataframe(df_a), state_for_dataframe(df_b))  # a batch DataFrame as one micro-batch
profile = profile_states({"orders": state})
```

From the command line, for file sources:

```bash
uv run --package dbldatagen-profiler --all-extras dbldatagen-profile stream \
  --table orders=data/orders --format parquet --checkpoint ckpt --trigger availableNow -o profile.json
```

With an interval trigger the command runs until interrupted and rewrites `profile.json` every 10 seconds.

The streaming engine requires pyspark 3.5 or later (for Spark's `hll_sketch_agg`); `StreamingProfiler` and
`dbldatagen-profile stream` fail with a clear error on older versions. The batch engine still supports pyspark 3.4.1.

A snapshot equals the batch profile of the same rows on these statistics: row count, null fraction, nested
`value_count`, numeric and temporal min/max (lossless for decimal and int64), string and collection length
min/max/mean, and boolean true fraction. Numeric mean/stddev and collection length stddev are merged in floating
point (Chan et al.) and match to rounding error. Distinct counts come from Spark's built-in HyperLogLog sketch
(Apache DataSketches HLL, lgConfigK 12) and are marked `exact: false`. Merging sketches is order-independent, so
the estimate does not depend on how rows are split into batches. DataSketches documents the relative standard error
of a merged HLL sketch as about 1.04/√K with K = 2^lgConfigK, which is 1.63% at lgConfigK 12 (`DISTINCT_RELATIVE_ERROR`).
The tests check estimates over 200,000 distinct values against three times that error.

State is written atomically, as versioned JSON, to `<checkpoint>/<table>/state.json` after every batch, with the set
of merged batch ids; the file and its directory are fsynced. Spark's own checkpoint is `<checkpoint>/<table>/spark`.
A batch whose id is already merged (a replay after a crash) is skipped, so restarts neither double-count nor lose
batches, in whatever order batches arrive. Loading a checkpoint never executes code; a tampered file or an unknown
version fails with a clear error. In the file, sketches are base64, int64, decimal, and exact length totals are
decimal strings, and floats are `float.hex` strings. The checkpoint must be on a local filesystem.

PII detection runs on the first `pii_warmup_rows` rows (default 10,000) in arrival order. Once a table has seen that
many rows the detection result is frozen, so later batches do not change semantic labels, and the raw rows are
dropped from the checkpoint. Arrival order is the micro-batch id, then, within a batch, the file sort key (file
modification time, file path) and the row position in the file. A stream cannot wait for data that might arrive
later, so rows in a later batch never enter the warmup set, even if their file has an earlier name or modification
time. A file that arrives after the freeze is still profiled into every statistic, but it never changes the frozen
labels. Spark's file source takes files in modification-time order, so for files present when the stream starts the
frozen labels do not depend on how the files are split into micro-batches. Row position is the Parquet row index;
for JSON, rows within a file are ordered by the batch engine's xxhash64 sampling key. For non-file sources, rows
within a batch are ordered by that sampling key alone. When a table has no more rows than the warmup, the sample is every row, as in the batch engine, so semantics
and quasi-identifier sets equal the batch profile's. Until the warmup ends the checkpoint holds those raw rows as
JSON; keep the checkpoint directory as private as the source data. Values of interval (other than day-time),
variant, and spatial types are not kept in the sample. Snapshots follow the same privacy rules as batch profiles. A
PII handling policy (`pii_policy`) is not supported yet.

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
- `semantic`: optional PII class, sensitivity, confidence, and value-free detection evidence. Tables may also record
  sampled pairs and triples that behave as quasi-identifiers.

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

Producers emit exactly six fractional digits when a temporal value has a non-zero sub-second
component (e.g. `"2024-02-04T16:27:43.791440Z"`) and omit the fraction otherwise. Consumers parse
1–9 fractional digits via `parse_temporal` so odd lengths from older profiles still load on Python 3.10.
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

PII detection is enabled during Spark profiling by default. Full-column validators require an 80% sampled match
rate; free-text span detection requires 20%; account IDs require 90% sampled uniqueness. A column-name hint alone
has confidence 0.30. Quasi-identifier search considers at most 12 moderate-cardinality columns and reports sampled
pairs or triples with at least 80% unique rows. Detection samples at most 10,000 rows using a seeded content hash.
Direct identifiers and secrets never retain literal-bearing statistics in the resulting profile.

Detection and keyed substitution can also run independently:

```bash
dbldatagen-profile pii detect --table customers=customers.parquet -o semantic.json
dbldatagen-profile pii substitute --table customers=customers.parquet \
  --profile profile.json --key-file key.bin --policy policy.yaml -o shareable
```

Substitution uses HMAC-SHA256 domains. Relationship-linked columns share a domain, preserving equality and joins;
nulls and empty strings are unchanged. Email domains are replaced by default because they may identify a group.

The default PII policy substitutes every sensitivity, including quasi-identifiers. Postal codes, cities, and other
categorical quasi-identifiers receive keyed surrogates that preserve equality, frequencies, distinct counts, and
joins. Dates of birth use an entity-keyed date shift. This keeps those statistical properties, but discards the
original geography and ordering semantics; choose `generalize` or `keep` when those semantics are required. Print the
default policy with `dbldatagen-profile pii policy --default`.

Date-of-birth surrogates changed from the previous release: the entity-keyed shift now excludes zero, so a substituted
DOB is always changed when `date_shift_days` is positive.

Policies are JSON or YAML mappings. A fully-qualified column rule (including nested paths) wins over a PII-class rule,
which wins over a sensitivity rule. The default `min_confidence` is 0.25, so name-only detections are substituted.
Direct identifiers and secrets are never skipped for low confidence. Other detections below the configured threshold
are reported as `skipped_low_confidence` and left unchanged unless a class or column override names them. Linked FK
columns use one action for the whole substitution domain (`drop` > `suppress` > `substitute` > `generalize` > `keep`);
conflicting explicit rules are rejected. One explicit `keep` in an FK group is therefore overridden by a stricter
action elsewhere in that group. Available actions are `substitute`, `generalize`, `suppress`, `drop`, and `keep`.
Generalization parameters include `prefix_length`, `date_granularity` (`month` or `year`), and `bucket_width`. For
example:

```yaml
min_confidence: 0.25
sensitivity:
  direct_identifier: {action: substitute}
  quasi_identifier: {action: substitute}
  secret: {action: substitute}
  free_text: {action: substitute}
pii_class:
  postal_code:
    action: generalize
    parameters: {prefix_length: 3}
columns:
  customers.notes: {action: suppress}
  customers.unclassified_contact: {action: substitute, pii_class: email}
```

A fully-qualified column rule also applies when detection found no semantic class. For `substitute`, an optional
`pii_class` selects class-specific formatting. Without it, substitution uses a generic opaque keyed surrogate while
preserving numeric, decimal, date, and timestamp types. This behavior also applies to nested fields. Generic boolean
substitution is rejected because two values cannot be replaced injectively while preserving the boolean type; choose
`suppress` instead.

The same file can be passed to `dbldatagen-profile profile --policy policy.yaml`. At profile time, policy rules may
only remove or coarsen emitted information. They cannot weaken the spec 0.2.0 floor for direct identifiers and
secrets; a conflicting `keep` rule is rejected. Profile-time `generalize` and `drop` strip sensitive statistics just
like `suppress`; they do not change the source schema or calculate generalized statistics.

JSON policy files always work. YAML policy files and `pii policy --default` require the optional
`dbldatagen-profiler[yaml]` extra.

Format preservation also preserves value length, so length remains visible. HMAC substitution of low-entropy values
such as SSNs and phone numbers is safe only while the key remains secret: keep the key under restricted custody and
rotate it when access changes. The CLI requires a key of at least 32 bytes in a file unreadable by group or other
users unless `--allow-insecure-key-file` is explicitly supplied. Names and street addresses embedded in free text are
not currently detected or replaced; validator-shaped spans such as emails, phones, cards, IDs, and IP addresses are.

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
