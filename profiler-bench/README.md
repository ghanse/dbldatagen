# dbldatagen-profiler-bench

Benchmark harness for `dbldatagen-profiler`. It is a separate distribution and a member of the repository's uv
workspace. This first part is the **dataset catalog**: synthetic datasets with planted, declared ground truth. The
sampler, fidelity metrics, runner, and evaluator described in `profiler/PLAN.md` will live here as well.

## The catalog

Each dataset is a seeded generator written with numpy and Arrow, not dbldatagen, so the benchmark does not favour
what the generator can already do. A dataset ships a **ground-truth manifest** (`profiler_bench.catalog.manifest`)
declaring the structure it planted: primary and unique keys, foreign keys and their fan-out distribution, check
constraints, functional dependencies, per-column distributions with parameters, string formats as reference regexes
with shares, pairwise dependencies with their strength, null rates, and cardinality. The manifest has its own schema
and does not reuse the profile spec. See [COVERAGE.md](COVERAGE.md) for what each dataset exercises.

| Tier | Base rows | Use |
|---|---|---|
| `xs` | 5,000 | unit tests |
| `s` | 100,000 | default RSI loop tier |
| `m` | 2,000,000 | speed tier |

The base row count is the size of each dataset's largest table; the other tables scale with it.

```bash
uv run --package dbldatagen-profiler-bench python -m profiler_bench.catalog list
uv run --package dbldatagen-profiler-bench python -m profiler_bench.catalog materialize --tier s
uv run --package dbldatagen-profiler-bench python -m profiler_bench.catalog describe retail --tier xs
```

From Python:

```python
from profiler_bench.catalog import load, materialize, verify

frames = load("retail", "s", seed=0, spark=spark)    # dict of table name -> DataFrame
truth = materialize("retail", "s", seed=0).manifest  # ground-truth manifest
assert verify(truth, frames) == []                   # the manifest is true of the data
```

`verify` needs `spark.sql.session.timeZone=UTC` and collects columns to the driver, so use it on the `xs` and `s`
tiers.

### Determinism and the cache

Every random stream is derived from the seed, the dataset name, and a per-column key, so adding a column does not
change the others. `materialize` writes one zstd parquet file per table, `manifest.json`, and `meta.json` with the
SHA-256 of each file and a combined `content_hash`. The cache is `profiler-bench/.cache/catalog` (gitignored), or
`$PROFILER_BENCH_CACHE`:

```
<cache>/<dataset>/<tier>/seed=<seed>/<fingerprint>/{<table>.parquet, manifest.json, meta.json}
```

The fingerprint hashes the generator source and the shared catalog code, so editing a generator invalidates its
cache. Content hashes are stable for a given lock file; a new numpy or pyarrow version may change them.

### Public datasets

`profiler_bench.catalog.public` downloads a registered real dataset once, verifies its SHA-256, and returns None when
the URL cannot be reached, so callers can skip it offline. Real datasets have no planted truth and carry no manifest.
None are registered yet.

## Development

```bash
make bench-test    # pytest profiler-bench/tests (m-tier tests are marked slow; add -m slow to run them)
make bench-lint    # black, ruff, mypy
```
