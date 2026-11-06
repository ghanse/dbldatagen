# dbldatagen-profiler-bench

Benchmark harness for `dbldatagen-profiler`. It is a separate distribution and a member of the repository's uv
workspace. It holds the **dataset catalog**, the **reference sampler**, the **RSI runner / evaluator**, and (once
merged) the fidelity metrics described in `profiler/PLAN.md`.

## The catalog

Each dataset is a seeded generator written with numpy and Arrow, not dbldatagen, so the benchmark does not favour
what the generator can already do. A dataset ships a **ground-truth manifest** (`profiler_bench.catalog.manifest`)
declaring the structure it planted: primary and unique keys, foreign keys and their fan-out distribution, check
constraints, functional dependencies, per-column distributions with parameters, string formats as reference regexes
with shares, pairwise dependencies with their strength, null rates, cardinality, privacy classes, and
quasi-identifier sets. The manifest has its own schema and does not reuse the profile spec. See
[COVERAGE.md](COVERAGE.md) for what each dataset exercises.

Declared values are **generating parameters** (`Manifest.declared_values == "generating_parameters"`), not statistics
of the realized sample. `realized(manifest, frames)` (or `realized_statistics(name, tier, seed, spark)`) returns what
one instance actually contains: null rates, distinct counts, extremes, moments, format and category shares,
dependency values, quasi-identifier uniqueness, and foreign-key coverage and fan-out.

| Tier | Base rows | Use |
|---|---|---|
| `xs` | 5,000 | unit tests |
| `s` | 100,000 | default RSI loop tier |
| `m` | 2,000,000 | speed tier |

The base row count is the size of each dataset's largest table; the other tables scale with it, and
`Dataset.sizes(n)` gives every table's expected row count without generating data. Tiers `s` and `m` write each
table as 4 and 16 parquet part files; `retail` and `strings` shuffle their rows and use hashed (non-sequential)
keys, and each table declares its `row_order` and `files`. Storage order is the part files in path order and the
rows in file order. Spark does not read multi-file tables in that order, so sort by `_metadata.file_path` first.

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
tiers. Exact structure must hold exactly. Statistical claims are tested at `ALPHA = 1e-5`:
- every distribution by goodness of fit against its declared family and parameters (KS for continuous families,
  chi-square for discrete ones, an occupancy test for Zipf), plus bounds and moments;
- Pearson and Spearman strengths by Fisher z (Spearman with the 1.06 variance correction);
- Cramér's V and the correlation ratio with seeded bootstrap standard errors.

See the docstrings of `profiler_bench.catalog.verification` and `profiler_bench.catalog.distributions`.

### Seeds

An instance has two seeds. The **structure seed** fixes the distribution: generating parameters, vocabularies
(names, cities, streets), which values are frequent, and the parameters of `nested.wide`. The manifest depends only
on it and the tier. The **sample seed** fixes the rows drawn from that distribution. `seed` alone sets both, so
existing calls keep their meaning. Instances that share a structure seed are independent samples of the same
distribution, which is what a real-vs-real ceiling needs:

```python
truth = materialize("retail", "s", seed=0).manifest          # same manifest for every sample seed
first = load("retail", "s", seed=0, spark=spark, sample_seed=1)
second = load("retail", "s", seed=0, spark=spark, sample_seed=2)
```

### Determinism and the cache

Every random stream is derived from its seed, the dataset name, and a per-column key, so adding a column does not
change the others. `materialize` writes each table as a directory of zstd parquet part files, plus `manifest.json`
and `meta.json` with both seeds, a **logical hash** of each table, and a combined `content_hash` over the seeds,
the table hashes, and the manifest. The logical hash
(`profiler_bench.catalog.hashing`) covers the Spark schema and every value in a canonical row order, so it does
not depend on the parquet writer, compression, file layout, or row order: rewriting a table with Spark at any
partition count gives the same hash. A cached instance is re-hashed before use and regenerated if its manifest, metadata,
or files no longer match; pass `check_hash=False` (CLI `--skip-hash-check`) to skip that full read at the `m` tier. The cache is
`profiler-bench/.cache/catalog` (gitignored), or `$PROFILER_BENCH_CACHE`:

```
<cache>/<dataset>/<tier>/seed-<structure seed>/sample-<sample seed>/<fingerprint>/{<table>.parquet, manifest.json, meta.json}
```

The fingerprint hashes the generator source, the shared catalog code, and the numpy, pandas, pyarrow, and pyspark
versions, so editing a generator or upgrading a library invalidates the cache. Logical hashes do not depend on the
parquet writer, but a new numpy version may change the generated values and so the hashes.

### Public datasets

`profiler_bench.catalog.public` downloads a registered real dataset once, verifies its SHA-256, and returns None when
the URL cannot be reached, so callers can skip it offline. Real datasets have no planted truth and carry no manifest.
None are registered yet.

## Reference sampler

The sampler reconstructs deterministic PySpark DataFrames from a profile without using dbldatagen. It consumes the
richest available marginal statistics, cardinality and string information, nested child profiles, keys, relationships,
constraints, and pairwise dependencies, then reports the fields used, approximated, or unmet for every column.

```python
from profiler_bench.sampler import sample

result = sample(profile, spark, seed=0, scale=1.0)
frames = result.tables
coverage = result.report
```

Pearson and Spearman dependencies use a Gaussian copula with a nearest-positive-semidefinite correlation matrix.
Cramér's V is approximated by replacing the requested share of sampled right-hand categories through a deterministic
left-to-right category mapping; the requested coefficient controls the mixture strength but is not guaranteed to equal
the resulting coefficient for imbalanced marginals.
Expression constraints use bounded oversampling followed by Spark SQL filtering; any unmet expression is recorded rather
than failing the sample.

The reference sampler is designed for benchmark tiers up to `s` (about 100,000 rows and the catalog's roughly
200-column wide table). Tier `m` measures profiler speed and is not a supported fidelity-sampling workload.

## RSI runner and evaluator

`profiler_bench.runner` compares a frozen **baseline** profiler build with a **candidate** build under a frozen
harness. Each build is a git worktree plus commit; the worker imports `profiler/src` from that worktree via
`PYTHONPATH` in a **fresh subprocess**. Speed samples time `profile_tables` (driver wall, process CPU, peak RSS)
after an untimed warmup, rotate baseline/candidate order across cases, and record profile hashes. Fidelity samples
take one profile per build/dataset at the fidelity tier and score it with a `Scorer` for K **paired** sampler seeds.
`--control` runs the baseline against itself. Fidelity metrics are not on this branch yet; inject a `Scorer`
(tests use `FakeScorer`). `try_real_scorer()` wires to sampler + fidelity when both import.

`profiler_bench.evaluate` applies `profiler-bench/policies/default.json`: paired bootstrap CIs (default B=2000),
accuracy-then-speed gates, and a hard privacy guard. Failed or missing samples block acceptance; a crossing interval
is `inconclusive`. Results land in `profiler/rsi/experiments/<id>/` as `run.json`, `samples/*.json`, `summary.json`,
`verdict.json`, `REPORT.md`, and `benchmark_entry.json` (shaped for `rsi_log.py append` kind `benchmark`).

```bash
uv run --package dbldatagen-profiler-bench python -m profiler_bench.rsi run \
  --baseline /path/to/baseline --candidate /path/to/candidate \
  --policy profiler-bench/policies/default.json --experiment-id demo [--datasets retail numeric] [--control]
uv run --package dbldatagen-profiler-bench python -m profiler_bench.rsi evaluate \
  profiler/rsi/experiments/demo
```

**Measured:** driver wall / CPU / peak RSS for `profile_tables`; profile content hash; paired reconstruction,
structure, and privacy scores. **Not measured:** Spark JVM/executor RSS, GPU time, network/disk IO, cluster-scale
Spark, and end-to-end dbldatagen generation from the profile (see `not_measured` in the policy and each `REPORT.md`).

## Development

```bash
make bench-test    # pytest profiler-bench/tests (m-tier tests are marked slow; add -m slow to run them)
make bench-lint    # black, ruff, mypy
```
