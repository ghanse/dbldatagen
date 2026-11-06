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

## Fidelity metrics

`profiler_bench.fidelity` scores a profiler from three angles. Every score is in [0, 1] (1 is perfect), is a
pydantic model that round-trips through JSON with full per-column, per-pair, per-check detail, and carries the
version and SHA-256 of the metrics configuration that produced it.

```python
from profiler_bench.fidelity import ceiling, overall, score_privacy, score_reconstruction, score_structure

recon = score_reconstruction(real, synth, truth, seed=0)  # synthetic tables vs real tables
structure = score_structure(profile, truth)               # profile vs the planted ground truth
privacy = score_privacy(real, synth, profile, truth)      # guard: privacy.passed
top = ceiling("retail", "s", spark=spark)                 # real seed 0 vs real seed 1
recon.relative_to(top)                                    # score as a fraction of the achievable
```

All functions need `spark.sql.session.timeZone=UTC`. Only bounded, content-hash-seeded samples reach the driver
(`sampling.max_rows` per table and side); row counts, null fractions, distinct counts, constraints, keys, and foreign
keys are Spark aggregates over the full tables.

### Configuration

[`metrics_config.json`](src/profiler_bench/fidelity/metrics_config.json) holds every sample size, tolerance,
threshold, and weight. It is frozen per experiment: changing a value is a benchmark change and needs a new `version`.

### Reconstruction

Each table's score is the weighted mean of the groups present in that table, renormalized over those groups. A
dataset's score is the mean of its table scores.

| Group | Weight | Per item |
|---|---|---|
| `schema` | 0.05 | mean of the share of columns with matching name and type, and `(min(rows)+1)/(max(rows)+1)` |
| `marginal` | 0.25 | numeric: 1 - KS and 1 - normalized Wasserstein-1 (clipped to the real 1–99% range widened by that range, divided by it, capped at 1); categorical (at most 20 distinct real values, or boolean): 1 - TVD over the top 20 real values plus `other`; strings: TVD, KS and Wasserstein of lengths, and 1 - TVD over the manifest formats plus `unmatched`; floats: 1 - TVD of the NaN/inf/-inf shares; arrays, maps, binary: length KS and Wasserstein, with elements, keys, and values scored as their own leaves |
| `null_fraction` | 0.10 | 1 - absolute null fraction error |
| `distinct` | 0.10 | `(min+1)/(max+1)` of the distinct counts, i.e. exp(-abs(log ratio)) |
| `dependence_declared` | 0.10 | 1 - abs(delta) of the manifest's declared measure for each declared pair |
| `dependence_random` | 0.05 | 1 - abs(delta) over 20 seeded random pairs: Spearman (numeric), Cramér's V (categorical), correlation ratio (mixed) |
| `constraints` | 0.10 | share of synth rows satisfying each check; for functional dependencies, the largest share of rows that can be kept so the dependency holds |
| `keys` | 0.05 | distinct non-null keys over rows (primary) or over rows with a key (unique) |
| `foreign_keys` | 0.10 | mean of 1 - abs(coverage error), 1 - fan-out KS, and 1 - fan-out Wasserstein |
| `discriminator` | 0.10 | 1 - abs(AUC - 0.5) * 2 for a k-nearest-neighbour (k=10) classifier on 2,000 rows per side, half train, half test |

A column or table the synth lacks, or a column with a different type, scores 0. The **ceiling** scores catalog seed 0
against seed 1, which shows how far apart two real draws already are.

### Structure recovery

Each planted item earns credit in [0, 1]. Recall is the mean credit over planted items; precision is the mean credit
over the profile's claims about the same columns. A category scores F1, or recall when there are no claims. Missing
structure is reported as `not recovered`, never as an error.

| Category | Weight | Recovered when |
|---|---|---|
| `keys` | 0.15 | the profile's primary or unique key has the same column set |
| `foreign_keys` | 0.15 | a relationship has the same tables and columns |
| `checks` | 0.15 | ranges: each bound within 5% of the planted span; enumerations: Jaccard of the values; comparisons: same columns and an operator that implies the planted one; expressions: identical after normalizing case, whitespace, backticks, and outer parentheses |
| `functional_dependencies` | 0.10 | the same determinant and dependent sets (spec 0.1.0 has no field for them) |
| `dependencies` | 0.15 | same measure and columns, value within 0.05 + 4(1 - v²)/√n |
| `patterns` | 0.10 | 200 strings generated from each regex match the other at a rate of at least 0.95, in both directions |
| `null_rates` | 0.10 | within the catalog's verification tolerance (exact for 0 and 1) |
| `cardinality` | 0.10 | distinct count within exp(±0.05) of a planted exact count; unique columns marked unique; capped domains not exceeded |

### Privacy guard

Data checks compare the synth with a **real holdout**, the same dataset at seed `manifest.seed + 1`, which the
profiler never saw. A rate check passes when the synth exceeds the holdout by at most the threshold plus three
standard errors. Reports carry counts and rates, never values.

| Check | Passes when |
|---|---|
| `<table>.copy_rate_excess` | share of synth rows identical to a real row exceeds the holdout's by at most 0.01 |
| `<table>.dcr_ratio` | median distance to closest real record, synth over holdout, is at least 0.8 |
| `<table>.<column>.value_leak_rate_excess` | share of distinct sensitive real values found in the synth exceeds the holdout's by at most 0.001 |
| `<table>.quasi_identifiers[...].match_rate_excess` | share of rows matching a real-unique quasi-identifier combination exceeds the holdout's by at most 0.01 |
| `profile.sensitive_value_leaks` | no sensitive real value equals, or (when at least 6 characters long) occurs in, any string of the profile JSON or any literal run of its regexes |
| `profile.k_threshold` | the profile validates, and every `top_values` entry and enumeration value occurs in at least `privacy.min_value_count` real rows |

Sensitivity categories and quasi-identifier sets come from the manifest through `profiler_bench.fidelity.truth`.

## Development

```bash
make bench-test    # pytest profiler-bench/tests (m-tier tests are marked slow; add -m slow to run them)
make bench-lint    # black, ruff, mypy
```
