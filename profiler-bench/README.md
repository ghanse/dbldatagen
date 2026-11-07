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

## Fidelity metrics

`profiler_bench.fidelity` scores a profiler from three angles. Every score is in [0, 1] (1 is perfect), is a
pydantic model that round-trips through JSON with full per-column, per-pair, per-check detail, and carries the
version and SHA-256 of the metrics configuration that produced it.

```python
from profiler_bench.fidelity import ceiling, overall, score_privacy, score_reconstruction, score_structure

recon = score_reconstruction(real, synth, truth, seed=0)  # synthetic tables vs real tables
structure = score_structure(profile, truth)               # profile vs the planted ground truth
privacy = score_privacy(real, synth, profile, truth)      # guard: privacy.passed
top = ceiling("retail", "s", spark=spark)                 # two real row samples of one structure
recon.relative_to(top)                                    # per-group fraction of the achievable
```

All functions need `spark.sql.session.timeZone=UTC`. Only bounded, content-hash-seeded samples reach the driver
(`sampling.max_rows` per table and side); row counts, null fractions, distinct counts, constraints, keys, and foreign
keys are Spark aggregates over the full tables.

### Configuration

[`metrics_config.json`](src/profiler_bench/fidelity/metrics_config.json) holds every sample size, tolerance,
threshold, and weight. It is frozen per experiment: changing a value is a benchmark change and needs a new `version`.

### Reconstruction

Each table's score is the weighted mean of the groups present in that table, renormalized over those groups, and
`TableReconstruction.weights` records the effective weights. Because a table without, say, foreign keys spreads that
weight over its other groups, two tables weight the same group differently: compare tables group by group rather than
by total score. A dataset's score is the mean of its table scores.

| Group | Weight | Per item |
|---|---|---|
| `schema` | 0.05 | mean of the share of columns with matching name and type, and `(min(rows)+1)/(max(rows)+1)` |
| `marginal` | 0.25 | numeric: 1 - KS and 1 - normalized Wasserstein-1 (clipped to the real 1–99% range widened by that range, divided by it, capped at 1); categorical (at most 20 distinct real values, or boolean): 1 - TVD over the top 20 real values plus `other`; strings: TVD, KS and Wasserstein of lengths, and 1 - TVD over the manifest formats plus `unmatched`; floats: 1 - TVD of the NaN/inf/-inf shares; arrays, maps, binary: length KS and Wasserstein, with elements, keys, and values scored as their own leaves |
| `null_fraction` | 0.10 | 1 - absolute null fraction error |
| `distinct` | 0.10 | `(min+1)/(max+1)` of the distinct counts, i.e. exp(-abs(log ratio)) |
| `dependence_declared` | 0.10 | 1 - abs(delta) of the manifest's declared measure for each declared pair |
| `dependence_random` | 0.05 | 1 - abs(delta) over 20 seeded random pairs: Spearman (numeric), Cramér's V (categorical), correlation ratio (mixed). High-cardinality strings join the pairs through derived features: `#format` (index of the manifest format matched), `#chars` (length), and `#digits`, `#letters`, `#punctuation` counts; derived features of the same column are not paired with each other. Up to 20 pairs of high-cardinality string columns also score value agreement: 1 - abs(exact-equality rate error) and the KS and Wasserstein distances of the per-row shared-prefix length, so two equal UUID columns or a copied id within a table must stay equal |
| `constraints` | 0.10 | share of synth rows satisfying each check; for functional dependencies, the largest share of rows that can be kept so the dependency holds |
| `keys` | 0.05 | distinct non-null keys over rows (primary) or over rows with a key (unique) |
| `foreign_keys` | 0.10 | mean of 1 - abs(coverage error), 1 - fan-out KS, and 1 - fan-out Wasserstein |
| `discriminator` | 0.10 | 1 - abs(AUC - 0.5) * 2 on the held-out half for a scikit-learn `HistGradientBoostingClassifier` (100 iterations, fixed `random_state`, one thread) on 2,000 seeded rows per side, split in stratified halves; features include the derived string features |

A column or table the synth lacks, or a column with a different type, scores 0. The **ceiling** scores two row
samples of the same dataset structure (same structure seed, sample seeds `s` and `s + 1`), which shows how far apart
two real draws already are. `relative_to(ceiling)` divides each group by the ceiling's same group, caps each at 1,
and then applies the table's effective weights, so a group above its ceiling cannot hide one below it.

### Structure recovery

Each planted item earns credit in [0, 1]. Recall is the mean credit over planted items; precision is the mean credit
over the profile's claims about the same columns. Key, foreign key, check, and dependency claims about structure the
manifest does not plant are checked against the real data (`profiler_bench.fidelity.claims`): fractions within the
catalog's share tolerance, keys exactly, dependencies within the dependency tolerance on a bounded sample. True ones
are left out of precision; false ones count as 0, including on tables with nothing planted in that category (recall
is then 1). A category scores F1, or recall when there are no claims. Missing structure is reported as
`not recovered`, never as an error. `TableStructure.weights` records the effective category weights.

Verifying claims against the real data gives the profiler no new information: it reads the same data when it
profiles. The score must not echo that data either, so a category records only how many unplanted claims held
(`verified_true`) and how many did not (`verified_false`), never the claimed or real values.

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

Data checks compare the synth with a **real holdout**, another row sample of the same structure (sample seed
`s + 2`, distinct from the ceiling's), which the profiler never saw. A rate check passes when the synth exceeds the holdout by at most the threshold plus three
standard errors. Reports carry counts and rates, never values.

| Check | Passes when |
|---|---|
| `<table>.synth_table_present`, `<table>.synth_columns_present`, `<table>.synth_rows_present` | the synth has the table, every real column with the same type, and rows when the real table has rows; a check that cannot be measured because its table or columns are missing is reported as failed, so an empty sampler fails |
| `<table>.copy_rate_excess` | share of synth rows identical to a real row exceeds the holdout's by at most 0.01 |
| `<table>.partial_copy_rate_excess` | for up to 20 seeded column pairs, excluding primary and foreign key columns (their values follow from the key structure), the share of synth rows whose pair of values is unique in the real table exceeds the holdout's by at most 0.01; this catches row-aligned copying of a subset of columns |
| `<table>.dcr_ratio` | median distance to closest real record, synth over holdout, is at least 0.5 (independent samples measured 0.68–1.27; a copy scores 0) |
| `<table>.<column>.value_leak_rate_excess` | share of distinct sensitive real values found in the synth exceeds the holdout's by at most 0.001 |
| `<table>.quasi_identifiers[...].match_rate_excess` | share of rows matching a real-unique quasi-identifier combination exceeds the holdout's by at most 0.01 |
| `profile.sensitive_value_leaks` | no sensitive real value equals a string of the profile JSON or appears in a scanned profile-text blob; every sensitive value (all rows) is checked against a bounded profile candidate set (decoded strings, concat, required literals, and fully enumerated small branch languages) and fails with `sensitive_value_in_profile` on a hit; every profile string is closed (below); every profile regex is in the allowlisted dialect; sensitive-column patterns pass the structural rule below; and no sensitive value in a seeded sample of 2,000 per column is fully matched by a small-language alternation branch of any other profile regex or contains a rare required literal of one |
| `profile.k_threshold` | the profile validates; every `top_values` entry and enumeration value occurs in at least `privacy.min_value_count` (k) real rows; closed-string, dialect, and structural pattern rules pass; and on non-sensitive columns no alternation branch is a small-language match or rare required literal against that column's values |

Profile regex and text privacy is layered (synth-data leakage is the last):

1. **Closed strings.** Every string in the serialized profile JSON must be exactly one of: a model enum value; a
   lossless numeric/decimal encoding; an ISO-8601 temporal value; a table or column name that exists in the real
   schema; a Spark `source_type` equal to that column's `simpleString()`; a regex in the allowlisted dialect (pattern
   fields only); an engine name/version (or expression `dialect`) matching `^[a-z0-9_.-]{1,64}$`; or a key in a
   model-defined map (`key` / `value` / `element`). Any other string fails with reason `unexpected_profile_string`,
   naming the JSON path and never the value. Unknown extension fields are allowed only when they hold non-string
   scalars. `top_values` and enumeration literals stay under the k-threshold audit instead.
2. **Allowlisted dialect.** Every profile regex is accepted only if a strict tokenizer over the *raw source string*
   (not Python's `sre_parse`) recognizes the portable dialect: literals and escaped literals, classes `[...]`
   (including negation and ranges), `.`, `\d`/`\D`/`\w`/`\W`/`\s`/`\S`, quantifiers `?`/`*`/`+`/`{m}`/`{m,}`/`{m,n}`
   (no lazy forms), groups `(...)` and `(?:...)`, alternation, and anchors `^`/`$`. Comments `(?#...)`, inline flags,
   lookarounds, named groups, backreferences, `\p`/`\N`/`\x`/`\u`, verbose whitespace, conditionals, and NUL/control
   characters fail with reason `pattern_not_in_dialect`. The source must equal the concatenation of the token texts so
   ignored bytes cannot hide data.
3. **Raw-text scan.** Independently of regex semantics, every profile string — alone and the concatenation of all
   strings in document order — is searched for every sampled sensitive value of length at least
   `privacy.min_literal_len`, after case-folding and backslash stripping, and again after percent-decoding and
   unicode-unescaping. A hit fails with reason `sensitive_value_in_profile_text`.
4. **Full sensitive-column candidate scan.** A bounded candidate set is built from every decoded profile string, the
   concatenation of all strings, each pattern branch's normalized required literals of length at least
   `privacy.min_literal_len`, and — when a branch's exact language has at most `privacy.max_enumerated_strings`
   (default 10_000) strings — the full enumeration of that language under the shared regex work budget (exhaustion
   fails conservatively). Every sensitive column is then scanned in Spark over **all** rows (case-folded): exact
   semi-join against enumerated strings; sensitive values of length at least `min_literal_len` that are substrings of
   any candidate text; and candidate factors that are substrings of sensitive values. A hit fails with reason
   `sensitive_value_in_profile`, naming the sensitive column and JSON path, never the value. The sampled per-value
   regex language check is kept as well.
5. **Structural generalization (sensitive columns).** Columns the manifest marks `direct_identifier`, `secret`, or
   `free_text` may only carry literal-free patterns (spec 0.2.0). After dropping group syntax and anchors, and folding
   single-character and case-only classes such as `[jJ]` into literals, the pattern must not contain a literal
   character (escaped punctuation counts), an alternation, or a back-reference. Allowed atoms are character classes
   with at least two distinct case-folded members, `.`, `\d` / `\w` / `\s`, and their negations, with any quantifier.
   A violation fails both profile checks with reason `sensitive_pattern_not_generalized`. Generalized email patterns
   must therefore use classes such as `[^\s]+`; forms like `[a-z]+@[a-z]+\.[a-z]+` fail because `@` and `.` are
   literals.
6. **Per-branch size and required literals (other columns).** Alternations are expanded into branches (bounded by the
   shared regex work budget; exhaustion fails with `regex_budget_exceeded`). Each branch is checked on its own: if it
   fully matches a real value and has fewer than max(k, 10⁶) distinct strings of length at most
   min(len(value), `privacy.max_regex_length`), or if it requires a literal of at least 4 characters that occurs in
   fewer than k rows, the regex fails. A decoy such as `(?:john@x\.com|[a-z0-9@.]{10})` is caught by the narrow
   branch even though the whole language equals the broad decoy's.
7. **Synth-data leakage.** Exact-row copy rate, partial-copy pairs, DCR, sensitive value reproduction, and
   quasi-identifier matches compare the synth with a real holdout (see the table above) and remain the final backstop
   when a profile pattern is generalized but a sampler still emits real values.

### Threat model

The privacy guard targets accidental leakage and leakage an optimizer would be rewarded for — anything the sampler can
emit from the profile. Deliberate covert encoding in numeric statistics (for example spelling secrets across least
significant digits) is out of scope. The final backstops are the synth-data leakage checks above and cross-vendor code
review of every RSI change.

Distance to closest record encodes each column so that it adds at most about 1 to the squared distance: a numeric
column is its rank in the real sample (a gap of r adds r², plus 1 for a null mismatch); a categorical column is a
one-hot of the top 20 real values plus `other` and null, scaled by 1/√2 (a mismatch adds 1); a string column adds the
same one-hot, its length rank, and 1 when the exact values differ (up to about 2). Wide tables therefore weigh each
column equally, whatever its type.

Sensitivity categories and quasi-identifier sets come from the manifest through `profiler_bench.fidelity.truth`.

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

`profiler_bench.evaluate` applies `profiler-bench/policies/default.json`. The accuracy mean estimand is the equally
weighted mean across datasets of each dataset's mean paired-seed difference; its CI uses a **stratified bootstrap**
that resamples seeds within each dataset (datasets are fixed strata). A hierarchical bootstrap that also resamples
datasets is a documented alternative and is not used. Per-dataset regression gates keep a within-dataset bootstrap.
Each gate reports a two-sided 95% percentile interval with **no family-wise guarantee** across gates. Privacy is a
hard guard (not policy-disableable). Completeness is checked against the expected matrix from `run.json`. Control runs
and identical `profiler_src_hash` builds can never be `accepted`. Results land in `profiler/rsi/experiments/<id>/` as
`run.json`, `samples/*.json`, `summary.json`, `verdict.json`, `REPORT.md`, and `benchmark_entry.json` (shaped for
`rsi_log.py append` kind `benchmark`).

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
