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
| `r` | 25,000 | screening stage (cheap pre-filter) |
| `s` | 100,000 | default RSI loop / official fidelity tier |
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

### Real datasets

`profiler_bench.catalog.real_datasets` registers four real datasets (`REAL_DATASETS`). Each declares its source
tables, the export SQL (`SELECT * FROM <source> ORDER BY <primary key>`, or `ORDER BY ALL` without one), the expected
row counts, the schema, and the logical hash of every exported table, pinned after the first export. Data is
never committed. It lives outside every checkout, so it survives worktree cleanup: under `$PROFILER_BENCH_REAL_CACHE`,
or `~/.cache/profiler-bench/real` by default, as `<dataset>/<tier>` (`real.real_cache_dir`). An explicit cache root
(`--cache-dir`, `ExperimentConfig.cache_dir`) uses `<root>/real` instead.

```bash
export PROFILER_BENCH_DATABRICKS_PROFILE=<profile> PROFILER_BENCH_DATABRICKS_WAREHOUSE_ID=<warehouse id>
python -m profiler_bench.catalog real --tier full   # export, check rows, schema, and pinned hashes
python -m profiler_bench.catalog real               # tier s, derived from the cached export
```

`real.materialize(dataset, tier)` exports through the Statement Execution API (`EXTERNAL_LINKS`, `ARROW_STREAM`). It
requests each chunk's link just before downloading the chunk; an expired link is requested again, up to three times
with backoff. Each table's source `SELECT COUNT(*)`, the rows received, and the declared row count must agree, and the
schema and pinned hash must match.

| Variable | Meaning |
|---|---|
| `PROFILER_BENCH_DATABRICKS_WAREHOUSE_ID` | SQL warehouse to export from (required) |
| `PROFILER_BENCH_DATABRICKS_PROFILE` | `~/.databrickscfg` profile for the host, with the token from `databricks auth token`; defaults to `DATABRICKS_CONFIG_PROFILE`, then `DEFAULT` |
| `DATABRICKS_HOST`, `DATABRICKS_TOKEN` | used instead of the profile when both are set |

If `databricks auth token` fails, unset a stale `DATABRICKS_WORKSPACE_ID`. Without a warehouse or credentials, or on
any network or authentication failure, the export raises `Unavailable` with the reason and the command line prints
`skipped <dataset>: <reason>`. A cached tier needs neither.

| Tier | Contents |
|---|---|
| `full` | the export (pinned logical hashes unchanged) |
| `s` | a deterministic root-entity subset capped at `REAL_SUBSET_CAP` (~50,000 rows): each root table over that cap keeps the rows whose primary key (all non-nested columns without one) hashes below `cap / rows`; descendants are not independently capped and remain only when they still reference a kept parent through a declared foreign key. After the benchmark root-entity split, the same cap applies to each part's root set |

Real data has no planted truth. Its manifest is marked `ground_truth: none` and claims only the schema, row counts,
documented primary, unique, and foreign keys that hold in the data, and privacy classes; it makes no distribution
claims. `verify` checks just those, and structure scoring is skipped: it is recorded as `not_applicable`, never 0.

For benchmarking, `real.split` divides each export into three parts (`real.SPLIT`, version 2, recorded in `run.json`
and the baseline cache key). `profiled` is the only part the profiler sees; `holdout` is the privacy baseline;
`ceiling` stands in for the synthetic data, so the ceiling is the reconstruction of `ceiling` against `profiled`. The
split follows root entities (`RealDataset.split_roots`: `customer` for TPC-DS, `users` and `hosts` for Wanderbricks;
by default, the tables that reference no other table). Every row has one home part:

- a row of a root table, or of a table not connected to a root, gets the part `blake2b-64("split", table, key) mod 3`,
  where the key is the primary key (all non-nested columns without one);
- any other row takes the home of the parent row behind its first non-null foreign key, in declared order, to a table
  connected to a root (a sale goes with its customer, a booking with its user).

Then, children first, every parent row that a row of a part references is copied into that part. So each part
satisfies the declared foreign keys and no row is dropped, but parts overlap on copied rows. `split_report.json` in
`<dataset>/split-v2/` (and `run.json` `real_split_report`) gives each table's rows, share of the export, and copied
rows per part. On the current exports, every table keeps at least 31.5% of its rows in every part. Copies are heavy in
shared dimensions: TPC-DS `item` and Wanderbricks `properties`, `amenities`, and `destinations` are in every part in
full, `customer_address` at 98%. They also reach sensitive tables: 28% of the holdout's TPC-DS `customer` rows, 42% of
Wanderbricks `hosts`, and all of `properties` are also in `profiled`. The privacy checks that compare synth against
the holdout use only holdout rows absent from `profiled` (see the privacy guard below).
Tier `s` of a part is the tier-s subset of that part. Parts are cached under `<dataset>/split-v2/<part>/<tier>`.

| Dataset | Source | Licence and terms |
|---|---|---|
| `tpcds_sf1` | `samples.tpcds_sf1`: `customer`, `customer_address`, `date_dim`, `item`, `store_sales`, `web_sales` | Internal use, no stated licence. TPC caveat: the data comes from the TPC-DS generator, and results measured on it are not TPC-DS benchmark results and must not be published as such (TPC fair-use policy) |
| `wanderbricks` | `samples.wanderbricks`, all 16 tables | Internal use, no stated licence |
| `nyctaxi_trips` | `samples.nyctaxi.trips` | Internal use, no stated licence |
| `simulated_retail` | Marketplace listing *Simulated Retail Customer Data* (provider Databricks), installed as catalog `simulated_retail_customer_data`; schema `v01`: `customers`, `sales`, `sales_orders` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) under the [Marketplace consumer terms](https://cms.databricks.com/sites/default/files/2023-01/marketplace-consumer-terms_0.pdf). Attribution: "Simulated Retail Customer Data" by Databricks, Databricks Marketplace, CC BY 4.0 |

### Public datasets

`profiler_bench.catalog.public` downloads a dataset from a URL once, verifies its SHA-256, and returns None when the
URL cannot be reached, so callers can skip it offline. None are registered.

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

The profile scans (`profile.sensitive_value_leaks` and `profile.k_threshold`) skip only canonical structural
strings, each verified on its own terms (`privacy._structural_paths`):
- a table or column name equal to the real schema, including references to one (key, dependency,
  quasi-identifier, relationship, and constraint columns);
- a `source_type` equal to the real column's `simpleString()`;
- a value the model defines for that enum or `Literal` field;
- the running engine's name and version.

Everything else is scanned, including patterns, top values, enumerations, extrema, and free strings, so a real value
planted in a pattern, a top value, or a column name the real schema lacks still fails. A sensitive value counts once
for each profile string that holds it. The concatenation of all strings counts only for values no single string
holds, the sampled text scan counts only values the full scan missed, and a check's value counts hits, not the
`<reason>:<path>` detail keys that locate them.

**Real datasets** (`--suite real`) adjust the data checks:
- The holdout baseline is the holdout part's rows that equal no profiled row (`privacy.independent_rows`).
- Copy and partial-copy checks are a membership test on the non-key columns: the synth's match rate against the
  profiled rows minus its match rate against the independent holdout rows. The larger reference is first cut to the
  size of the smaller one by a seeded sample. A sampler that knows only the distribution, dependencies between
  columns included, matches seen and unseen rows alike; one that memorised profiled rows matches them more.
- Tables with no non-key columns (key-only bridge tables) compare the synth with the profiled part with each column
  shuffled independently (`privacy.shuffled_columns`, seeded by the sampler seed), labelled `shuffled_control`.
- **Power check.** Match rates are means over synth rows, so replacing 5% of the synth's rows with verbatim profiled
  rows gives each rate `0.95 * rate + 0.05 * copied rate` (the shuffled control's rate does not move). The check is
  rerun on those expected rates, with the pooled rate and noise allowance recomputed under that alternative. If it
  would still pass, a pass could not rule out such a copy: the check is reported unmeasured with `underpowered`
  (never as a pass), and a failure stays a failure. This happens on tiny tables, on tables whose non-key columns
  repeat across rows, and when few independent holdout rows remain.
- A table whose baseline has fewer than max(k, 100) independent rows reports those checks as unmeasured
  (`GuardCheck.measured` false): recorded in `privacy_unmeasured`, neither a pass nor a fail.
- **Privacy status** (`PrivacyScore.status`, the record's `privacy_status`): `fail` when a measured check failed;
  `pass` only when, in addition, at least one copy, partial-copy, or distance-to-closest-record check was measured;
  otherwise `unmeasured`. A guard that measured nothing about the rows never passes.
- Each record's `holdout_overlap` gives, per table, the holdout rows and the share also in `profiled`. The
  `real_holdout` gate lists each dataset's privacy status and its unmeasured, underpowered, and shuffled-control
  checks.

Known gap: on a key-only table, a 5% copy can stay below the shuffled chance level when the rest of the synth matches
real key pairs less often than the shuffle does (Wanderbricks `property_amenities`). The power check then reports
the check underpowered rather than passed; exact copies are caught. The profile carries only aggregates, so
reproducing key edges requires memorising rows, which the exact-copy control catches. The synthetic suite is
unchanged: its status is `pass` or `fail` as before.

Profile regex and text privacy is layered (synth-data leakage is the last):

1. **Closed strings.** Every string in the serialized profile JSON must be exactly one of: a model enum value; a
   lossless numeric/decimal encoding; an ISO-8601 temporal value; a table or column name that exists in the real
   schema; a Spark `source_type` equal to that column's `simpleString()`; a regex in the allowlisted dialect (pattern
   fields only); an engine name/version (or expression `dialect`) matching `^[a-z0-9_.-]{1,64}$`; a source
   name/format/location matching `^[a-zA-Z0-9_./:+-]{1,512}$`; a check-constraint `expression` when
   `privacy.allow_expression_constraints` is true (still covered by the raw-text and all-sensitive-value scans); or a
   key in a model-defined map (`key` / `value` / `element`). Any other string fails with reason
   `unexpected_profile_string`, naming the JSON path and never the value. Unknown extension fields are allowed only
   when they hold non-string scalars. `top_values` and enumeration literals stay under the k-threshold audit instead.
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
   any candidate text; and candidate factors that are substrings of sensitive values. Independently, every
   dialect-accepted non-structural pattern branch is fullmatched against all sensitive values in Spark; matched
   values are aggregated to distinct `(branch, length)` pairs (length capped at `max_regex_length`), and each pair's
   language size is counted once on the driver under the shared budget. A count below max(k, `min_regex_language`)
   fails with reason `sensitive_value_in_profile`, naming the sensitive column, JSON path, and branch index — never
   the value. Budget exhaustion fails conservatively. The sampled per-value regex language check is kept as well.
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

When only min/max/mean/stddev are available, numeric draws use a truncated normal (inverse-CDF within the observed bounds) rather than clipping, so exact profiled extrema are not piled as point masses.
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
harness. Each build is a git worktree plus commit; the **speed** worker imports that worktree's `profiler/src` via
`PYTHONPATH` in a **fresh subprocess**. Speed samples time `profile_tables` (driver wall, process CPU, peak RSS)
after an untimed warmup, rotate baseline/candidate order across cases, and record profile hashes under the policy's
timed Spark config (unchanged by fidelity settings). The default policy (version 2.2.0) times speed on `local[8]` with
an 8 GiB driver (other confs unchanged; 1.x used `local[2]` and 2 GiB). Speed results under 1.x are not comparable
with 2.0.0; old experiments re-evaluate against the policy recorded in their own `run.json`. The speed baselines in
`profiler/rsi/BASELINE.md` were measured under 1.x and must be re-measured (P1.8e). The nested tier-m OOM at 2 GiB is
no longer the binding limit, so rsi-0007 targets wall time rather than completion. Speed and data-volume proposals take
N=15 speed samples per build and case for tiers `s` and `m`. Accuracy proposals, whose speed gates are off, take only
a smoke check from `speed_samples_accuracy` (2 per build and case at tier `s`, none at `m`). `run.json` records the
count actually used as `speed_samples_effective`, and the evaluator's expected matrix uses it (runs recorded before
this field use the policy's `speed_samples`). Fidelity samples take one
profile per build/dataset at the fidelity tier and score it with the reference sampler and fidelity metrics for K
**paired** sampler seeds. Each
`(build, dataset)` cell runs in its own fidelity worker subprocess with one SparkSession reused across all K seeds
(profile in via file, score JSON out via file; cross-cell isolation and per-seed failure records unchanged; default
per-cell timeout 45 minutes via `ExperimentConfig.fidelity_cell_timeout_s`, recorded as `fidelity_cell_timeout_s`).
Results are byte-identical to creating a fresh session per seed; set `PROFILER_BENCH_FIDELITY_SESSION=fresh` only for
equivalence or timing measurements. Speed samples keep one fresh process per sample so timing does not change. On
timeout the parent SIGTERMs then SIGKILLs the worker process group so orphan JVMs cannot survive. Builds rotate per dataset the same way speed
samples do. Up to `fidelity_parallelism` cells (policy default 3) run at once, each on `local[4]` with an 8 GiB
driver; each cell writes only its own sample files and records keep the serial order, so results are byte-identical
to serial execution. Fidelity always starts after every speed sample has finished, and speed samples run one at a
time, so timing never shares the machine with other benchmark work. A crash, timeout, or invalid worker payload in one cell cannot affect any other cell; that cell's seed
records are `ok=false` with `error_class` (`WorkerTimeout`, `WorkerCrashed`, `WorkerOutputInvalid`,
`WorkerOutputMismatch`, or `MissingProfileError`). Fidelity workers use a fixed Spark config recorded in `run.json`
as `fidelity_spark` (default `spark.driver.memory=8g`, override with `ExperimentConfig.fidelity_driver_memory`;
shuffle partitions sized for `local[N]`). Fidelity workers deliberately parse profiles with the **frozen harness**
`dbldatagen_profiler.model` (`run.json` records `fidelity_model_source: harness`): the harness, sampler, and scorer
are frozen per experiment, and a candidate's profile JSON is the only candidate input. Speed workers keep importing
the candidate profiler from its worktree. Each result includes the full reconstruction, structure, privacy, and
real-vs-real ceiling detail. Ceilings are cached by dataset, tier, and harness hash in the experiment cache under an
fcntl lock with atomic writes. Baseline-build samples are also cached across experiments in
`profiler/rsi/experiments/.cache/baseline/` (git-ignored). Fidelity entries (the baseline profile and its K per-seed
scores) are reused for any proposal type. Speed entries (all N baseline samples of one dataset and tier) are reused
only for accuracy proposals; speed and data-volume proposals always time both builds fresh in the same run, with
rotated order. Keys cover the baseline `profiler/src` hash, harness hash, sampling-policy hash, metrics config hash, dataset,
tier, dataset seed, Spark config hash, the catalog generator fingerprint and instance content hash (or the pinned
hashes and split method of a real dataset), and the Python, PySpark, NumPy, pandas, scikit-learn, and pyarrow versions. Speed keys add
the Java version, sample count, and machine; fidelity keys add the sampler seeds and scorer. Any mismatch, corrupt
file, or failed sample recomputes. Gate thresholds are excluded from the sampling-policy hash, so cached samples can be
reevaluated under new gates; sample counts, seeds, tiers, dataset seed, and Spark configuration remain key inputs. Writes
use an fcntl lock with atomic replace. `run.json` records `baseline_cached`,
`baseline_cache_parts` (whether speed caching was allowed and whether speed or fidelity came from the cache),
`baseline_cache_sources`, and `baseline_cache_hits`. Pass `--no-baseline-cache` to measure the baseline fresh. The
candidate role never reads the cache, including in control runs. `--control` runs the baseline against itself. Unmeasured privacy checks keep the
sample successful with `privacy_passed=false` and are listed in `privacy_unmeasured`, so the evaluator's hard guard
rejects them. Tests can explicitly inject a `FakeScorer`; command-line runs never use it.

`profiler_bench.evaluate` applies `profiler-bench/policies/default.json`. The accuracy mean estimand is the equally
weighted mean across datasets of each dataset's mean paired-seed difference; its CI uses a **stratified bootstrap**
that resamples seeds within each dataset (datasets are fixed strata). A hierarchical bootstrap that also resamples
datasets is a documented alternative and is not used. Per-dataset regression gates keep a within-dataset bootstrap.
An accuracy proposal may instead declare `targets.datasets`. It is accepted when either the mean gate passes or every
declared target improves relative reconstruction by at least `accuracy.min_target_improvement` with a paired per-dataset
CI lower bound above zero. The existing per-dataset regression, structure, privacy, completeness, and control gates apply
to both paths; without structured targets, only the mean path is available. The verdict records the accepting
`accuracy_path` and the targeted gate records each target CI.
Each gate reports a two-sided 95% percentile interval with **no family-wise guarantee** across gates. Privacy is a
hard guard (not policy-disableable). Completeness is checked against the expected matrix from `run.json`. Control runs
and identical `profiler_src_hash` builds can never be `accepted`. Results land in `profiler/rsi/experiments/<id>/` as
`run.json`, `samples/*.json`, `summary.json`, `verdict.json`, `REPORT.md`, and `benchmark_entry.json` (shaped for
`rsi_log.py append` kind `benchmark`).

### Real holdout suite

`--suite` (`ExperimentConfig.suite`) picks the datasets: `synthetic` (default, the catalog generators; behaviour and cache keys
as before), `real` (the four real datasets), or `all`. Before any timing, each requested real dataset's parts are
materialized at the fidelity tier. A dataset whose export is neither cached nor exportable (no warehouse, rejected
credentials, offline), fails its pinned checks, or has no such tier (real datasets have only `s` and `full`) is
skipped with the reason in `run.json` `real_skipped`; the run continues. Scored real datasets are listed in
`real_datasets` and take speed samples only at the fidelity tier, for their profile.

The evaluator treats the real suite as a **holdout**, not part of acceptance: real samples are excluded from
completeness, privacy, accuracy mean, per-dataset, structure, and speed gates, and checked only by the `real_holdout`
gate. It reports each real dataset's baseline and candidate relative reconstruction with bootstrap CIs, the paired
regression CI, and the candidate's privacy status. A candidate sample that fails privacy on a real dataset
**rejects** (`real_holdout_privacy_failed`). A candidate sample whose privacy is `unmeasured` makes the verdict
**inconclusive** (`real_holdout_privacy_unmeasured`), never accepted. So does a regression CI whose upper bound
exceeds `accuracy.max_dataset_regression` (`real_holdout_regression`), and failed or missing real samples
(`real_holdout_incomplete`). With every real dataset skipped the gate passes as `real_holdout_skipped`. A
`real`-only run has no synthetic pairs, so it can measure but never accept.

```bash
uv run --package dbldatagen-profiler-bench python -m profiler_bench.rsi run \
  --baseline /path/to/baseline --candidate /path/to/candidate \
  --policy profiler-bench/policies/default.json --experiment-id demo [--datasets retail numeric] [--control] \
  [--suite synthetic|real|all]
uv run --package dbldatagen-profiler-bench python -m profiler_bench.rsi screen \
  --baseline /path/to/baseline --candidate /path/to/candidate \
  --policy profiler-bench/policies/default.json --experiment-id demo-screen
uv run --package dbldatagen-profiler-bench python -m profiler_bench.rsi evaluate \
  profiler/rsi/experiments/demo
```

### Screening stage

Accuracy proposals can run a cheap **screen** before the official tier-`s` run: `python -m profiler_bench.rsi screen`
(or `run --stage screen`). The policy's `screen` section (tier `r`, 3 fidelity seeds, margin 0.5, 2 smoke speed
samples on `r`) overrides sampling for that experiment; `run.json` records `stage: screen` and the effective policy
(including loosened accuracy thresholds). Screening uses the synthetic catalog only and never returns `accepted`:
verdicts are `screen_pass` (would plausibly pass under the loosened margins, including borderline / inconclusive
cases) or `screen_fail` (a rejection, including any privacy failure). A `screen_pass` proceeds to the full official
run (tier `s`, 5 seeds, and the real holdout suite when that suite is enabled). Old experiments still re-evaluate
under the policy recorded in their own `run.json`.

### Data-volume objective

`profiler-bench/policies/data_volume.json` freezes a second RSI objective: the least input fraction `n*` that still
reaches a per-dataset accuracy target `T_d`. The harness, not the profiler, chooses the input. For each fraction
`f` on the geometric grid `(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)` and each subset seed, it
builds a deterministic **FK-preserving uniform** subset: sample **every** table independently at `f` with a seeded
hash of the primary key (or full row content when there is no PK), then take the **upward closure** under foreign
keys (add every parent row referenced by a selected child, including composite keys, self-references, and cycles)
until a fixpoint; method name `fk_preserving_uniform`. Closure truncates Spark lineage each iteration and is capped
by policy `data_volume.max_closure_iterations` (default 64, recorded in `run.json`); exceeding the cap fails
loudly. Realized per-table and overall input fractions are recorded alongside the grid `f` that indexes `n*`
(gates stay on nominal `f`; `REPORT.md` / `benchmark_entry.json` show both). Every non-empty table keeps at least
one row. The profiler sees only that subset and is not told the full row counts. Reconstruction is scored against
the **full** real tables; before sampling, each profiled `row_count` is scaled by `full_rows / profile_rows` for
that table (equivalent to calling the reference sampler with a per-table scale) so synthetic row counts match.
Distinct counts, cardinality, FK fan-out, and extrema remain the profiler's job to extrapolate.

`T_d` and the control noise width are frozen per dataset by `calibrate-targets` from a same-build control at
`f = 1` (K × S seeds), with provenance (commit, harness hash, run id). Calibration refuses to write unless
`run.control` is true, baseline/candidate `profiler_src_hash` match, the f=1 matrix is complete (every cell ok),
and privacy passed on every f=1 record. `run --mode data-volume` refuses to start unless every selected dataset
has both. Compute a proposed policy with:

```bash
uv run --package dbldatagen-profiler-bench python -m profiler_bench.rsi calibrate-targets \
  --baseline profiler/rsi/experiments/<control-at-f1> \
  --output /tmp/data_volume.proposed.json
```

Polly freezes the accepted file. The measured curve reports relative reconstruction at every grid fraction over
`S` subset seeds and `K` sampler seeds; `n*_d` is the smallest fraction whose paired-bootstrap CI lower bound on
the mean clears `T_d` minus the frozen control noise width. If no fraction reaches `T_d`, `n* = 1.0` and the
dataset is flagged. A candidate privacy failure anywhere rejects; a baseline privacy failure is inconclusive
(`baseline_privacy`). Completeness covers both builds. The `f = 1` reconstruction regression bound is loaded by
reference from `default.json` (`accuracy.max_dataset_regression`), with that file's hash recorded on the gate.

A `data_volume` proposal is accepted when the geometric mean of `n*` decreases by at least one grid step, at least
one declared target improves by one grid step, no dataset's `n*` increases, reconstruction at `f = 1` does not
regress beyond the default bound, candidate privacy passes, and every expected sample is present on both builds.
Crossing intervals are inconclusive. Control runs are never accepted.

```bash
uv run --package dbldatagen-profiler-bench python -m profiler_bench.rsi run --mode data-volume \
  --baseline /path/to/baseline --candidate /path/to/candidate \
  --policy profiler-bench/policies/data_volume.json --experiment-id demo-dv \
  [--datasets retail numeric] [--fractions 0.1 0.5 1.0] [--subset-seeds 1] [--fidelity-seeds 1] [--control]
```

**Measured:** driver wall / CPU / peak RSS for `profile_tables`; profile content hash; paired reconstruction,
structure, and privacy scores. **Not measured:** Spark JVM/executor RSS, GPU time, network/disk IO, cluster-scale
Spark, and end-to-end dbldatagen generation from the profile (see `not_measured` in the policy and each `REPORT.md`).

## Development

```bash
make bench-test    # pytest profiler-bench/tests (m-tier tests are marked slow; add -m slow to run them)
make bench-lint    # black, ruff, mypy
```

Tests marked `network` export one tiny real table from a warehouse. They are skipped unless selected with
`-m network`, and skip themselves when `PROFILER_BENCH_DATABRICKS_WAREHOUSE_ID` is unset.
