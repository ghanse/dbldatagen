# dbldatagen-profiler — Project Plan

Shared, living plan for the dataset profiler and its recursive self-improvement (RSI) system.
Every work item has an ID; the RSI audit log (`rsi/log.jsonl`, rendered to `rsi/LOG.md`) references these IDs.

## Goal

Profile one or more datasets (possibly a relational model) into a portable **profile** that captures
the structure needed to regenerate realistic benchmark data for DE / ML / analytics workloads,
without shipping production data. The profile is consumed by dbldatagen (and anything else) to
generate data.

Optimization priorities, in order:

1. **Reconstruction accuracy** — synthetic data generated from the profile matches the source.
2. **Speed** — wall time / CPU / memory to profile.

Guard: **privacy** — the profile must not leak raw rows or rare identifying values.

## Architecture

```
profiler/                     distribution `dbldatagen-profiler` (uv workspace member) — the system under improvement
  PLAN.md                     this file
  spec/profile.schema.json    engine-neutral, versioned profile contract (generated from the model)
  src/dbldatagen_profiler/
    model/                    profile data model (pydantic) — the contract
    engines/spark/            PySpark engine
    cli.py                    `dbldatagen-profile` CLI
  rust/                       Rust engine (Polars or DataFusion), single binary, same contract (deferred)
  rsi/
    RSI.md                    the RSI procedure and roles
    LOG_SCHEMA.md             audit log entry schema
    log.jsonl                 append-only, hash-chained audit log (source of truth)
    LOG.md                    rendered from log.jsonl
    tools/rsi_log.py          append / verify / render / show
    experiments/<id>/         per-experiment frozen inputs, raw results, verdict, retained patch
profiler-bench/               distribution `dbldatagen-profiler-bench` (uv workspace member) — the measuring stick
  src/profiler_bench/
    catalog/                  seeded datasets + ground-truth manifests, tiers xs / s / m
    sampler/                  reference reconstructor: profile -> synthetic data
    fidelity/                 reconstruction, structure-recovery, and privacy metrics
    runner/                   fresh-process sampling of profile time / CPU / RSS
    evaluate/                 paired bootstrap gate against policy
  policies/*.json             frozen acceptance policies
  COVERAGE.md                 which structures each dataset exercises; known gaps
```

The benchmark is a separate distribution, so a profiler change cannot silently change how it is measured.

Design rules:

- The **profile spec is the contract.** Both engines must emit spec-valid profiles; the benchmark scores
  them the same way. Spec changes are versioned (semver) and logged as proposals.
- The **reference sampler** belongs to the harness and evolves with the spec. Profile fields the sampler does not
  use cannot improve the score, so a proposal that adds a profile feature must also extend the sampler.
- **Frozen per experiment:** the dataset catalog, fidelity metrics, and policy. Changing them is a separate,
  logged benchmark change, never mixed with a profiler change.
- **Two kinds of fidelity:** data-level reconstruction (synthetic vs source) and structure recovery (profile vs the dataset's ground-truth manifest).
- **Tiers:** fidelity is scored at tier `s` (~100k rows). Tier `m` (~2M) measures profiling speed only, so the sampler and metrics need not scale to it.
- **Ceiling check:** fidelity of a real-vs-held-out-real split gives the best achievable score for each dataset.
- No anchoring to current dbldatagen features. Generator gaps go in the "Generator follow-ups" section below.

## RSI loop (summary — full procedure in `rsi/RSI.md`)

1. **Propose** — a hypothesis naming the mechanism, the target datasets/metrics, the expected gain, and the
   behavior that must not change. Logged `proposal`.
2. **Accept for implementation** — polly triages proposals against priorities. Logged `triage` (accepted / deferred / rejected + reason).
3. **Implement** — a sub-agent works in its own worktree and opens a PR against `feat/profiler-rsi` on the fork. Logged `implementation`.
4. **Benchmark** — frozen baseline vs candidate, paired seeds, fresh processes. Logged `benchmark`.
5. **Accept / reject** — the evaluator applies the frozen policy; a different-vendor reviewer checks the diff. Logged `disposition`.
   Rejected patches are retained in `rsi/experiments/<id>/`. The human merges accepted PRs.

Phase 4 features P4.2 and P4.3 enter as RSI proposals and merge only through the accuracy gate (human decision 2026-11-18).

## Work items

Status: `todo` · `in-progress` · `review` · `ready` (PR passed review, awaiting human merge) · `merged` · `blocked`

### Phase 1 — RSI system (Step 1)

| ID | Item | Depends on | Status |
|----|------|------------|--------|
| P1.1 | Package scaffold (`profiler/` uv workspace member) + profile spec v0.1 (model + JSON Schema) + CLI stub | — | merged (PR #6) |
| P1.2 | Baseline PySpark engine: schema, row count, nulls, min/max/mean/std, distinct count | P1.1 | merged (PR #7) |
| P1.3 | Dataset catalog: synthetic datasets with known ground truth (relational PK/FK model, skew, correlated columns, regex-shaped strings, check constraints, nested types, wide/tall), deterministic seeds, scale tiers | P1.1 | merged (PR #8) |
| P1.4 | Reference sampler (profile -> PySpark DataFrames, multi-table, respects keys) | P1.1 | merged (PR #9) |
| P1.5 | Fidelity metrics: marginals (KS / Wasserstein / TVD), cardinality error, pattern match rate, pairwise dependence error, constraint satisfaction, referential integrity, discriminator AUC, privacy guard (exact-row copy rate, distance to closest record) | P1.1 | merged (PR #10) |
| P1.6 | Runner + evaluator: fresh-process samples, wall/CPU/RSS, paired bootstrap gate, policy JSON, results layout, noise control (same build vs itself) | P1.3–P1.5 | merged (PR #12); real Scorer wiring after P1.5 (P1.6a) |
| P1.6a | Wire the runner to the real sampler + fidelity Scorer; end-to-end control run on tier xs | P1.5, P1.6 | merged (PR #14) |
| P1.4b | Fix sampler privacy leak: clipped-normal draws land on exact observed min/max, recreating real-unique pairs (numeric fails partial_copy_rate_excess) | P1.4, P1.6a | merged (PR #15) |
| P1.4c | Fix sampler leak: integer discretization over-weights exact profiled extrema (nested/wide privacy fail) | P1.4b | merged (PR #19) |
| P1.4d | Follow-ups from P1.4c review: tests should use calibrated mu/sigma for endpoint mass and discrete moments in property test | P1.4c | todo |
| rsi-0003a | Follow-ups to rsi-0003 (after its disposition, so the benchmarked commit stays the merged one): tie the Pearson support floor to privacy.min_value_count/k instead of 3; README streaming-equivalence list names the dependency gap | rsi-0003 | todo |
| rsi-0002b | Quantile tails + k-anonymous int histograms | rsi-0002 | ACCEPTED, merged (PR #31) |
| rsi-0002c | Speed follow-up: rsi-0002b slowed nested@s profiling 197s -> 334s (extra histogram value-count job); fold histogram counts into the existing aggregate | rsi-0002b | todo |
| P1.7 | RSI log tooling: append-only `log.jsonl` writer/validator + `LOG.md` renderer; `rsi/RSI.md` | P1.1 | merged (PR #4) |
| P1.7a | Fix: `rsi_log.py append --kind` flag is dropped; add CLI-flag tests | P1.7 | merged (PR #5) |
| P1.8 | Baseline run: freeze the harness, record the baseline + ceiling for every dataset | P1.2–P1.7 | merged (PR #25): frozen baseline; see profiler/rsi/BASELINE.md |
| P1.2a | Fix: baseline engine overflows on `avg`/`stddev` of decimal(38,x) under ANSI mode (found by P1.5 at tier s) | P1.2 | merged (PR #11) |
| P1.3c | Investigate retail real-vs-real ceiling 0.94 (order_items 0.86): suspected catalog Zipf parent ranking varying by sample seed | P1.3, P1.5 | todo |
| P1.3b | Register real public datasets in the catalog (download + checksum hook exists; none registered) — needs a decision on which datasets are reachable through the proxy | P1.3 | todo |
| P1.8a | Fix: profiler fails on nested at tier s (`TypeError: unhashable type: 'list'`) | P1.2, P2.1 | merged (PR #16) |
| P1.8b | Fix: runner scores all fidelity cells in one SparkSession; driver OOM cascades into every later sample. Isolate per (build, dataset) worker, 8g driver | P1.6a | merged (PR #17) |
| P1.8d | Follow-ups from P1.8c review: catalog manifest `_order_key` drops tz offset without normalizing to UTC (manifest.py:133); add 7- and 9-digit fraction parse tests | P1.8c | todo |
| P1.9 | Rust engine scaffold (Polars vs DataFusion decision, CLI binary, spec-valid output for the baseline features) | P1.1, Rust toolchain | deferred (human decision 2026-11-06: PySpark-only Phase 1; Rust engine later) |

### Phase 1b — PII (human request 2026-11-07)

| ID | Item | Depends on | Status |
|----|------|------------|--------|
| P2.1 | PII detection (spec 0.2.0 `semantic` + quasi-identifier sets) and keyed, format-preserving, join-consistent substitution of real data (contract: `.polly/contracts/p2-1-pii.md`) | P1.2 | merged (PR #13) |
| P2.2 | Benchmark side: catalog PII ground truth for every PII class, detection precision/recall metric, substitution fidelity (stats + relational integrity preserved) and leakage metrics, sampler consumes `semantic.pii_class` | P1.5, P2.1 | todo |

| P2.3 | Configurable PII handling policy: QIs substituted by default; per sensitivity / pii_class / column actions (substitute, generalize, suppress, drop, keep) (human decision 2026-11-18; contract `.polly/contracts/p2-3-pii-policy.md`) | P2.1 | merged (PR #21) |

| P2.3a | PII policy follow-ups part 1: FK full-drop with undetected side, DOB shift on original entity keys, NaN/Inf error, small-space capacity pre-check, docs | P2.3 | merged (PR #23) |
| P2.4 | Opt-in `perturb` PII action (noise / affine) to keep Pearson correlations on sensitive numeric and temporal columns (human decision 2026-11-21) | P2.3 | merged (PR #34) |
| P2.5 | PII policy preset `keys_map_rest_perturb`: substitute (mapping) on PK/FK/unique keys, perturb on non-key numeric/temporal columns (sensitive/QI by default; `perturb_scope: all_numeric` optional) | P2.4 | queued (after PR #34) |
| P2.3b | PII follow-ups: hidden entity-key column names must be checked case-insensitively (substitute.py ~L376-379; silent user-column loss); replace the _protect_column argument shim by updating the streaming call site | P2.3a, P3.1 | merged (PR #24) |

### Phase 3 — Streaming and data volume (human request 2026-11-18)

| ID | Item | Depends on | Status |
|----|------|------------|--------|
| P3.1 | Incremental profiling with Spark Structured Streaming: mergeable per-batch states, restart-safe checkpoints, snapshots equal to batch on exact stats (contract `.polly/contracts/p3-1-streaming.md`) | P1.2 | merged (PR #20) |
| P3.1a | Streaming follow-ups: wire pii_policy after P2.3; replace private imports from engines/spark and pii/detect with public helpers; cloud/DBFS checkpoint paths; avoid raw warmup rows on disk (encrypt or non-raw form) | P3.1, P2.3 | todo |
| P3.2 | Data-volume RSI objective: least input fraction reaching a frozen accuracy target per dataset; harness-controlled FK-preserving subsets; new policy `data_volume.json` (contract `.polly/contracts/p3-2-data-volume.md`) | P1.6a, P1.8 | merged (PR #22) |
| P3.3 | Calibrate data-volume targets T_d from the frozen baseline and run the first data-volume baseline | P1.8, P3.2 | todo |
| P1.5b | Benchmark change: fix project-then-sample bias in fidelity verifiers (metrics 1.9.0) | P1.5 | merged (PR #30) |
| P1.8e | Refresh BASELINE.md under the fixed harness (metrics 1.9.0); serial run on a quiet machine | P1.5b | todo |
| P1.6e | Discriminator-first accuracy gate: log-loss gap vs real-vs-real, key/ID feature exclusion (retail ceiling AUC 0.71), cross-table discriminator, policy 3.0.0 (contract `.polly/contracts/p1-6e-discriminator-first.md`) | P1.6d, P3.4a | queued (retail ceiling investigation running) |
| P3.4a | Benchmark change (logged as benchmark_change, never mixed with a profiler proposal): wire real datasets into fidelity/evaluate as a separate HOLDOUT suite: skip structure scoring for ground_truth none; split each export into disjoint PK-hash halves for profiled sample vs privacy holdout, and use them for the real-vs-real ceiling | P3.4 | todo |
| P3.4 | Real public datasets, exported via the Statement Execution API (chunked ARROW_STREAM) into the gitignored cache, never committed. Measured shortlist: samples.tpcds_sf1 (~180 MB: relational, real fan-out skew, decimals, QIs), samples.wanderbricks (~20 MB: map/struct/array, emails/phones, skewed hosts->properties), Marketplace Simulated Retail (CC BY 4.0, nested + CDC; needs install), samples.nyctaxi.trips (heavy tails, dirty values), TPC-H generated locally. Skip bakehouse. Export after the P1.8 baseline finishes | P1.3 | merged (PR #27) |

### Phase 4 — Queued features (human request 2026-11-18; start after Phase 1 baseline, P2.3, P3.1, P3.2)

| ID | Item | Depends on | Status |
|----|------|------------|--------|
| P4.1 | Profile subsets by criteria: random, stratified by key, temporal by date range/window, and LLM-prompted (natural-language criterion translated to a reviewed, deterministic filter/sample spec; LLM is a pluggable interface, none shipped by default). Record the subset spec in profile source metadata | P3.2 (shares subset machinery) | queued |
| P4.2 | Fit regex pattern(s) to fixed-length text columns: per-position character-class inference, multiple patterns with coverage, privacy-safe (dialect + k rules from P1.5). Enters RSI as an accuracy proposal | P1.8 | queued |
| P4.3 | Variable-length text: a small local model fit to PII-masked real text, shipped in the profile, and used by the sampler to generate like text (human decision). Required guards: memorization metric (long-n-gram overlap with real text vs a holdout baseline), canary-insertion test, k-threshold on vocabulary, PII masking before fitting, model size cap; privacy guard fails the proposal if any trip. New text fidelity metrics; spec 0.3 | P2.3, P1.8 | queued |

### Phase 2 — RSI iterations (Step 2)

Candidate proposal areas (they enter the log as proposals when triaged):

- Quantile / histogram marginals; heavy-tail and multimodal fits
- Cardinality: exact vs HLL, bucketed frequency profiles, top-k with a privacy threshold
- String structure: regex inference (character-class tokenization, multiple patterns with coverage weights), length distributions
- Spec 0.3 candidate: lossless (string-encoded) mean/stddev for decimal and int64 columns; float mean collapses location at ~1e28 (found in P1.4b review)
- Spec 0.3 candidate: make the profile regex dialect (allowlisted, portable to Rust; enforced by the bench in P1.5) part of the spec and validate it in the model
- Spec 0.2 candidates: functional dependencies (no field in 0.1.0, so the metrics score them 'not recovered'); correlation_ratio consumption in the sampler
- Keys: PK / unique / composite key discovery; FK inclusion-dependency discovery across tables; FK fan-out distributions
- Check constraints: ranges, enumerations, cross-column inequalities (`a <= b`), functional dependencies
- Dependence: pairwise Spearman / Cramér's V / correlation ratio; Gaussian copula; conditional distributions for categorical pairs
- Temporal: seasonality, monotonic / sequential columns, inter-arrival distributions
- Nested types (struct / array / map) and semi-structured strings (JSON)
- Speed (top): nested at tier m OOMs 10/10 under the frozen 2g/local[2] speed config (dies in collectToPython); tier-m comparisons exclude nested until fixed
- Speed: nested tier-s profiling takes 371 s for 110k rows (one Spark job per nested child + 200-col wide table + PII xxhash sample); top speed candidate
- Speed: batch nested-child aggregates into one Spark job (baseline runs one job per nested child; found in P1.2 review)
- Speed: sampling strategies, single-pass aggregation, approximate sketches, Rust engine parity and performance

### Generator follow-ups (dbldatagen gaps found along the way)

| ID | Gap | Source |
|----|-----|--------|
| G1 | No regex-based string strategy | layout exploration |
| G2 | No quantile / histogram / empirical distribution input | layout exploration |
| G3 | No composite PK / FK support in core planner | `dbldatagen/core/planner.py` |
| G4 | No correlation / copula model across columns | layout exploration |
| G5 | No check-constraint support in core spec | layout exploration |
| G6 | No bucketed / approximate cardinality strategy | layout exploration |
| G7 | No map type | `docs/core/limitations.md` |
| G8 | Profile -> `DataGenPlan` translator | — |
