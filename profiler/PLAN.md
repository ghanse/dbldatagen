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

## Work items

Status: `todo` · `in-progress` · `review` · `ready` (PR passed review, awaiting human merge) · `merged` · `blocked`

### Phase 1 — RSI system (Step 1)

| ID | Item | Depends on | Status |
|----|------|------------|--------|
| P1.1 | Package scaffold (`profiler/` uv workspace member) + profile spec v0.1 (model + JSON Schema) + CLI stub | — | merged (PR #6) |
| P1.2 | Baseline PySpark engine: schema, row count, nulls, min/max/mean/std, distinct count | P1.1 | merged (PR #7) |
| P1.3 | Dataset catalog: synthetic datasets with known ground truth (relational PK/FK model, skew, correlated columns, regex-shaped strings, check constraints, nested types, wide/tall), deterministic seeds, scale tiers | P1.1 | merged (PR #8) |
| P1.4 | Reference sampler (profile -> PySpark DataFrames, multi-table, respects keys) | P1.1 | approved (PR #9), waiting on #8 |
| P1.5 | Fidelity metrics: marginals (KS / Wasserstein / TVD), cardinality error, pattern match rate, pairwise dependence error, constraint satisfaction, referential integrity, discriminator AUC, privacy guard (exact-row copy rate, distance to closest record) | P1.1 | review (PR #10, stacked on #8) |
| P1.6 | Runner + evaluator: fresh-process samples, wall/CPU/RSS, paired bootstrap gate, policy JSON, results layout, noise control (same build vs itself) | P1.3–P1.5 | todo |
| P1.7 | RSI log tooling: append-only `log.jsonl` writer/validator + `LOG.md` renderer; `rsi/RSI.md` | P1.1 | merged (PR #4) |
| P1.7a | Fix: `rsi_log.py append --kind` flag is dropped; add CLI-flag tests | P1.7 | merged (PR #5) |
| P1.8 | Baseline run: freeze the harness, record the baseline + ceiling for every dataset | P1.2–P1.7 | todo |
| P1.2a | Fix: baseline engine overflows on `avg`/`stddev` of decimal(38,x) under ANSI mode (found by P1.5 at tier s) | P1.2 | merged (PR #11) |
| P1.3b | Register real public datasets in the catalog (download + checksum hook exists; none registered) — needs a decision on which datasets are reachable through the proxy | P1.3 | todo |
| P1.9 | Rust engine scaffold (Polars vs DataFusion decision, CLI binary, spec-valid output for the baseline features) | P1.1, Rust toolchain | deferred (human decision 2026-11-06: PySpark-only Phase 1; Rust engine later) |

### Phase 2 — RSI iterations (Step 2)

Candidate proposal areas (they enter the log as proposals when triaged):

- Quantile / histogram marginals; heavy-tail and multimodal fits
- Cardinality: exact vs HLL, bucketed frequency profiles, top-k with a privacy threshold
- String structure: regex inference (character-class tokenization, multiple patterns with coverage weights), length distributions
- Spec 0.2 candidates: functional dependencies (no field in 0.1.0, so the metrics score them 'not recovered'); correlation_ratio consumption in the sampler
- Keys: PK / unique / composite key discovery; FK inclusion-dependency discovery across tables; FK fan-out distributions
- Check constraints: ranges, enumerations, cross-column inequalities (`a <= b`), functional dependencies
- Dependence: pairwise Spearman / Cramér's V / correlation ratio; Gaussian copula; conditional distributions for categorical pairs
- Temporal: seasonality, monotonic / sequential columns, inter-arrival distributions
- Nested types (struct / array / map) and semi-structured strings (JSON)
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
