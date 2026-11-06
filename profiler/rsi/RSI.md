# Recursive self-improvement procedure

The profiler optimizes reconstruction accuracy first and profiling speed second. No candidate may weaken the privacy guard:
profiles must not expose raw rows or rare identifying values.

## Roles

- **Polly** owns priorities, freezes experiments, triages proposals, and records dispositions against the policy.
- **Workers** propose and implement one focused change in an isolated worktree and open a reviewable PR.
- **Benchmark workers** run the frozen baseline and candidate in controlled, fresh processes and retain raw results.
- **Reviewers** from a different vendor inspect correctness, privacy, scope, and benchmark evidence.
- **The human maintainer** merges accepted changes and controls releases.

## Loop

1. Log a proposal with its mechanism, target datasets and metrics, expected effect, and invariants.
2. Polly logs an accepted, deferred, or rejected triage. Only accepted proposals are implemented.
3. A worker makes one focused patch and logs its branch, PR, commit, and worktree.
4. Run the frozen baseline and candidate with paired seeds and fresh processes. Log revisions, artifact hashes, policy, raw
   results, and summary metrics. Preserve failures and uncertainty.
5. A different-vendor reviewer checks the patch and evidence. Polly applies the frozen policy and logs an accepted,
   rejected, or inconclusive disposition. The human merges accepted PRs.

## Frozen experiment contract

Freeze the dataset catalog and scale tiers, reference sampler, fidelity and privacy metrics, target and guard datasets,
seeds, runner, environment, and acceptance policy before implementation. Record the baseline and candidate commits and a
hash of the harness. A catalog, metric, sampler, runner, or policy change is a separate `benchmark_change`; validate and
freeze it before using it to judge a profiler patch.

Accuracy gates run before speed gates. Compare reconstruction metrics with the real-versus-held-out-real ceiling where
available. Speed measurements cover wall time, CPU, and peak memory. The privacy guard blocks acceptance regardless of
accuracy or speed. A favorable point estimate without sufficient policy evidence is inconclusive.

## Evidence and retention

Each disposition links to reproducible commands, exact source revisions, environment and dependency versions, frozen
inputs and hashes, raw samples, failures, summary metrics, uncertainty, review, and the applied policy. Evidence must state
what was measured and what was not.

Keep every experiment under `profiler/rsi/experiments/<experiment_id>/`. For a rejected or inconclusive candidate, retain
the patch, frozen inputs, raw results, evaluation, and reason there before reverting the working branch. Do not present
effects from rejected patches as accepted gains. Append events with `tools/rsi_log.py`; never rewrite prior log lines.
