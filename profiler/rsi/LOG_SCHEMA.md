# RSI log schema

`log.jsonl` is the append-only source of truth. Each line is one JSON object. `LOG.md` is a generated view.

Every entry has these fields:

| Field | Meaning |
|---|---|
| `seq` | Positive integer sequence number, starting at 1 without gaps. |
| `ts` | UTC ISO 8601 timestamp. |
| `prev_hash` | Lowercase SHA-256 of the exact preceding line, excluding its newline. |
| `kind` | One of the entry kinds below. |
| `actor` | Author, such as `polly` or `worker:<vendor>/<session>`. |
| `work_item` | Plan work item ID. |
| `summary` | Concise description of the event. |

The existing sequence 1 decision is the genesis entry. It may omit `prev_hash`; its value is treated as null. Extra metadata
fields are allowed. Generated fields (`seq`, `ts`, and `prev_hash`) cannot be supplied to `append`.

## Entry kinds

| Kind | Required fields |
|---|---|
| `decision` | No additional fields. Records a project decision. |
| `proposal` | `proposal_id`, `hypothesis`, `mechanism`, `targets`, `expected_effect`, `invariants` |
| `triage` | `proposal_id`, `verdict` (`accepted`, `deferred`, or `rejected`), `reason` |
| `implementation` | `proposal_id`, `branch`, `pr_url`, `commit`, `worktree` |
| `benchmark` | `proposal_id`, `experiment_id`, `baseline_commit`, `candidate_commit`, `harness_hash`, `policy`, `results_path`, `summary_metrics` |
| `disposition` | `proposal_id`, `verdict` (`accepted`, `rejected`, or `inconclusive`), `reason`, `evidence_path`, `retained_patch` |
| `review` | `proposal_id`, `reviewer`, `verdict`, `blocking_issues` |
| `benchmark_change` | `justification` |

`targets` is an array or object naming datasets and metrics. `invariants` and `blocking_issues` are arrays.
`summary_metrics` is an object.

## Referential rules

- A proposal ID is introduced once by `proposal`.
- `triage` requires a prior proposal.
- `implementation` requires a prior accepted triage.
- `benchmark` requires a prior implementation.
- `review` requires a prior implementation.
- `disposition` requires a prior benchmark.

## Commands

Run `python profiler/rsi/tools/rsi_log.py <command>`. `append` accepts fields as flags or with
`--json-file PATH`; use `--json-file -` for stdin. JSON-valued flags contain JSON text. `verify` checks the full log.
`render` rewrites `LOG.md` deterministically. `show PROPOSAL_ID` prints one proposal thread.
