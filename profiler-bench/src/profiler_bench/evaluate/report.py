"""Writes summary.json, verdict.json, REPORT.md, and benchmark_entry.json."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def write_outputs(
    experiment_dir: Path,
    *,
    run: dict[str, Any],
    speed_records: list[dict[str, Any]],
    fidelity_records: list[dict[str, Any]],
    verdict: dict[str, Any],
    determinism: dict[str, Any] | None = None,
) -> None:
    """Writes the evaluator artifacts under `experiment_dir`."""
    summary = build_summary(run, speed_records, fidelity_records, determinism)
    (experiment_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (experiment_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    entry = build_benchmark_entry(run, summary, verdict, experiment_dir)
    (experiment_dir / "benchmark_entry.json").write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
    (experiment_dir / "REPORT.md").write_text(render_report(run, summary, verdict, determinism))


def build_summary(
    run: dict[str, Any],
    speed_records: list[dict[str, Any]],
    fidelity_records: list[dict[str, Any]],
    determinism: dict[str, Any] | None,
) -> dict[str, Any]:
    """Aggregates raw samples into summary metrics."""
    speed: dict[str, Any] = {}
    by_case: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in speed_records:
        if record.get("ok"):
            key = (str(record["role"]), str(record["dataset"]), str(record["tier"]))
            by_case[key].append(record)
    for (role, dataset, tier), rows in sorted(by_case.items()):
        speed[f"{role}/{dataset}@{tier}"] = {
            "n": len(rows),
            "wall_s_median": median(float(row["wall_s"]) for row in rows),
            "cpu_s_median": median(float(row["cpu_s"]) for row in rows),
            "peak_rss_bytes_median": median(float(row["peak_rss_bytes"]) for row in rows),
            "profile_hashes": sorted({str(row["profile_hash"]) for row in rows if row.get("profile_hash")}),
        }

    fidelity: dict[str, Any] = {}
    by_fid: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in fidelity_records:
        if record.get("ok"):
            by_fid[(str(record["role"]), str(record["dataset"]))].append(record)
    for (role, dataset), rows in sorted(by_fid.items()):
        fidelity[f"{role}/{dataset}"] = {
            "n": len(rows),
            "reconstruction_mean": sum(float(row["reconstruction"]) for row in rows) / len(rows),
            "relative_mean": sum(float(row["relative_reconstruction"]) for row in rows) / len(rows),
            "structure_mean": (
                "not_applicable"
                if any(row["structure"] is None for row in rows)
                else sum(float(row["structure"]) for row in rows) / len(rows)
            ),
            "privacy_all_passed": all(bool(row["privacy_passed"]) for row in rows),
        }

    summary: dict[str, Any] = {
        "experiment_id": run.get("experiment_id"),
        "control": run.get("control"),
        "speed": speed,
        "fidelity": fidelity,
        "failed_speed": sum(1 for row in speed_records if not row.get("ok")),
        "failed_fidelity": sum(1 for row in fidelity_records if not row.get("ok")),
        "profile_determinism": determinism or {},
        "not_measured": run.get("not_measured", []),
    }
    if str(run.get("mode") or "") == "data-volume" or any(
        record.get("subset") is not None for record in fidelity_records
    ):
        summary["input_fractions"] = _input_fraction_summary(fidelity_records)
    return summary


def _input_fraction_summary(fidelity_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean realized overall input fraction next to each nominal grid ``f``."""
    by_key: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for record in fidelity_records:
        if not record.get("ok"):
            continue
        subset = record.get("subset") or {}
        raw = subset.get("overall_actual_fraction")
        if raw is None:
            continue
        key = (str(record["role"]), str(record["dataset"]), float(record.get("fraction", 1.0)))
        by_key[key].append(float(raw))
    out: dict[str, Any] = {}
    for (role, dataset, fraction), values in sorted(by_key.items()):
        out[f"{role}/{dataset}@f={fraction:g}"] = {
            "fraction": fraction,
            "overall_actual_fraction_mean": sum(values) / len(values),
            "n": len(values),
        }
    return out


def build_benchmark_entry(
    run: dict[str, Any],
    summary: dict[str, Any],
    verdict: dict[str, Any],
    experiment_dir: Path,
) -> dict[str, Any]:
    """Returns a payload shaped for `rsi_log.py append` with kind `benchmark`."""
    baseline = run.get("baseline", {})
    candidate = run.get("candidate", {})
    harness = run.get("harness", {})
    proposal = {}
    proposal_path = experiment_dir / "proposal.json"
    if proposal_path.is_file():
        proposal = json.loads(proposal_path.read_text())
    relative = experiment_dir.as_posix()
    for depth in (3, 2, 1):
        if len(experiment_dir.parents) <= depth:
            continue
        try:
            relative = str(experiment_dir.relative_to(experiment_dir.parents[depth]))
            break
        except ValueError:
            continue
    return {
        "kind": "benchmark",
        "actor": "worker:omnigent/p1-6-runner",
        "work_item": "P1.6",
        "summary": (
            f"Benchmark {run.get('experiment_id')} verdict={verdict.get('verdict')} "
            f"baseline={baseline.get('commit', '')[:12]} candidate={candidate.get('commit', '')[:12]}"
        ),
        "proposal_id": str(proposal.get("proposal_id", run.get("experiment_id"))),
        "experiment_id": str(run.get("experiment_id")),
        "baseline_commit": str(baseline.get("commit", "unknown")),
        "candidate_commit": str(candidate.get("commit", "unknown")),
        "harness_hash": str(harness.get("harness_hash", "")),
        "policy": str(harness.get("policy_hash", "")),
        "results_path": relative,
        "summary_metrics": {
            "verdict": verdict.get("verdict"),
            "reasons": verdict.get("reasons", []),
            "failed_speed": summary.get("failed_speed"),
            "failed_fidelity": summary.get("failed_fidelity"),
            "fidelity": summary.get("fidelity"),
            "speed": {
                key: {
                    "wall_s_median": value["wall_s_median"],
                    "cpu_s_median": value["cpu_s_median"],
                    "peak_rss_bytes_median": value["peak_rss_bytes_median"],
                }
                for key, value in summary.get("speed", {}).items()
            },
            "n_star": _nstar_both(verdict.get("n_star")),
            "input_fractions": summary.get("input_fractions"),
        },
    }


def _nstar_both(n_star: object) -> dict[str, Any] | None:
    """Returns n* as both grid ``f`` and realized overall input fraction."""
    if not isinstance(n_star, dict):
        return None
    out: dict[str, Any] = {}
    for role, datasets in n_star.items():
        if not isinstance(datasets, dict):
            continue
        out[str(role)] = {}
        for dataset, payload in datasets.items():
            if not isinstance(payload, dict):
                continue
            out[str(role)][str(dataset)] = {
                "n_star": payload.get("n_star"),
                "n_star_overall_actual_fraction": payload.get("n_star_overall_actual_fraction"),
                "curve": [
                    {
                        "fraction": point.get("fraction"),
                        "overall_actual_fraction": point.get("overall_actual_fraction"),
                        "status": point.get("status"),
                        "mean": point.get("mean"),
                    }
                    for point in (payload.get("curve") or [])
                    if isinstance(point, dict)
                ],
            }
    return out or None


def render_report(
    run: dict[str, Any],
    summary: dict[str, Any],
    verdict: dict[str, Any],
    determinism: dict[str, Any] | None,
) -> str:
    """Returns a concise Markdown report for humans."""
    lines = [
        f"# Experiment {run.get('experiment_id')}",
        "",
        f"Verdict: **{verdict.get('verdict')}**",
        "",
        "## Builds",
        "",
        f"- Baseline: `{run.get('baseline', {}).get('commit')}` (`{run.get('baseline', {}).get('worktree')}`)",
        f"- Candidate: `{run.get('candidate', {}).get('commit')}` (`{run.get('candidate', {}).get('worktree')}`)",
        f"- Harness: `{run.get('harness', {}).get('commit')}` hash `{run.get('harness', {}).get('harness_hash', '')[:12]}`",
        f"- Control run: `{run.get('control')}`",
        "",
        "## Measured",
        "",
        "- Driver wall time, process CPU time, and peak RSS for `profile_tables`",
        "- Profile JSON hash (created_at stripped) and cross-sample determinism",
        "- Paired reconstruction / structure / privacy scores at the fidelity tier",
        "",
        "## Not measured",
        "",
    ]
    for item in run.get("not_measured", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Gates", ""])
    for gate in verdict.get("gates", []):
        lines.append(f"- `{gate['name']}`: {gate['status']} ({gate['reason']})")
    n_star = verdict.get("n_star")
    if isinstance(n_star, dict) and n_star:
        lines.extend(["", "## n* (grid f and realized overall input fraction)", ""])
        for role, datasets in sorted(n_star.items()):
            if not isinstance(datasets, dict):
                continue
            for dataset, payload in sorted(datasets.items()):
                if not isinstance(payload, dict):
                    continue
                lines.append(
                    f"- `{role}/{dataset}`: n* grid `f={payload.get('n_star')}` "
                    f"(overall_actual={payload.get('n_star_overall_actual_fraction')})"
                )
                for point in payload.get("curve") or []:
                    if not isinstance(point, dict):
                        continue
                    lines.append(
                        f"  - f={point.get('fraction')} overall_actual={point.get('overall_actual_fraction')} "
                        f"status={point.get('status')} mean={point.get('mean')}"
                    )
    lines.extend(["", "## Summary", "", "```json", json.dumps(summary, indent=2, sort_keys=True), "```", ""])
    if determinism is not None:
        lines.extend(
            [
                "## Profile determinism",
                "",
                f"Deterministic: `{determinism.get('deterministic')}`",
                "",
            ]
        )
    return "\n".join(lines)
