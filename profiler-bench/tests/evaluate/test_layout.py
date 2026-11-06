"""Results layout and benchmark_entry.json schema validation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from profiler_bench.evaluate.evaluate import evaluate_experiment
from profiler_bench.evaluate.report import build_benchmark_entry


REPO = Path(__file__).resolve().parents[3]
RSI_LOG = REPO / "profiler" / "rsi" / "tools" / "rsi_log.py"


def test_results_layout_and_benchmark_entry(tmp_path: Path) -> None:
    experiment = tmp_path / "exp-layout"
    samples = experiment / "samples"
    samples.mkdir(parents=True)
    run = {
        "experiment_id": "exp-layout",
        "control": True,
        "baseline": {"commit": "aaa", "worktree": "/tmp/a"},
        "candidate": {"commit": "aaa", "worktree": "/tmp/a"},
        "harness": {"commit": "bbb", "harness_hash": "h" * 64, "policy_hash": "p" * 64},
        "policy": _policy(),
        "not_measured": ["Spark JVM RSS"],
    }
    (experiment / "run.json").write_text(json.dumps(run))
    (experiment / "proposal.json").write_text(
        json.dumps({"type": "accuracy", "proposal_id": "prop-1", "targets": ["numeric"]})
    )
    for index in range(2):
        for role in ("baseline", "candidate"):
            (samples / f"speed-{role}-numeric-s-{index}.json").write_text(
                json.dumps(
                    {
                        "kind": "speed",
                        "role": role,
                        "dataset": "numeric",
                        "tier": "s",
                        "sample_index": index,
                        "ok": True,
                        "wall_s": 1.0,
                        "cpu_s": 0.8,
                        "peak_rss_bytes": 1000,
                        "profile_hash": "abc",
                    }
                )
            )
    for seed in range(2):
        for role in ("baseline", "candidate"):
            (samples / f"fidelity-{role}-numeric-s-seed-{seed}.json").write_text(
                json.dumps(
                    {
                        "kind": "fidelity",
                        "role": role,
                        "dataset": "numeric",
                        "tier": "s",
                        "seed": seed,
                        "ok": True,
                        "reconstruction": 0.8,
                        "structure": 0.7,
                        "privacy_passed": True,
                        "ceiling": 0.9,
                        "relative_reconstruction": 0.8 / 0.9,
                    }
                )
            )

    verdict = evaluate_experiment(experiment)
    assert (experiment / "summary.json").is_file()
    assert (experiment / "verdict.json").is_file()
    assert (experiment / "REPORT.md").is_file()
    assert (experiment / "benchmark_entry.json").is_file()
    assert verdict["verdict"] != "accepted"

    entry = json.loads((experiment / "benchmark_entry.json").read_text())
    required = {
        "kind",
        "actor",
        "work_item",
        "summary",
        "proposal_id",
        "experiment_id",
        "baseline_commit",
        "candidate_commit",
        "harness_hash",
        "policy",
        "results_path",
        "summary_metrics",
    }
    assert required <= entry.keys()
    assert entry["kind"] == "benchmark"
    assert isinstance(entry["summary_metrics"], dict)

    _assert_appendable(tmp_path, entry)


def test_build_benchmark_entry_fields() -> None:
    run = {
        "experiment_id": "e1",
        "baseline": {"commit": "c1"},
        "candidate": {"commit": "c2"},
        "harness": {"harness_hash": "hh", "policy_hash": "pp"},
    }
    summary: dict[str, Any] = {"failed_speed": 0, "failed_fidelity": 0, "fidelity": {}, "speed": {}}
    verdict = {"verdict": "inconclusive", "reasons": ["accuracy_crossing_interval"]}
    entry = build_benchmark_entry(run, summary, verdict, Path("/tmp/e1"))
    assert entry["baseline_commit"] == "c1"
    assert entry["candidate_commit"] == "c2"


def _load_rsi_log() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rsi_log_under_test", RSI_LOG)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_appendable(tmp_path: Path, entry: dict[str, Any]) -> None:
    rsi_log = _load_rsi_log()
    log_path = tmp_path / "log.jsonl"
    rsi_log.append_entry(
        log_path,
        {
            "kind": "decision",
            "actor": "polly",
            "work_item": "P1.6",
            "summary": "test genesis",
        },
    )
    rsi_log.append_entry(
        log_path,
        {
            "kind": "proposal",
            "actor": "polly",
            "work_item": "P1.6",
            "summary": "test proposal",
            "proposal_id": entry["proposal_id"],
            "hypothesis": "h",
            "mechanism": "m",
            "targets": ["numeric"],
            "expected_effect": "e",
            "invariants": [],
        },
    )
    rsi_log.append_entry(
        log_path,
        {
            "kind": "triage",
            "actor": "polly",
            "work_item": "P1.6",
            "summary": "accept",
            "proposal_id": entry["proposal_id"],
            "verdict": "accepted",
            "reason": "test",
        },
    )
    rsi_log.append_entry(
        log_path,
        {
            "kind": "implementation",
            "actor": "worker:test",
            "work_item": "P1.6",
            "summary": "impl",
            "proposal_id": entry["proposal_id"],
            "branch": "polly/p1-6-runner",
            "pr_url": "https://example.com/pr",
            "commit": "deadbeef",
            "worktree": "/tmp/wt",
        },
    )
    written = rsi_log.append_entry(log_path, entry)
    assert written["kind"] == "benchmark"
    rsi_log.verify(log_path)


def _policy() -> dict[str, Any]:
    return {
        "bootstrap": {"samples": 200, "seed": 0, "confidence": 0.95},
        "accuracy": {
            "min_mean_improvement": 0.005,
            "ci_lower_must_exceed": 0.0,
            "max_dataset_regression": 0.01,
            "max_structure_regression": 0.02,
        },
        "speed": {
            "min_target_improvement": 0.05,
            "max_wall_regression": 0.10,
            "max_cpu_regression": 0.10,
            "max_rss_regression_frac": 0.05,
            "max_rss_regression_abs_mib": 64.0,
        },
        "privacy": {"require_all_pass": True},
    }
