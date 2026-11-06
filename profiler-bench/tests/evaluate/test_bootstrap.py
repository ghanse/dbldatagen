"""Bootstrap CI behaviour on synthetic paired differences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from profiler_bench.evaluate.bootstrap import paired_bootstrap_ci, stratified_dataset_mean_ci
from profiler_bench.evaluate.evaluate import evaluate_experiment
from profiler_bench.evaluate.gates import evaluate_gates


def test_known_positive_effect_accepted() -> None:
    diffs = [0.02] * 8
    ci = paired_bootstrap_ci(diffs, statistic="mean", n_bootstrap=500, seed=1)
    assert ci.point == pytest.approx(0.02)
    assert ci.low > 0
    assert not ci.crosses_zero()


def test_null_effect_crosses_zero() -> None:
    diffs = [0.01, -0.01, 0.005, -0.005, 0.0, 0.002, -0.002, 0.001]
    ci = paired_bootstrap_ci(diffs, statistic="mean", n_bootstrap=2000, seed=0)
    assert ci.crosses_zero() or abs(ci.point) < 0.01


def test_known_regression_ci_below_zero() -> None:
    diffs = [-0.03] * 10
    ci = paired_bootstrap_ci(diffs, statistic="mean", n_bootstrap=500, seed=2)
    assert ci.high < 0


def test_bootstrap_is_deterministic() -> None:
    diffs = [0.01, 0.02, -0.005, 0.015, 0.008]
    first = paired_bootstrap_ci(diffs, n_bootstrap=300, seed=99)
    second = paired_bootstrap_ci(diffs, n_bootstrap=300, seed=99)
    assert first == second


def test_stratified_mean_equals_equal_dataset_weights() -> None:
    # Dataset A has more seeds; stratified mean still weights A and B equally.
    paired = {"a": [0.10, 0.10, 0.10, 0.10], "b": [0.00]}
    ci = stratified_dataset_mean_ci(paired, n_bootstrap=200, seed=0)
    assert ci.point == pytest.approx(0.05)
    assert ci.method == "stratified_dataset_mean"


def test_gates_known_effect_accepted() -> None:
    fidelity = _fidelity_pairs(baseline=0.70, candidate=0.78, ceiling=0.90, seeds=6)
    speed = _speed_pairs(baseline=1.0, candidate=1.0, n=4)
    verdict = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=_policy(n_speed=4, n_fid=6),
        proposal={"type": "accuracy", "targets": ["numeric"]},
        run=_run(control=False, n_speed=4, n_fid=6, distinct=True),
    )
    assert verdict["verdict"] == "accepted"
    mean_gate = next(gate for gate in verdict["gates"] if gate["name"] == "accuracy_mean")
    assert mean_gate["detail"]["ci"]["method"] == "stratified_dataset_mean"


def test_gates_null_inconclusive() -> None:
    records = []
    for seed, delta in enumerate([0.002, -0.002, 0.001, -0.001, 0.003, -0.003]):
        for role, score in (("baseline", 0.80), ("candidate", 0.80 + delta)):
            records.append(
                {
                    "kind": "fidelity",
                    "role": role,
                    "dataset": "numeric",
                    "tier": "s",
                    "seed": seed,
                    "ok": True,
                    "reconstruction": score,
                    "structure": 0.7,
                    "privacy_passed": True,
                    "ceiling": 0.9,
                    "relative_reconstruction": score / 0.9,
                }
            )
    speed = _speed_pairs(baseline=1.0, candidate=1.0, n=4)
    verdict = evaluate_gates(
        speed_records=speed,
        fidelity_records=records,
        policy=_policy(n_speed=4, n_fid=6),
        proposal={"type": "accuracy", "targets": ["numeric"]},
        run=_run(control=False, n_speed=4, n_fid=6, distinct=True),
    )
    assert verdict["verdict"] in {"inconclusive", "rejected"}
    assert verdict["verdict"] != "accepted"


def test_gates_known_regression_rejected() -> None:
    fidelity = _fidelity_pairs(baseline=0.80, candidate=0.70, ceiling=0.90, seeds=6)
    speed = _speed_pairs(baseline=1.0, candidate=1.0, n=4)
    verdict = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=_policy(n_speed=4, n_fid=6),
        proposal={"type": "accuracy", "targets": ["numeric"]},
        run=_run(control=False, n_speed=4, n_fid=6, distinct=True),
    )
    assert verdict["verdict"] == "rejected"


def test_privacy_failure_rejects_despite_gains() -> None:
    fidelity = _fidelity_pairs(baseline=0.70, candidate=0.85, ceiling=0.90, seeds=6)
    for record in fidelity:
        if record["role"] == "candidate":
            record["privacy_passed"] = False
    speed = _speed_pairs(baseline=1.0, candidate=0.5, n=4)
    verdict = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=_policy(n_speed=4, n_fid=6),
        proposal={"type": "accuracy", "targets": ["numeric"]},
        run=_run(control=False, n_speed=4, n_fid=6, distinct=True),
    )
    assert verdict["verdict"] == "rejected"
    assert "privacy_guard_failed" in verdict["reasons"]


def test_privacy_null_or_missing_rejects() -> None:
    fidelity = _fidelity_pairs(baseline=0.70, candidate=0.85, ceiling=0.90, seeds=4)
    for record in fidelity:
        if record["role"] == "candidate" and record["seed"] == 0:
            record["privacy_passed"] = None
        if record["role"] == "candidate" and record["seed"] == 1:
            del record["privacy_passed"]
    speed = _speed_pairs(baseline=1.0, candidate=0.5, n=2)
    # Policy cannot disable the hard guard.
    policy = _policy(n_speed=2, n_fid=4)
    policy["privacy"] = {"require_all_pass": False}
    verdict = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=policy,
        proposal={"type": "accuracy", "targets": ["numeric"]},
        run=_run(control=False, n_speed=2, n_fid=4, distinct=True, policy=policy),
    )
    assert verdict["verdict"] == "rejected"
    assert "privacy_guard_failed" in verdict["reasons"]


def test_missing_samples_block_acceptance() -> None:
    fidelity = _fidelity_pairs(baseline=0.70, candidate=0.85, ceiling=0.90, seeds=4)
    fidelity[0]["ok"] = False
    speed = _speed_pairs(baseline=1.0, candidate=0.5, n=4)
    verdict = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=_policy(n_speed=4, n_fid=4),
        proposal={"type": "accuracy", "targets": ["numeric"]},
        run=_run(control=False, n_speed=4, n_fid=4, distinct=True),
    )
    assert verdict["verdict"] != "accepted"
    assert "failed_or_missing_samples" in verdict["reasons"]


def test_deleted_sample_file_blocks_acceptance_on_reevaluate(tmp_path: Path) -> None:
    experiment = _write_complete_experiment(tmp_path / "exp-missing", n_speed=2, n_fid=2, distinct=True)
    sample = next((experiment / "samples").glob("speed-candidate-*.json"))
    sample.unlink()
    verdict = evaluate_experiment(experiment)
    assert verdict["verdict"] != "accepted"
    assert "failed_or_missing_samples" in verdict["reasons"]
    completeness = next(gate for gate in verdict["gates"] if gate["name"] == "sample_completeness")
    assert completeness["detail"]["missing"]


def test_control_favorable_noise_never_accepted() -> None:
    fidelity = _fidelity_pairs(baseline=0.50, candidate=0.90, ceiling=0.95, seeds=6)
    speed = _speed_pairs(baseline=2.0, candidate=0.5, n=4)
    verdict = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=_policy(n_speed=4, n_fid=6),
        proposal={"type": "accuracy", "targets": ["numeric"]},
        run=_run(control=True, n_speed=4, n_fid=6, distinct=False),
    )
    assert verdict["verdict"] == "inconclusive"
    assert "control" in verdict["reasons"]


def test_identical_profiler_hash_never_accepted() -> None:
    fidelity = _fidelity_pairs(baseline=0.50, candidate=0.90, ceiling=0.95, seeds=6)
    speed = _speed_pairs(baseline=2.0, candidate=0.5, n=4)
    verdict = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=_policy(n_speed=4, n_fid=6),
        proposal={"type": "accuracy", "targets": ["numeric"]},
        run=_run(control=False, n_speed=4, n_fid=6, distinct=False),
    )
    assert verdict["verdict"] == "inconclusive"
    assert "control" in verdict["reasons"]


def test_verdict_deterministic_for_same_inputs() -> None:
    fidelity = _fidelity_pairs(baseline=0.70, candidate=0.78, ceiling=0.90, seeds=5)
    speed = _speed_pairs(baseline=1.0, candidate=0.9, n=4)
    run = _run(control=False, n_speed=4, n_fid=5, distinct=True)
    first = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=_policy(n_speed=4, n_fid=5),
        proposal={"type": "accuracy", "targets": ["numeric"]},
        run=run,
    )
    second = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=_policy(n_speed=4, n_fid=5),
        proposal={"type": "accuracy", "targets": ["numeric"]},
        run=run,
    )
    assert first == second


def _policy(*, n_speed: int, n_fid: int) -> dict[str, Any]:
    return {
        "bootstrap": {"samples": 500, "seed": 0, "confidence": 0.95},
        "speed_samples": {"s": n_speed, "xs": n_speed, "m": n_speed},
        "fidelity_seeds": n_fid,
        "fidelity_tier": "s",
        "speed_tiers": ["s"],
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
        "privacy": {"hard_guard": True},
    }


def _run(
    *,
    control: bool,
    n_speed: int,
    n_fid: int,
    distinct: bool,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = policy or _policy(n_speed=n_speed, n_fid=n_fid)
    base_hash = "baseline-src"
    cand_hash = "candidate-src" if distinct else base_hash
    return {
        "control": control,
        "datasets": ["numeric"],
        "speed_tiers": ["s"],
        "fidelity_tier": "s",
        "baseline": {"profiler_src_hash": base_hash, "commit": "aaa"},
        "candidate": {"profiler_src_hash": cand_hash, "commit": "bbb" if distinct else "aaa"},
        "policy": policy,
    }


def _fidelity_pairs(
    *,
    baseline: float,
    candidate: float,
    ceiling: float,
    seeds: int,
    noise: float = 0.0,
) -> list[dict[str, Any]]:
    records = []
    for seed in range(seeds):
        for role, value in (("baseline", baseline), ("candidate", candidate)):
            score = value + (noise if seed % 2 == 0 and role == "candidate" else 0.0)
            relative = min(1.0, score / ceiling)
            records.append(
                {
                    "kind": "fidelity",
                    "role": role,
                    "dataset": "numeric",
                    "tier": "s",
                    "seed": seed,
                    "ok": True,
                    "reconstruction": score,
                    "structure": 0.7,
                    "privacy_passed": True,
                    "ceiling": ceiling,
                    "relative_reconstruction": relative,
                }
            )
    return records


def _speed_pairs(*, baseline: float, candidate: float, n: int) -> list[dict[str, Any]]:
    records = []
    for index in range(n):
        for role, wall in (("baseline", baseline), ("candidate", candidate)):
            records.append(
                {
                    "kind": "speed",
                    "role": role,
                    "dataset": "numeric",
                    "tier": "s",
                    "sample_index": index,
                    "ok": True,
                    "wall_s": wall,
                    "cpu_s": wall * 0.8,
                    "peak_rss_bytes": 200_000_000,
                }
            )
    return records


def _write_complete_experiment(root: Path, *, n_speed: int, n_fid: int, distinct: bool) -> Path:
    samples = root / "samples"
    samples.mkdir(parents=True)
    run = _run(control=False, n_speed=n_speed, n_fid=n_fid, distinct=distinct)
    run["experiment_id"] = root.name
    run["harness"] = {"commit": "h", "harness_hash": "x" * 64, "policy_hash": "y" * 64}
    run["not_measured"] = []
    (root / "run.json").write_text(json.dumps(run))
    (root / "proposal.json").write_text(
        json.dumps({"type": "accuracy", "proposal_id": root.name, "targets": ["numeric"]})
    )
    for record in _speed_pairs(baseline=1.0, candidate=0.5, n=n_speed):
        path = samples / f"speed-{record['role']}-{record['dataset']}-{record['tier']}-{record['sample_index']}.json"
        path.write_text(json.dumps(record))
    for record in _fidelity_pairs(baseline=0.7, candidate=0.85, ceiling=0.9, seeds=n_fid):
        path = samples / (f"fidelity-{record['role']}-{record['dataset']}-{record['tier']}-seed-{record['seed']}.json")
        path.write_text(json.dumps(record))
    return root
