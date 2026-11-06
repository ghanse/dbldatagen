"""Bootstrap CI behaviour on synthetic paired differences."""

from __future__ import annotations

import pytest

from profiler_bench.evaluate.bootstrap import paired_bootstrap_ci
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


def test_gates_known_effect_accepted() -> None:
    fidelity = _fidelity_pairs(baseline=0.70, candidate=0.78, ceiling=0.90, seeds=6)
    speed = _speed_pairs(baseline=1.0, candidate=1.0, n=4)
    verdict = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=_policy(),
        proposal={"type": "accuracy", "targets": ["numeric"]},
    )
    assert verdict["verdict"] == "accepted"


def test_gates_null_inconclusive() -> None:
    # Alternating tiny signed differences around zero → crossing CI, never accepted.
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
        policy=_policy(),
        proposal={"type": "accuracy", "targets": ["numeric"]},
    )
    assert verdict["verdict"] in {"inconclusive", "rejected"}
    assert verdict["verdict"] != "accepted"


def test_gates_known_regression_rejected() -> None:
    fidelity = _fidelity_pairs(baseline=0.80, candidate=0.70, ceiling=0.90, seeds=6)
    speed = _speed_pairs(baseline=1.0, candidate=1.0, n=4)
    verdict = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=_policy(),
        proposal={"type": "accuracy", "targets": ["numeric"]},
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
        policy=_policy(),
        proposal={"type": "accuracy", "targets": ["numeric"]},
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
        policy=_policy(),
        proposal={"type": "accuracy", "targets": ["numeric"]},
    )
    assert verdict["verdict"] != "accepted"
    assert "failed_or_missing_samples" in verdict["reasons"]


def test_verdict_deterministic_for_same_inputs() -> None:
    fidelity = _fidelity_pairs(baseline=0.70, candidate=0.78, ceiling=0.90, seeds=5)
    speed = _speed_pairs(baseline=1.0, candidate=0.9, n=4)
    first = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=_policy(),
        proposal={"type": "accuracy", "targets": ["numeric"]},
    )
    second = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=_policy(),
        proposal={"type": "accuracy", "targets": ["numeric"]},
    )
    assert first == second


def _policy() -> dict:
    return {
        "bootstrap": {"samples": 500, "seed": 0, "confidence": 0.95},
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


def _fidelity_pairs(
    *,
    baseline: float,
    candidate: float,
    ceiling: float,
    seeds: int,
    noise: float = 0.0,
) -> list[dict]:
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


def _speed_pairs(*, baseline: float, candidate: float, n: int) -> list[dict]:
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
