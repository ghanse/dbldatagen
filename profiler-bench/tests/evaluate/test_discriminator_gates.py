"""Discriminator-first accuracy gate (policy 3.x): paths, guards, and re-evaluation of composite policies."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from profiler_bench.evaluate.gates import evaluate_gates
from profiler_bench.runner.policy import DEFAULT_POLICY, apply_screen_overrides, load_policy


SEEDS = range(4)
DATASETS = ["numeric", "retail", "temporal"]
REAL = ["nyctaxi_trips"]
GROUPS = {"keys": 1.0, "foreign_keys": 0.9, "constraints": 0.95}


def _policy() -> dict[str, Any]:
    policy = load_policy(DEFAULT_POLICY)
    policy.update(fidelity_seeds=len(SEEDS))
    policy["bootstrap"]["samples"] = 300
    return policy


def _run(policy: dict[str, Any], *, real: bool = False, **extra: object) -> dict[str, Any]:
    run = {
        "suite": "all" if real else "synthetic",
        "control": False,
        "datasets": DATASETS,
        "speed_tiers": ["s"],
        "fidelity_tier": "s",
        "speed_samples_effective": {"s": 1},
        "baseline": {"profiler_src_hash": "a"},
        "candidate": {"profiler_src_hash": "b"},
        "policy": policy,
    }
    if real:
        run |= {"real_datasets": REAL, "real_skipped": {}, "real_split": {"version": 2}, "real_speed_tiers": ["s"]}
    return run | extra


def _record(dataset: str, role: str, seed: int, **values: object) -> dict[str, Any]:
    real = dataset in REAL
    return {
        "kind": "fidelity",
        "role": role,
        "dataset": dataset,
        "tier": "s",
        "seed": seed,
        "ok": True,
        "relative_reconstruction": values.get("relative", 0.8),
        "structure": None if real else values.get("structure", 0.7),
        **({"structure_status": "not_applicable"} if real else {}),
        "privacy_passed": values.get("privacy", True),
        "discriminator_gap": values.get("gap", 0.5),
        "discriminator_auc": 0.9,
        "discriminator_log_loss": 0.3,
        "reconstruction_groups": values.get("groups", GROUPS),
    }


def _records(
    candidate: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]] | None = None, real: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns speed and fidelity records; `candidate` and `baseline` override values per dataset."""
    baseline = baseline or {}
    datasets = DATASETS + (REAL if real else [])
    speed = [
        {"kind": "speed", "role": role, "dataset": d, "tier": "s", "sample_index": 0, "ok": True, "wall_s": 1.0}
        for d in datasets
        for role in ("baseline", "candidate")
    ]
    fidelity = []
    for dataset in datasets:
        for seed in SEEDS:
            jitter = 0.001 * seed
            base = {"gap": 0.5} | baseline.get(dataset, {})
            cand = {"gap": 0.5} | candidate.get(dataset, {})
            base["gap"] = base["gap"] + jitter
            cand["gap"] = cand["gap"] + jitter + 0.0005 * (seed % 2)
            fidelity += [_record(dataset, "baseline", seed, **base), _record(dataset, "candidate", seed, **cand)]
    return speed, fidelity


def _evaluate(
    candidate: dict[str, dict[str, Any]],
    *,
    baseline: dict[str, dict[str, Any]] | None = None,
    targets: list[str] | None = None,
    real: bool = False,
    policy: dict[str, Any] | None = None,
    **run_extra: object,
) -> dict[str, Any]:
    policy = policy or _policy()
    speed, fidelity = _records(candidate, baseline, real)
    proposal: dict[str, Any] = {"type": "accuracy"}
    if targets:
        proposal["targets"] = {"datasets": targets}
    return evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=policy,
        proposal=proposal,
        run=_run(policy, real=real, **run_extra),
    )


def _gate(verdict: dict[str, Any], name: str) -> dict[str, Any]:
    return next(g for g in verdict["gates"] if g["name"] == name)


def test_policy_is_discriminator_first() -> None:
    policy = load_policy(DEFAULT_POLICY)
    assert policy["version"] == "3.0.0"
    assert policy["accuracy_metric"] == "discriminator"
    assert policy["discriminator"]["min_mean_improvement"] == 0.01
    assert policy["discriminator"]["min_target_improvement"] == 0.02


def test_global_path_accepts_a_mean_gap_improvement() -> None:
    verdict = _evaluate({"numeric": {"gap": 0.47}, "retail": {"gap": 0.48}})
    assert verdict["verdict"] == "accepted"
    assert verdict["accuracy_path"] == "mean"
    assert verdict["accuracy_metric"] == "discriminator"
    assert verdict["discriminator"]["numeric"]["baseline_gap"] > verdict["discriminator"]["numeric"]["candidate_gap"]
    assert verdict["discriminator"]["numeric"]["regression"]["high"] < 0


def test_global_path_rejects_an_improvement_below_the_floor() -> None:
    verdict = _evaluate({"numeric": {"gap": 0.495}, "retail": {"gap": 0.497}})
    assert verdict["verdict"] == "rejected"
    assert _gate(verdict, "accuracy_mean")["reason"] == "accuracy_insufficient"


def test_targeted_path_accepts_one_improved_target() -> None:
    verdict = _evaluate({"retail": {"gap": 0.475}}, targets=["retail", "numeric"])
    assert _gate(verdict, "accuracy_mean")["status"] != "pass"
    assert verdict["verdict"] == "accepted"
    assert verdict["accuracy_path"] == "targeted"
    assert _gate(verdict, "accuracy_targeted")["detail"]["outcome"]["improved"] == ["retail"]


def test_targeted_path_rejects_a_worse_target() -> None:
    verdict = _evaluate({"retail": {"gap": 0.47}, "numeric": {"gap": 0.51}}, targets=["retail", "numeric"])
    assert _gate(verdict, "accuracy_targeted")["reason"] == "target_worsened"
    assert verdict["verdict"] == "rejected"


def test_targeted_path_is_inconclusive_when_ci_straddles_zero() -> None:
    policy = _policy()
    speed, fidelity = _records({})
    improvements = [0.04, 0.04, -0.01, -0.01]
    baseline = {
        (record["dataset"], record["seed"]): record["discriminator_gap"]
        for record in fidelity
        if record["role"] == "baseline"
    }
    for record in fidelity:
        if record["dataset"] == "numeric" and record["role"] == "candidate":
            record["discriminator_gap"] = baseline[("numeric", record["seed"])] - improvements[record["seed"]]
    verdict = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=policy,
        proposal={"type": "accuracy", "targets": {"datasets": ["numeric"]}},
        run=_run(policy),
    )
    gate = _gate(verdict, "accuracy_targeted")
    assert gate["status"] == "inconclusive"
    assert gate["reason"] == "target_improvement_inconclusive"


def test_targeted_path_rejects_improvement_below_required_minimum() -> None:
    verdict = _evaluate({"numeric": {"gap": 0.49}}, targets=["numeric"])
    gate = _gate(verdict, "accuracy_targeted")
    assert gate["status"] == "fail"
    assert gate["reason"] == "target_improvement_insufficient"


def test_saturated_target_cannot_carry_the_targeted_path() -> None:
    verdict = _evaluate({"numeric": {"gap": -0.025}}, baseline={"numeric": {"gap": 0.001}}, targets=["numeric"])
    gate = _gate(verdict, "accuracy_targeted")
    assert gate["reason"] == "targets_saturated"
    assert gate["detail"]["targets"]["numeric"]["saturated"] is True
    assert gate["detail"]["outcome"]["saturated"] == ["numeric"]
    assert gate["detail"]["outcome"]["improved"] == []
    assert verdict["verdict"] == "rejected"


def test_dataset_gap_worsening_is_rejected_even_when_the_mean_improves() -> None:
    verdict = _evaluate({"numeric": {"gap": 0.40}, "retail": {"gap": 0.53}})
    assert _gate(verdict, "accuracy_mean")["status"] == "pass"
    assert _gate(verdict, "discriminator_per_dataset")["reason"] == "discriminator_regression"
    assert verdict["verdict"] == "rejected"


@pytest.mark.parametrize("group", ["keys", "foreign_keys", "constraints"])
def test_guard_group_regression_is_rejected(group: str) -> None:
    groups = GROUPS | {group: GROUPS[group] - 0.05}
    verdict = _evaluate({"numeric": {"gap": 0.45}, "retail": {"gap": 0.45, "groups": groups}})
    gate = _gate(verdict, "reconstruction_groups")
    assert gate["reason"] == "group_regression"
    assert gate["detail"]["regressions"][f"retail:{group}"]["low"] > 0.02
    assert verdict["verdict"] == "rejected"


def test_composite_regression_is_a_guard() -> None:
    verdict = _evaluate({"numeric": {"gap": 0.45}, "retail": {"gap": 0.45, "relative": 0.75}})
    assert _gate(verdict, "accuracy_per_dataset")["reason"] == "reconstruction_regression"
    assert verdict["verdict"] == "rejected"


def test_structure_regression_is_a_guard() -> None:
    verdict = _evaluate({"numeric": {"gap": 0.45, "structure": 0.6}, "retail": {"gap": 0.45}})
    assert _gate(verdict, "structure")["reason"] == "structure_regression"
    assert verdict["verdict"] == "rejected"


def test_privacy_failure_is_rejected() -> None:
    verdict = _evaluate({"numeric": {"gap": 0.45, "privacy": False}, "retail": {"gap": 0.45}})
    assert _gate(verdict, "privacy")["status"] == "fail"
    assert verdict["verdict"] == "rejected"


def test_control_run_is_never_accepted() -> None:
    verdict = _evaluate({"numeric": {"gap": 0.45}, "retail": {"gap": 0.45}}, control=True)
    assert _gate(verdict, "control")["status"] == "inconclusive"
    assert verdict["verdict"] == "inconclusive"


def test_real_holdout_reports_and_guards_the_discriminator_gap() -> None:
    improved = _evaluate({"numeric": {"gap": 0.45}, "retail": {"gap": 0.45}}, real=True)
    holdout = _gate(improved, "real_holdout")
    detail = holdout["detail"]["datasets"]["nyctaxi_trips"]["discriminator"]
    assert {"baseline_gap", "candidate_gap", "baseline_auc", "candidate_auc", "regression"} <= set(detail)
    assert improved["verdict"] == "accepted"
    assert "nyctaxi_trips" not in improved["discriminator"]
    worse = _evaluate({"numeric": {"gap": 0.45}, "retail": {"gap": 0.45}, "nyctaxi_trips": {"gap": 0.6}}, real=True)
    holdout = _gate(worse, "real_holdout")
    assert holdout["reason"] == "real_holdout_regression"
    assert holdout["detail"]["regressed"] == ["nyctaxi_trips"]
    assert worse["verdict"] == "inconclusive"


def test_real_holdout_privacy_failure_rejects() -> None:
    verdict = _evaluate(
        {"numeric": {"gap": 0.45}, "retail": {"gap": 0.45}, "nyctaxi_trips": {"privacy": False}}, real=True
    )
    assert _gate(verdict, "real_holdout")["reason"] == "real_holdout_privacy_failed"
    assert verdict["verdict"] == "rejected"


def test_screen_stage_uses_loosened_discriminator_thresholds() -> None:
    policy = apply_screen_overrides(_policy()) | {"fidelity_seeds": len(SEEDS)}
    assert policy["discriminator"]["min_mean_improvement"] == pytest.approx(0.005)
    assert policy["discriminator"]["max_dataset_regression"] == pytest.approx(0.03)
    assert policy["discriminator"]["max_group_regression"] == pytest.approx(0.03)
    borderline = {name: {"gap": 0.493} for name in DATASETS}
    assert _evaluate(borderline)["verdict"] == "rejected"
    screened = _evaluate(borderline, policy=policy, stage="screen")
    assert screened["verdict"] == "screen_pass"
    assert screened["accuracy_metric"] == "discriminator"
    worse = _evaluate({"numeric": {"gap": 0.40}, "retail": {"gap": 0.56}}, policy=policy, stage="screen")
    assert worse["verdict"] == "screen_fail"


def test_composite_policy_re_evaluates_under_its_own_rules() -> None:
    policy = copy.deepcopy(_policy())
    del policy["accuracy_metric"], policy["discriminator"]
    policy["version"] = "2.2.0"
    policy["accuracy"] |= {"min_mean_improvement": 0.005, "min_target_improvement": 0.01}
    verdict = _evaluate(
        {"numeric": {"gap": 0.9, "relative": 0.83}, "retail": {"gap": 0.9, "relative": 0.83}}, policy=policy
    )
    assert verdict["verdict"] == "accepted"
    assert verdict["accuracy_metric"] == "composite"
    assert "discriminator" not in verdict
    assert all(g["name"] not in ("discriminator_per_dataset", "reconstruction_groups") for g in verdict["gates"])
