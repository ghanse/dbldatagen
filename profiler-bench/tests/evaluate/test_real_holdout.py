"""The real-dataset holdout gate: excluded from acceptance, rejects on privacy, inconclusive on regression."""

from __future__ import annotations

from typing import Any

import pytest

from profiler_bench.evaluate.gates import evaluate_gates
from profiler_bench.runner.policy import DEFAULT_POLICY, load_policy


SEEDS = range(3)
SYNTHETIC = ["numeric"]
REAL = ["nyctaxi_trips"]


def _policy() -> dict[str, Any]:
    policy = load_policy(DEFAULT_POLICY)
    policy.update(fidelity_seeds=len(SEEDS))
    policy["bootstrap"]["samples"] = 200
    return policy


def _run(suite: str = "all") -> dict[str, Any]:
    run = {
        "suite": suite,
        "control": False,
        "datasets": SYNTHETIC,
        "speed_tiers": ["s"],
        "fidelity_tier": "s",
        "speed_samples_effective": {"s": 1},
        "baseline": {"profiler_src_hash": "a"},
        "candidate": {"profiler_src_hash": "b"},
        "policy": _policy(),
    }
    if suite != "synthetic":
        run |= {"real_datasets": REAL, "real_skipped": {}, "real_split": {"version": 1}, "real_speed_tiers": ["s"]}
    return run


def _speed(dataset: str) -> list[dict[str, Any]]:
    return [
        {"kind": "speed", "role": role, "dataset": dataset, "tier": "s", "sample_index": 0, "ok": True, "wall_s": 1.0}
        for role in ("baseline", "candidate")
    ]


def _fidelity(dataset: str, role: str, seed: int, relative: float, privacy: bool = True) -> dict[str, Any]:
    real = dataset in REAL
    return {
        "kind": "fidelity",
        "role": role,
        "dataset": dataset,
        "tier": "s",
        "seed": seed,
        "ok": True,
        "relative_reconstruction": relative,
        "structure": None if real else 0.7,
        **({"structure_status": "not_applicable"} if real else {}),
        "privacy_passed": privacy,
    }


def _records(
    real_candidate: float = 0.8, real_privacy: bool = True, suite: str = "all"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    datasets = SYNTHETIC + (REAL if suite != "synthetic" else [])
    speed = [record for dataset in datasets for record in _speed(dataset)]
    fidelity = []
    for seed in SEEDS:
        fidelity += [
            _fidelity("numeric", "baseline", seed, 0.80),
            _fidelity("numeric", "candidate", seed, 0.83 + 0.001 * seed),
        ]
        if suite != "synthetic":
            fidelity += [
                _fidelity("nyctaxi_trips", "baseline", seed, 0.8),
                _fidelity("nyctaxi_trips", "candidate", seed, real_candidate + 0.001 * seed, real_privacy),
            ]
    return speed, fidelity


def _verdict(**kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
    suite = kwargs.get("suite", "all")
    speed, fidelity = _records(**kwargs)
    return evaluate_gates(
        speed_records=speed, fidelity_records=fidelity, policy=_policy(), proposal={"type": "accuracy"}, run=_run(suite)
    )


def _gate(verdict: dict[str, Any], name: str) -> dict[str, Any]:
    return next(gate for gate in verdict["gates"] if gate["name"] == name)


def test_synthetic_verdict_has_no_holdout_gate() -> None:
    verdict = _verdict(suite="synthetic")
    assert verdict["verdict"] == "accepted"
    assert [g["name"] for g in verdict["gates"]] == [
        "sample_completeness",
        "privacy",
        "control",
        "accuracy_mean",
        "accuracy_per_dataset",
        "structure",
    ]


def test_holdout_passes_and_stays_out_of_the_accuracy_gates() -> None:
    verdict = _verdict(real_candidate=0.805)
    assert verdict["verdict"] == "accepted"
    gate = _gate(verdict, "real_holdout")
    assert gate["reason"] == "real_holdout_passed"
    detail = gate["detail"]["datasets"]["nyctaxi_trips"]
    assert {"baseline_relative_reconstruction", "candidate_relative_reconstruction", "regression"} <= set(detail)
    assert detail["candidate_privacy_passed"] is True
    assert set(_gate(verdict, "accuracy_mean")["detail"]["per_dataset_mean_improvement"]) == {"numeric"}
    assert set(_gate(verdict, "structure")["detail"]["datasets"]) == {"numeric"}


def test_real_regression_makes_the_verdict_inconclusive() -> None:
    verdict = _verdict(real_candidate=0.7)
    assert _gate(verdict, "accuracy_mean")["status"] == "pass"
    assert _gate(verdict, "real_holdout")["reason"] == "real_holdout_regression"
    assert verdict["verdict"] == "inconclusive"


def test_real_privacy_failure_rejects() -> None:
    verdict = _verdict(real_candidate=0.9, real_privacy=False)
    gate = _gate(verdict, "real_holdout")
    assert (gate["status"], gate["reason"]) == ("fail", "real_holdout_privacy_failed")
    assert _gate(verdict, "privacy")["status"] == "pass"
    assert verdict["verdict"] == "rejected"


def test_missing_real_samples_are_inconclusive_not_a_completeness_failure() -> None:
    speed, fidelity = _records()
    fidelity = [r for r in fidelity if not (r["dataset"] in REAL and r["role"] == "candidate" and r["seed"] == 0)]
    verdict = evaluate_gates(
        speed_records=speed, fidelity_records=fidelity, policy=_policy(), proposal={"type": "accuracy"}, run=_run()
    )
    assert _gate(verdict, "sample_completeness")["status"] == "pass"
    assert _gate(verdict, "real_holdout")["reason"] == "real_holdout_incomplete"
    assert verdict["verdict"] == "inconclusive"


def test_all_real_datasets_skipped_passes_with_reason() -> None:
    speed, fidelity = _records(suite="synthetic")
    run = _run() | {"real_datasets": [], "real_skipped": {"nyctaxi_trips": "Unavailable: offline"}}
    verdict = evaluate_gates(
        speed_records=speed, fidelity_records=fidelity, policy=_policy(), proposal={"type": "accuracy"}, run=run
    )
    gate = _gate(verdict, "real_holdout")
    assert (gate["status"], gate["reason"]) == ("pass", "real_holdout_skipped")
    assert gate["detail"]["skipped"] == {"nyctaxi_trips": "Unavailable: offline"}
    assert verdict["verdict"] == "accepted"


def _verdict_with(fidelity: list[dict[str, Any]], speed: list[dict[str, Any]], **policy: float) -> dict[str, Any]:
    full = _policy()
    full["accuracy"] = {**full["accuracy"], **policy}
    return evaluate_gates(
        speed_records=speed, fidelity_records=fidelity, policy=full, proposal={"type": "accuracy"}, run=_run()
    )


def _set_candidate_privacy(fidelity: list[dict[str, Any]], statuses: list[str]) -> None:
    for record in fidelity:
        if record["dataset"] in REAL and record["role"] == "candidate":
            status = statuses[record["seed"]]
            record["privacy_status"], record["privacy_passed"] = status, status == "pass"


def test_unmeasured_real_privacy_is_inconclusive_never_accepted() -> None:
    speed, fidelity = _records(real_candidate=0.805)
    _set_candidate_privacy(fidelity, ["unmeasured"] * len(SEEDS))
    verdict = _verdict_with(fidelity, speed)
    gate = _gate(verdict, "real_holdout")
    assert (gate["status"], gate["reason"]) == ("inconclusive", "real_holdout_privacy_unmeasured")
    detail = gate["detail"]["datasets"]["nyctaxi_trips"]
    assert (detail["candidate_privacy_status"], detail["candidate_privacy_passed"]) == ("unmeasured", False)
    assert gate["detail"]["privacy_failed"] == []
    assert verdict["verdict"] == "inconclusive"


def test_failed_real_privacy_rejects_even_when_other_seeds_are_unmeasured() -> None:
    speed, fidelity = _records(real_candidate=0.805)
    _set_candidate_privacy(fidelity, ["unmeasured", "fail", "pass"])
    verdict = _verdict_with(fidelity, speed)
    gate = _gate(verdict, "real_holdout")
    assert (gate["status"], gate["reason"]) == ("fail", "real_holdout_privacy_failed")
    assert gate["detail"]["datasets"]["nyctaxi_trips"]["candidate_privacy_status"] == "fail"
    assert verdict["verdict"] == "rejected"


def test_measured_privacy_reports_unmeasured_checks_and_overlap() -> None:
    speed, fidelity = _records(real_candidate=0.805)
    _set_candidate_privacy(fidelity, ["pass"] * len(SEEDS))
    for record in fidelity:
        if record["dataset"] in REAL and record["role"] == "candidate":
            record["privacy_unmeasured"] = ["trips.dcr_ratio"]
            record["detail"] = {"holdout_overlap": {"trips": {"holdout_overlap_share": 0.25}}}
    verdict = _verdict_with(fidelity, speed)
    detail = _gate(verdict, "real_holdout")["detail"]["datasets"]["nyctaxi_trips"]
    assert detail["candidate_privacy_status"] == "pass"
    assert detail["candidate_privacy_unmeasured"] == ["trips.dcr_ratio"]
    assert detail["holdout_overlap"] == {"trips": {"holdout_overlap_share": 0.25}}
    assert verdict["verdict"] == "accepted"


def test_regression_at_exactly_the_limit_is_not_a_regression() -> None:
    limit = 0.0078125
    speed, fidelity = _records()
    for record in fidelity:
        if record["dataset"] in REAL:
            record["relative_reconstruction"] = 0.5 if record["role"] == "baseline" else 0.5 - limit
    at_limit = _gate(_verdict_with(fidelity, speed, max_dataset_regression=limit), "real_holdout")
    assert at_limit["detail"]["datasets"]["nyctaxi_trips"]["regression"]["high"] == limit
    assert at_limit["reason"] == "real_holdout_passed"
    above = _gate(_verdict_with(fidelity, speed, max_dataset_regression=limit / 2), "real_holdout")
    assert above["reason"] == "real_holdout_regression"


TARGETED = {"numeric": 0.02, "retail": -0.005, "strings": -0.005}


def _targeted_verdict(real_candidate: float, real_privacy: str) -> dict[str, Any]:
    run = _run() | {"datasets": sorted(TARGETED)}
    speed = [record for dataset in [*TARGETED, *REAL] for record in _speed(dataset)]
    fidelity = []
    for seed in SEEDS:
        for dataset, difference in TARGETED.items():
            fidelity += [
                _fidelity(dataset, "baseline", seed, 0.7),
                _fidelity(dataset, "candidate", seed, 0.7 + difference),
            ]
        candidate = _fidelity("nyctaxi_trips", "candidate", seed, real_candidate, real_privacy == "pass")
        fidelity += [_fidelity("nyctaxi_trips", "baseline", seed, 0.8), candidate | {"privacy_status": real_privacy}]
    return evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=_policy(),
        proposal={"type": "accuracy", "targets": {"datasets": ["numeric"]}},
        run=run,
    )


@pytest.mark.parametrize(
    ("real_candidate", "real_privacy", "verdict", "reason"),
    [
        (0.8, "pass", "accepted", None),
        (0.7, "pass", "inconclusive", "real_holdout_regression"),
        (0.8, "fail", "rejected", "real_holdout_privacy_failed"),
        (0.8, "unmeasured", "inconclusive", "real_holdout_privacy_unmeasured"),
    ],
)
def test_targeted_accept_is_bound_by_the_real_holdout(
    real_candidate: float, real_privacy: str, verdict: str, reason: str | None
) -> None:
    result = _targeted_verdict(real_candidate, real_privacy)
    assert _gate(result, "accuracy_targeted")["status"] == "pass"
    assert _gate(result, "accuracy_mean")["status"] == "fail"
    assert result["verdict"] == verdict
    if verdict == "accepted":
        assert (result["accuracy_path"], result["reasons"]) == ("targeted", [])
    else:
        assert "accuracy_path" not in result
        assert reason in result["reasons"]
        assert _gate(result, "real_holdout")["reason"] == reason
