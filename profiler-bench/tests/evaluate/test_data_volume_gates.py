"""Data-volume n* search and gate decisions on synthetic curves."""

from __future__ import annotations

from typing import Any

from profiler_bench.evaluate.data_volume import (
    calibrate_targets_from_records,
    evaluate_data_volume_gates,
    geometric_mean,
    search_nstar,
)
from profiler_bench.evaluate.gates import evaluate_gates


FRACTIONS = (0.05, 0.1, 0.2, 0.5, 1.0)


def test_nstar_search_known_curve() -> None:
    # Relative scores clear T_d=0.80 from f=0.2 upward.
    curve = {
        0.05: [0.50, 0.51, 0.49, 0.50],
        0.1: [0.70, 0.71, 0.69, 0.70],
        0.2: [0.90, 0.91, 0.89, 0.90],
        0.5: [0.95, 0.95, 0.95, 0.95],
        1.0: [0.96, 0.96, 0.96, 0.96],
    }
    result = search_nstar(curve, target=0.80, noise_width=0.0, fractions=FRACTIONS, n_bootstrap=200, seed=0)
    assert result["n_star"] == 0.2
    assert result["flagged"] is False
    assert result["curve"][2]["status"] == "reached"


def test_nstar_flagged_when_never_reached() -> None:
    curve = {fraction: [0.5, 0.5, 0.5, 0.5] for fraction in FRACTIONS}
    result = search_nstar(curve, target=0.90, noise_width=0.0, fractions=FRACTIONS, n_bootstrap=100, seed=1)
    assert result["n_star"] == 1.0
    assert result["flagged"] is True


def test_gate_accepted_on_improved_nstar() -> None:
    fidelity = _curve_records(
        baseline={0.05: 0.5, 0.1: 0.5, 0.2: 0.5, 0.5: 0.85, 1.0: 0.90},
        candidate={0.05: 0.5, 0.1: 0.5, 0.2: 0.85, 0.5: 0.90, 1.0: 0.91},
        seeds=4,
    )
    verdict = evaluate_gates(
        speed_records=[],
        fidelity_records=fidelity,
        policy=_policy(target=0.80),
        proposal={"type": "data_volume", "targets": ["numeric"]},
        run=_run(),
    )
    assert verdict["verdict"] == "accepted"
    assert verdict["n_star"]["baseline"]["numeric"]["n_star"] == 0.5
    assert verdict["n_star"]["candidate"]["numeric"]["n_star"] == 0.2


def test_gate_rejected_when_nstar_increases() -> None:
    fidelity = _curve_records(
        baseline={0.05: 0.5, 0.1: 0.5, 0.2: 0.85, 0.5: 0.90, 1.0: 0.91},
        candidate={0.05: 0.5, 0.1: 0.5, 0.2: 0.5, 0.5: 0.85, 1.0: 0.90},
        seeds=4,
    )
    verdict = evaluate_data_volume_gates(
        fidelity_records=fidelity,
        policy=_policy(target=0.80),
        proposal={"type": "data_volume", "targets": ["numeric"]},
        run=_run(),
    )
    assert verdict["verdict"] == "rejected"
    assert "nstar_increased" in verdict["reasons"]


def test_gate_inconclusive_when_fraction_ci_crosses() -> None:
    # Wide within-fraction scatter so the bootstrap CI straddles T_d at the
    # candidate's first clearing fraction; improvement is then inconclusive.
    fidelity = _curve_records(
        baseline={0.05: 0.5, 0.1: 0.5, 0.2: 0.5, 0.5: 0.85, 1.0: 0.90},
        candidate={0.05: 0.5, 0.1: 0.5, 0.2: 0.85, 0.5: 0.90, 1.0: 0.91},
        seeds=4,
    )
    for record in fidelity:
        if record["role"] == "candidate" and record["fraction"] == 0.2:
            # Alternate high/low so the mean is near T_d but the CI is wide.
            record["relative_reconstruction"] = 0.95 if record["seed"] % 2 == 0 else 0.65
            record["reconstruction"] = record["relative_reconstruction"]
    verdict = evaluate_data_volume_gates(
        fidelity_records=fidelity,
        policy=_policy(target=0.80),
        proposal={"type": "data_volume", "targets": ["numeric"]},
        run=_run(),
    )
    assert verdict["verdict"] == "inconclusive"
    assert "nstar_inconclusive" in verdict["reasons"]


def test_gate_control_never_accepted() -> None:
    fidelity = _curve_records(
        baseline={0.05: 0.5, 0.1: 0.5, 0.2: 0.5, 0.5: 0.85, 1.0: 0.90},
        candidate={0.05: 0.5, 0.1: 0.5, 0.2: 0.85, 0.5: 0.90, 1.0: 0.91},
        seeds=4,
    )
    run = _run()
    run["control"] = True
    run["baseline"] = {"profiler_src_hash": "abc"}
    run["candidate"] = {"profiler_src_hash": "abc"}
    verdict = evaluate_data_volume_gates(
        fidelity_records=fidelity,
        policy=_policy(target=0.80),
        proposal={"type": "data_volume", "targets": ["numeric"]},
        run=run,
    )
    assert verdict["verdict"] != "accepted"
    assert "control" in verdict["reasons"]


def test_gate_incomplete_samples_rejected() -> None:
    fidelity = _curve_records(
        baseline={0.05: 0.5, 0.1: 0.5, 0.2: 0.85, 0.5: 0.90, 1.0: 0.91},
        candidate={0.05: 0.5, 0.1: 0.5, 0.2: 0.85, 0.5: 0.90, 1.0: 0.91},
        seeds=2,
    )[:-1]
    verdict = evaluate_data_volume_gates(
        fidelity_records=fidelity,
        policy=_policy(target=0.80, subset_seeds=1, fidelity_seeds=2),
        proposal={"type": "data_volume", "targets": ["numeric"]},
        run=_run(subset_seeds=1, fidelity_seeds=2),
    )
    assert verdict["verdict"] == "rejected"
    assert "failed_or_missing_samples" in verdict["reasons"]


def test_gate_privacy_failure_rejects() -> None:
    fidelity = _curve_records(
        baseline={0.05: 0.5, 0.1: 0.5, 0.2: 0.5, 0.5: 0.85, 1.0: 0.90},
        candidate={0.05: 0.5, 0.1: 0.5, 0.2: 0.85, 0.5: 0.90, 1.0: 0.91},
        seeds=4,
    )
    for record in fidelity:
        if record["role"] == "candidate" and record["fraction"] == 0.2:
            record["privacy_passed"] = False
    verdict = evaluate_data_volume_gates(
        fidelity_records=fidelity,
        policy=_policy(target=0.80),
        proposal={"type": "data_volume", "targets": ["numeric"]},
        run=_run(),
    )
    assert verdict["verdict"] == "rejected"
    assert "privacy_guard_failed" in verdict["reasons"]


def test_calibrate_targets_from_f1_records() -> None:
    records = []
    for seed in range(4):
        records.append(
            {
                "kind": "fidelity",
                "role": "baseline",
                "dataset": "numeric",
                "fraction": 1.0,
                "subset_seed": 0,
                "seed": seed,
                "ok": True,
                "relative_reconstruction": 0.84,
            }
        )
    targets = calibrate_targets_from_records(records, margin=0.01)
    assert targets["numeric"]["T_d"] == 0.83
    assert "provenance" in targets["numeric"]


def test_geometric_mean() -> None:
    assert geometric_mean([0.1, 0.4]) == 0.2


def _curve_records(
    *,
    baseline: dict[float, float],
    candidate: dict[float, float],
    seeds: int,
    dataset: str = "numeric",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for role, curve in (("baseline", baseline), ("candidate", candidate)):
        for fraction, relative in curve.items():
            for seed in range(seeds):
                records.append(
                    {
                        "kind": "fidelity",
                        "role": role,
                        "dataset": dataset,
                        "tier": "xs",
                        "fraction": fraction,
                        "subset_seed": 0,
                        "seed": seed,
                        "ok": True,
                        "reconstruction": relative,
                        "structure": 0.5,
                        "privacy_passed": True,
                        "ceiling": 1.0,
                        "relative_reconstruction": relative,
                    }
                )
    return records


def _policy(*, target: float, subset_seeds: int = 1, fidelity_seeds: int = 4) -> dict[str, Any]:
    return {
        "bootstrap": {"samples": 200, "seed": 0, "confidence": 0.95},
        "fractions": list(FRACTIONS),
        "subset_seeds": subset_seeds,
        "fidelity_seeds": fidelity_seeds,
        "fidelity_tier": "xs",
        "targets": {"numeric": {"T_d": target, "provenance": "test"}},
        "data_volume": {"control_noise_width": {"numeric": 0.0}},
        "accuracy": {"max_dataset_regression": 0.01},
    }


def _run(*, subset_seeds: int = 1, fidelity_seeds: int = 4) -> dict[str, Any]:
    return {
        "control": False,
        "datasets": ["numeric"],
        "fractions": list(FRACTIONS),
        "subset_seeds": subset_seeds,
        "fidelity_seeds": fidelity_seeds,
        "fidelity_tier": "xs",
        "baseline": {"profiler_src_hash": "base"},
        "candidate": {"profiler_src_hash": "cand"},
        "policy": _policy(target=0.80, subset_seeds=subset_seeds, fidelity_seeds=fidelity_seeds),
    }
