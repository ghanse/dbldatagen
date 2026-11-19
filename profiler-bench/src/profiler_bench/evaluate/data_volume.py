"""n* search and data-volume acceptance gates.

For each dataset the harness records relative reconstruction on a fixed geometric
fraction grid. ``n*_d`` is the smallest fraction whose paired-bootstrap CI lower
bound on the mean relative score clears ``T_d`` minus the frozen control noise
width. A ``data_volume`` proposal is accepted when at least one declared target
improves by one grid step, no dataset's ``n*`` increases, reconstruction at
``f = 1`` does not regress beyond the default policy's per-dataset bound (loaded
by reference from ``default.json``), privacy passes on the candidate (baseline
privacy failures are inconclusive), and every expected sample is present on both
builds. Control runs are never accepted.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from profiler_bench.evaluate.bootstrap import paired_bootstrap_ci
from profiler_bench.evaluate.gates import GateResult, _combine, _control_guard
from profiler_bench.runner.policy import DEFAULT_POLICY, load_policy, policy_path_hash
from profiler_bench.runner.subset import DEFAULT_FRACTIONS


ReachedStatus = Literal["reached", "not_reached", "inconclusive"]


def default_fractions() -> tuple[float, ...]:
    """Returns the frozen geometric input-fraction grid."""
    return DEFAULT_FRACTIONS


def search_nstar(
    curve: Mapping[float, Sequence[float]],
    *,
    target: float,
    noise_width: float,
    fractions: Sequence[float] | None = None,
    n_bootstrap: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
    actual_fractions: Mapping[float, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Finds ``n*`` on ``fractions`` for one dataset from per-fraction scores.

    A fraction is reached only when the bootstrap CI lower bound on the mean
    relative reconstruction is ``>= target - noise_width``. When the interval
    straddles that threshold the fraction is inconclusive. If no fraction
    reaches the target, ``n_star`` is ``1.0`` and ``flagged`` is true.

    Gates stay on the nominal grid ``f``. Each curve point and ``n_star`` also
    record the mean realized overall input fraction (closure may pull extra rows).
    """
    grid = list(fractions) if fractions is not None else list(DEFAULT_FRACTIONS)
    actuals = actual_fractions or {}
    points: list[dict[str, Any]] = []
    n_star = 1.0
    n_star_actual: float | None = None
    flagged = True
    first_reached: float | None = None
    for index, fraction in enumerate(grid):
        scores = list(curve.get(fraction, []))
        overall_actual = _mean_or_none(actuals.get(fraction, []))
        if not scores:
            points.append(
                {
                    "fraction": fraction,
                    "overall_actual_fraction": overall_actual,
                    "status": "not_reached",
                    "reason": "missing_scores",
                    "mean": None,
                    "ci": None,
                }
            )
            continue
        ci = paired_bootstrap_ci(
            scores,
            statistic="mean",
            n_bootstrap=n_bootstrap,
            seed=seed + index,
            confidence=confidence,
        )
        threshold = float(target) - float(noise_width)
        if ci.low >= threshold:
            status: ReachedStatus = "reached"
            reason = "ci_lower_clears_threshold"
        elif ci.high < threshold:
            status = "not_reached"
            reason = "ci_upper_below_threshold"
        else:
            status = "inconclusive"
            reason = "ci_crosses_threshold"
        points.append(
            {
                "fraction": fraction,
                "overall_actual_fraction": overall_actual,
                "status": status,
                "reason": reason,
                "mean": ci.point,
                "ci": ci.to_dict(),
                "threshold": threshold,
            }
        )
        if status == "reached" and first_reached is None:
            first_reached = fraction
            n_star = fraction
            n_star_actual = overall_actual
            flagged = False
    if n_star_actual is None:
        n_star_actual = _mean_or_none(actuals.get(n_star, []))
        if n_star_actual is None:
            # Fall back to the curve point recorded for the grid n*.
            for point in points:
                if math.isclose(float(point["fraction"]), float(n_star), abs_tol=1e-12):
                    n_star_actual = point.get("overall_actual_fraction")
                    break
    return {
        "n_star": n_star,
        "n_star_overall_actual_fraction": n_star_actual,
        "flagged": flagged,
        "curve": points,
        "target": float(target),
        "noise_width": float(noise_width),
        "reached_rule": "ci_lower >= T_d - control_noise_width",
    }


def geometric_mean(values: Sequence[float]) -> float:
    """Returns the geometric mean of positive values."""
    if not values:
        raise ValueError("values must be non-empty")
    if any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def grid_index(fraction: float, fractions: Sequence[float] | None = None) -> int:
    """Returns the index of ``fraction`` on the grid."""
    grid = list(fractions) if fractions is not None else list(DEFAULT_FRACTIONS)
    for index, value in enumerate(grid):
        if math.isclose(value, fraction, rel_tol=0.0, abs_tol=1e-12):
            return index
    raise ValueError(f"fraction {fraction} is not on the grid {grid}")


def previous_grid_value(fraction: float, fractions: Sequence[float] | None = None) -> float | None:
    """Returns the grid point immediately below ``fraction``, if any."""
    grid = list(fractions) if fractions is not None else list(DEFAULT_FRACTIONS)
    index = grid_index(fraction, grid)
    return grid[index - 1] if index > 0 else None


def evaluate_data_volume_gates(
    *,
    fidelity_records: Sequence[dict[str, Any]],
    policy: dict[str, Any],
    proposal: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any]:
    """Applies the data-volume policy and returns a verdict with per-gate reasons."""
    gates: list[GateResult] = []
    expected = expected_data_volume_keys(run)
    gates.append(_data_volume_completeness(fidelity_records, expected))
    gates.extend(_data_volume_privacy_gates(fidelity_records, expected))
    gates.append(_control_guard(run))

    curves = _curves_by_role(fidelity_records)
    actuals = _actual_fractions_by_role(fidelity_records)
    boot = policy.get("bootstrap") or {}
    fractions = [float(value) for value in (run.get("fractions") or policy.get("fractions") or DEFAULT_FRACTIONS)]
    targets = _targets_from_policy(policy)
    noise = frozen_noise_widths(policy)
    per_role_nstar: dict[str, dict[str, Any]] = {}
    for role, datasets in curves.items():
        per_role_nstar[role] = {}
        for dataset, curve in datasets.items():
            if dataset not in targets:
                continue
            if dataset not in noise:
                gates.append(
                    GateResult(
                        name="data_volume_noise_width",
                        passed=False,
                        status="fail",
                        reason="missing_frozen_noise_width",
                        detail={"dataset": dataset},
                    )
                )
                continue
            per_role_nstar[role][dataset] = search_nstar(
                curve,
                target=float(targets[dataset]["T_d"]),
                noise_width=float(noise[dataset]),
                fractions=fractions,
                n_bootstrap=int(boot.get("samples", 2000)),
                seed=int(boot.get("seed", 0)),
                confidence=float(boot.get("confidence", 0.95)),
                actual_fractions=(actuals.get(role) or {}).get(dataset),
            )

    gates.append(_nstar_improvement_gate(per_role_nstar, proposal, fractions))
    gates.append(_f1_regression_gate(fidelity_records, policy, boot))
    gates.append(
        GateResult(
            name="data_volume_curves",
            passed=True,
            status="pass",
            reason="curves_recorded",
            detail={"n_star": per_role_nstar, "noise_width": noise, "targets": targets},
        )
    )

    verdict = _combine(gates)
    return {
        "verdict": verdict,
        "proposal_type": "data_volume",
        "gates": [gate.to_dict() for gate in gates],
        "reasons": [gate.reason for gate in gates if gate.status != "pass"],
        "n_star": per_role_nstar,
        "interval_note": (
            "Each per-fraction accuracy gate uses a paired bootstrap over (subset seed, sampler seed); "
            "a fraction is reached only if ci.low >= T_d - control_noise_width. "
            "Crossing intervals are inconclusive. There is no family-wise coverage guarantee."
        ),
    }


def expected_data_volume_keys(run: dict[str, Any]) -> set[str]:
    """Derives expected fidelity sample keys for a data-volume experiment."""
    policy = run.get("policy") or {}
    datasets = [str(name) for name in run.get("datasets") or []]
    tier = str(run.get("fidelity_tier") or policy.get("fidelity_tier") or "s")
    fractions = [float(value) for value in (run.get("fractions") or policy.get("fractions") or DEFAULT_FRACTIONS)]
    subset_seeds = int(run.get("subset_seeds") or policy.get("subset_seeds") or 1)
    fidelity_seeds = int(run.get("fidelity_seeds") or policy.get("fidelity_seeds") or 1)
    roles = ("baseline", "candidate")
    keys: set[str] = set()
    for dataset in datasets:
        for role in roles:
            for fraction in fractions:
                for subset_seed in range(subset_seeds):
                    for seed in range(fidelity_seeds):
                        keys.add(
                            f"fidelity/{role}/{dataset}/{tier}/f-{fraction_token(fraction)}"
                            f"/subset-{subset_seed}/seed-{seed}"
                        )
    return keys


def data_volume_sample_key(record: Mapping[str, Any]) -> str:
    """Returns the completeness key for one data-volume fidelity record."""
    return (
        f"fidelity/{record.get('role')}/{record.get('dataset')}/{record.get('tier')}"
        f"/f-{fraction_token(float(record.get('fraction', 1.0)))}"
        f"/subset-{record.get('subset_seed')}/seed-{record.get('seed')}"
    )


def calibrate_targets_from_records(
    fidelity_records: Sequence[dict[str, Any]],
    *,
    role: str = "baseline",
    fraction: float = 1.0,
    margin: float = 0.01,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 0,
    confidence: float = 0.95,
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Computes ``T_d`` and control noise widths from a same-build control at ``f``.

    ``T_d = mean_relative(f) - margin``. The noise width is half the paired-bootstrap
    CI width of that mean (K x S cells). Both are frozen into the policy by
    ``calibrate-targets``.
    """
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for record in fidelity_records:
        if record.get("kind") != "fidelity" or not record.get("ok"):
            continue
        if str(record.get("role")) != role:
            continue
        if not math.isclose(float(record.get("fraction", 1.0)), fraction, abs_tol=1e-12):
            continue
        if record.get("relative_reconstruction") is None:
            continue
        by_dataset[str(record["dataset"])].append(float(record["relative_reconstruction"]))
    targets: dict[str, dict[str, Any]] = {}
    noise: dict[str, float] = {}
    for dataset, scores in sorted(by_dataset.items()):
        mean = float(sum(scores) / len(scores))
        targets[dataset] = {
            "T_d": mean - margin,
            "baseline_relative_mean": mean,
            "margin": margin,
            "n": len(scores),
            "fraction": fraction,
            "role": role,
            "provenance": (
                f"T_d = mean relative reconstruction at f={fraction} for role={role} "
                f"minus {margin} (n={len(scores)} paired cells)"
            ),
        }
        if len(scores) < 2:
            noise[dataset] = 0.0
            continue
        ci = paired_bootstrap_ci(
            scores,
            statistic="mean",
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed + 17,
            confidence=confidence,
        )
        noise[dataset] = 0.5 * (ci.high - ci.low)
    return targets, noise


def fraction_token(fraction: float) -> str:
    """Returns a filesystem-safe token for a fraction on the grid."""
    return f"{fraction:.6f}".rstrip("0").rstrip(".")


def frozen_noise_widths(policy: Mapping[str, Any]) -> dict[str, float]:
    """Returns the frozen per-dataset control noise widths from the policy (no fallback)."""
    configured = (policy.get("data_volume") or {}).get("control_noise_width") or {}
    return {str(name): float(width) for name, width in dict(configured).items()}


def require_calibrated_targets(policy: Mapping[str, Any], datasets: Sequence[str]) -> None:
    """Raises ``ValueError`` unless every dataset has a frozen ``T_d`` and noise width."""
    targets = _targets_from_policy(policy)
    noise = frozen_noise_widths(policy)
    missing_td = [name for name in datasets if name not in targets]
    missing_noise = [name for name in datasets if name not in noise]
    if missing_td or missing_noise:
        raise ValueError(
            "data-volume policy is missing frozen calibration for selected datasets: "
            f"T_d missing={missing_td or []}; control_noise_width missing={missing_noise or []}. "
            "Run `python -m profiler_bench.rsi calibrate-targets` on a same-build control at f=1."
        )


def load_f1_regression_bound() -> tuple[float, str]:
    """Loads ``accuracy.max_dataset_regression`` from ``default.json`` and returns ``(bound, file_hash)``."""
    default_policy = load_policy(DEFAULT_POLICY)
    bound = float((default_policy.get("accuracy") or {}).get("max_dataset_regression", 0.01))
    return bound, policy_path_hash(DEFAULT_POLICY)


def _targets_from_policy(policy: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = policy.get("targets") or {}
    out: dict[str, dict[str, Any]] = {}
    for dataset, value in raw.items():
        if isinstance(value, Mapping):
            if value.get("T_d") is None:
                continue
            out[str(dataset)] = dict(value)
        else:
            out[str(dataset)] = {"T_d": float(value), "provenance": "policy_scalar"}
    return out


def _curves_by_role(
    fidelity_records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, dict[float, list[float]]]]:
    out: dict[str, dict[str, dict[float, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for record in fidelity_records:
        if record.get("kind") != "fidelity" or not record.get("ok"):
            continue
        if record.get("relative_reconstruction") is None:
            continue
        role = str(record["role"])
        dataset = str(record["dataset"])
        fraction = float(record.get("fraction", 1.0))
        out[role][dataset][fraction].append(float(record["relative_reconstruction"]))
    return {
        role: {dataset: {frac: list(scores) for frac, scores in curve.items()} for dataset, curve in datasets.items()}
        for role, datasets in out.items()
    }


def _actual_fractions_by_role(
    fidelity_records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, dict[float, list[float]]]]:
    """Collects realized overall input fractions keyed by role/dataset/grid ``f``."""
    out: dict[str, dict[str, dict[float, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for record in fidelity_records:
        if record.get("kind") != "fidelity" or not record.get("ok"):
            continue
        subset = record.get("subset") or {}
        raw = subset.get("overall_actual_fraction")
        if raw is None:
            raw = record.get("overall_actual_fraction")
        if raw is None:
            continue
        role = str(record["role"])
        dataset = str(record["dataset"])
        fraction = float(record.get("fraction", 1.0))
        out[role][dataset][fraction].append(float(raw))
    return {
        role: {dataset: {frac: list(values) for frac, values in curve.items()} for dataset, curve in datasets.items()}
        for role, datasets in out.items()
    }


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _nstar_improvement_gate(
    per_role_nstar: Mapping[str, Mapping[str, Any]],
    proposal: Mapping[str, Any],
    fractions: Sequence[float],
) -> GateResult:
    baseline = per_role_nstar.get("baseline") or {}
    candidate = per_role_nstar.get("candidate") or {}
    if not baseline or not candidate:
        return GateResult(
            name="data_volume_nstar",
            passed=False,
            status="fail",
            reason="missing_nstar_curves",
            detail={"baseline": list(baseline), "candidate": list(candidate)},
        )
    comparison = _compare_nstar_roles(baseline, candidate, proposal, fractions)
    status, reason = _decide_nstar_status(comparison)
    return GateResult(
        name="data_volume_nstar",
        passed=status == "pass",
        status=status,
        reason=reason,
        detail=comparison["detail"],
    )


def _compare_nstar_roles(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    proposal: Mapping[str, Any],
    fractions: Sequence[float],
) -> dict[str, Any]:
    datasets = sorted(set(baseline) & set(candidate))
    targets = proposal.get("targets") or datasets
    if isinstance(targets, Mapping):
        target_names = [str(name) for name in targets.get("datasets", targets)]
    else:
        target_names = [str(name) for name in targets]

    detail: dict[str, Any] = {"datasets": {}, "targets": target_names}
    increased: list[str] = []
    improved: list[str] = []
    inconclusive = False
    for dataset in datasets:
        base = float(baseline[dataset]["n_star"])
        cand = float(candidate[dataset]["n_star"])
        detail["datasets"][dataset] = {
            "baseline_n_star": base,
            "candidate_n_star": cand,
            "baseline_n_star_overall_actual_fraction": baseline[dataset].get("n_star_overall_actual_fraction"),
            "candidate_n_star_overall_actual_fraction": candidate[dataset].get("n_star_overall_actual_fraction"),
            "baseline_flagged": baseline[dataset].get("flagged"),
            "candidate_flagged": candidate[dataset].get("flagged"),
            "baseline_curve": baseline[dataset].get("curve"),
            "candidate_curve": candidate[dataset].get("curve"),
        }
        if cand > base and not math.isclose(cand, base, abs_tol=1e-12):
            increased.append(dataset)
        if dataset in target_names and grid_index(cand, fractions) <= grid_index(base, fractions) - 1:
            improved.append(dataset)
        curves = list(baseline[dataset].get("curve") or []) + list(candidate[dataset].get("curve") or [])
        inconclusive = inconclusive or any(point.get("status") == "inconclusive" for point in curves)

    base_geo = geometric_mean([float(baseline[name]["n_star"]) for name in datasets])
    cand_geo = geometric_mean([float(candidate[name]["n_star"]) for name in datasets])
    snapped = min((g for g in fractions if g >= base_geo), default=max(fractions))
    geo_prev = previous_grid_value(snapped, fractions)
    geo_improved = geo_prev is not None and cand_geo <= geo_prev + 1e-12
    detail["geometric_mean"] = {
        "baseline": base_geo,
        "candidate": cand_geo,
        "snapped_baseline_grid": snapped,
        "required_candidate_at_most": geo_prev,
        "improved_by_grid_step": geo_improved,
    }
    detail["improved_targets"] = improved
    detail["increased"] = increased
    detail["accept_rule"] = (
        "geomean(n*) decreases by >=1 grid step; at least one declared target improves by >=1 "
        "grid step; n* increases on none; crossing fraction CIs make the result inconclusive"
    )
    return {
        "detail": detail,
        "increased": increased,
        "improved": improved,
        "inconclusive": inconclusive,
        "base_geo": base_geo,
        "cand_geo": cand_geo,
        "geo_improved": geo_improved,
    }


def _decide_nstar_status(comparison: Mapping[str, Any]) -> tuple[Literal["pass", "fail", "inconclusive"], str]:
    increased = comparison["increased"]
    improved = comparison["improved"]
    inconclusive = bool(comparison["inconclusive"])
    base_geo = float(comparison["base_geo"])
    cand_geo = float(comparison["cand_geo"])
    geo_improved = bool(comparison["geo_improved"])
    status: Literal["pass", "fail", "inconclusive"] = "pass"
    reason = "nstar_improved"
    if increased:
        status, reason = "fail", "nstar_increased"
    elif cand_geo > base_geo + 1e-12:
        status, reason = "fail", "geomean_nstar_increased"
    elif not improved and inconclusive:
        status, reason = "inconclusive", "nstar_inconclusive"
    elif not improved:
        status, reason = "fail", "no_target_nstar_improvement"
    elif not geo_improved and inconclusive:
        status, reason = "inconclusive", "nstar_inconclusive"
    elif not geo_improved:
        status, reason = "fail", "geomean_nstar_not_improved_by_grid_step"
    elif inconclusive:
        status, reason = "inconclusive", "nstar_inconclusive"
    return status, reason


def _f1_regression_gate(
    fidelity_records: Sequence[dict[str, Any]],
    policy: Mapping[str, Any],
    boot: Mapping[str, Any],
) -> GateResult:
    """Reuses the default per-dataset reconstruction regression bound at f = 1 by reference."""
    del policy  # Bound comes from default.json, not the data-volume policy copy.
    max_reg, default_hash = load_f1_regression_bound()
    paired: dict[str, list[float]] = defaultdict(list)
    by: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)
    for record in fidelity_records:
        if record.get("kind") != "fidelity" or not record.get("ok"):
            continue
        if not math.isclose(float(record.get("fraction", 1.0)), 1.0, abs_tol=1e-12):
            continue
        if record.get("relative_reconstruction") is None:
            continue
        key = (str(record["dataset"]), int(record["subset_seed"]), int(record["seed"]))
        by[key][str(record["role"])] = float(record["relative_reconstruction"])
    for (dataset, _subset, _seed), roles in sorted(by.items()):
        if "baseline" in roles and "candidate" in roles:
            paired[dataset].append(roles["candidate"] - roles["baseline"])
    if not paired:
        return GateResult(
            name="data_volume_f1_regression",
            passed=False,
            status="fail",
            reason="no_f1_pairs",
        )
    details: dict[str, Any] = {}
    status: Literal["pass", "fail", "inconclusive"] = "pass"
    reason = "no_reconstruction_regression_at_f1"
    for dataset, diffs in paired.items():
        regressions = [-diff for diff in diffs]
        ci = paired_bootstrap_ci(
            regressions,
            statistic="mean",
            n_bootstrap=int(boot.get("samples", 2000)),
            seed=int(boot.get("seed", 0)) + 1,
            confidence=float(boot.get("confidence", 0.95)),
        )
        details[dataset] = ci.to_dict()
        if ci.high >= max_reg:
            if ci.low < max_reg:
                status = "inconclusive" if status != "fail" else status
                reason = "reconstruction_regression_inconclusive"
            else:
                status = "fail"
                reason = "reconstruction_regression"
    return GateResult(
        name="data_volume_f1_regression",
        passed=status == "pass",
        status=status,
        reason=reason,
        detail={
            "datasets": details,
            "max_regression": max_reg,
            "fraction": 1.0,
            "bound_source": str(DEFAULT_POLICY.name),
            "bound_path": "accuracy.max_dataset_regression",
            "default_policy_hash": default_hash,
        },
    )


def _data_volume_completeness(
    fidelity_records: Sequence[dict[str, Any]],
    expected: set[str],
) -> GateResult:
    failed = [data_volume_sample_key(record) for record in fidelity_records if not record.get("ok")]
    if failed:
        return GateResult(
            name="sample_completeness",
            passed=False,
            status="fail",
            reason="failed_or_missing_samples",
            detail={"failed": failed},
        )
    counts: dict[str, int] = defaultdict(int)
    for record in fidelity_records:
        counts[data_volume_sample_key(record)] += 1
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        return GateResult(
            name="sample_completeness",
            passed=False,
            status="fail",
            reason="duplicate_sample",
            detail={"duplicates": duplicates},
        )
    present = set(counts)
    unexpected = sorted(present - expected)
    if unexpected:
        return GateResult(
            name="sample_completeness",
            passed=False,
            status="fail",
            reason="unexpected_sample",
            detail={"unexpected": unexpected},
        )
    missing = sorted(expected - present)
    if missing:
        return GateResult(
            name="sample_completeness",
            passed=False,
            status="fail",
            reason="failed_or_missing_samples",
            detail={"missing": missing},
        )
    return GateResult(
        name="sample_completeness",
        passed=True,
        status="pass",
        reason="all_samples_ok",
        detail={"expected_fidelity": len(expected)},
    )


def _data_volume_privacy_gates(
    fidelity_records: Sequence[dict[str, Any]],
    expected: set[str],
) -> list[GateResult]:
    """Candidate privacy failures reject; baseline privacy failures are inconclusive."""
    by_key = {data_volume_sample_key(record): record for record in fidelity_records}
    candidate_failures: list[str] = []
    baseline_failures: list[str] = []
    for key in sorted(expected):
        record = by_key.get(key)
        failed = record is None or not record.get("ok") or record.get("privacy_passed") is not True
        if not failed:
            continue
        if "/candidate/" in key:
            candidate_failures.append(key)
        elif "/baseline/" in key:
            baseline_failures.append(key)
    gates: list[GateResult] = []
    if candidate_failures:
        gates.append(
            GateResult(
                name="privacy",
                passed=False,
                status="fail",
                reason="privacy_guard_failed",
                detail={"failed": candidate_failures},
            )
        )
    else:
        gates.append(GateResult(name="privacy", passed=True, status="pass", reason="privacy_passed"))
    if baseline_failures:
        gates.append(
            GateResult(
                name="baseline_privacy",
                passed=False,
                status="inconclusive",
                reason="baseline_privacy",
                detail={"failed": baseline_failures},
            )
        )
    else:
        gates.append(
            GateResult(
                name="baseline_privacy",
                passed=True,
                status="pass",
                reason="baseline_privacy_passed",
            )
        )
    return gates
