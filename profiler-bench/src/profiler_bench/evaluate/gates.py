"""Policy gates over paired fidelity and speed differences."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Literal

from profiler_bench.evaluate.bootstrap import BootstrapCI, paired_bootstrap_ci


Verdict = Literal["accepted", "rejected", "inconclusive"]


@dataclass
class GateResult:
    """Outcome of one named gate.

    Attributes:
        name: Gate name.
        passed: Whether the gate's accept condition held.
        status: `pass`, `fail`, or `inconclusive`.
        reason: Machine-readable reason code.
        detail: Supporting numbers and intervals.
    """

    name: str
    passed: bool
    status: Literal["pass", "fail", "inconclusive"]
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return {
            "name": self.name,
            "passed": self.passed,
            "status": self.status,
            "reason": self.reason,
            "detail": self.detail,
        }


def evaluate_gates(
    *,
    speed_records: Sequence[dict[str, Any]],
    fidelity_records: Sequence[dict[str, Any]],
    policy: dict[str, Any],
    proposal: dict[str, Any],
) -> dict[str, Any]:
    """Applies the frozen policy and returns a verdict with per-gate reasons."""
    gates: list[GateResult] = []
    missing = _missing_or_failed(speed_records, fidelity_records)
    gates.append(missing)

    privacy = _privacy_guard(fidelity_records, policy)
    gates.append(privacy)

    proposal_type = str(proposal.get("type", "accuracy"))
    if proposal_type == "accuracy":
        gates.extend(_accuracy_gates(fidelity_records, policy))
    elif proposal_type == "speed":
        gates.extend(_accuracy_regression_gates(fidelity_records, policy))
        gates.extend(_speed_gates(speed_records, policy, proposal))
    else:
        gates.append(
            GateResult(
                name="proposal_type",
                passed=False,
                status="fail",
                reason="unknown_proposal_type",
                detail={"type": proposal_type},
            )
        )

    verdict = _combine(gates)
    return {
        "verdict": verdict,
        "proposal_type": proposal_type,
        "gates": [gate.to_dict() for gate in gates],
        "reasons": [gate.reason for gate in gates if gate.status != "pass"],
    }


def _missing_or_failed(
    speed_records: Sequence[dict[str, Any]],
    fidelity_records: Sequence[dict[str, Any]],
) -> GateResult:
    failed = [_sample_key(record) for record in [*speed_records, *fidelity_records] if not record.get("ok")]
    if failed:
        return GateResult(
            name="sample_completeness",
            passed=False,
            status="fail",
            reason="failed_or_missing_samples",
            detail={"failed": failed},
        )
    return GateResult(
        name="sample_completeness",
        passed=True,
        status="pass",
        reason="all_samples_ok",
    )


def _privacy_guard(fidelity_records: Sequence[dict[str, Any]], policy: dict[str, Any]) -> GateResult:
    if not policy.get("privacy", {}).get("require_all_pass", True):
        return GateResult(name="privacy", passed=True, status="pass", reason="privacy_disabled")
    failures = [
        _sample_key(record)
        for record in fidelity_records
        if record.get("role") == "candidate" and record.get("ok") and record.get("privacy_passed") is False
    ]
    if failures:
        return GateResult(
            name="privacy",
            passed=False,
            status="fail",
            reason="privacy_guard_failed",
            detail={"failed": failures},
        )
    return GateResult(name="privacy", passed=True, status="pass", reason="privacy_passed")


def _accuracy_gates(fidelity_records: Sequence[dict[str, Any]], policy: dict[str, Any]) -> list[GateResult]:
    cfg = policy.get("accuracy", {})
    boot = policy.get("bootstrap", {})
    paired = _paired_relative(fidelity_records)
    if not paired:
        return [
            GateResult(
                name="accuracy_mean",
                passed=False,
                status="fail",
                reason="no_fidelity_pairs",
            )
        ]
    # Mean improvement across datasets: average per-dataset mean seed improvement.
    per_dataset_means = {dataset: float(sum(diffs) / len(diffs)) for dataset, diffs in paired.items()}
    dataset_improvements = list(per_dataset_means.values())
    # Bootstrap over all paired seed differences pooled, and also check per-dataset.
    all_diffs = [diff for diffs in paired.values() for diff in diffs]
    mean_ci = paired_bootstrap_ci(
        all_diffs,
        statistic="mean",
        n_bootstrap=int(boot.get("samples", 2000)),
        seed=int(boot.get("seed", 0)),
        confidence=float(boot.get("confidence", 0.95)),
    )
    min_improvement = float(cfg.get("min_mean_improvement", 0.005))
    ci_floor = float(cfg.get("ci_lower_must_exceed", 0.0))
    gates = [_improvement_gate("accuracy_mean", mean_ci, min_improvement, ci_floor)]

    max_reg = float(cfg.get("max_dataset_regression", 0.01))
    reg_details = {}
    reg_status: Literal["pass", "fail", "inconclusive"] = "pass"
    reg_reason = "no_reconstruction_regression"
    for dataset, diffs in paired.items():
        # Regression of reconstruction: baseline - candidate = -improvement.
        regressions = [-diff for diff in diffs]
        ci = paired_bootstrap_ci(
            regressions,
            statistic="mean",
            n_bootstrap=int(boot.get("samples", 2000)),
            seed=int(boot.get("seed", 0)) + 1,
            confidence=float(boot.get("confidence", 0.95)),
        )
        reg_details[dataset] = ci.to_dict()
        # Acceptable if CI upper bound of regression < max_reg.
        if ci.high >= max_reg:
            if ci.low < max_reg < ci.high or (ci.low < max_reg and ci.point < max_reg < ci.high):
                reg_status = "inconclusive" if reg_status != "fail" else reg_status
                reg_reason = "reconstruction_regression_inconclusive"
            else:
                reg_status = "fail"
                reg_reason = "reconstruction_regression"
    gates.append(
        GateResult(
            name="accuracy_per_dataset",
            passed=reg_status == "pass",
            status=reg_status,
            reason=reg_reason,
            detail={"datasets": reg_details, "max_regression": max_reg},
        )
    )

    structure_paired = _paired_structure(fidelity_records)
    max_struct = float(cfg.get("max_structure_regression", 0.02))
    struct_details = {}
    struct_status: Literal["pass", "fail", "inconclusive"] = "pass"
    struct_reason = "no_structure_regression"
    for dataset, diffs in structure_paired.items():
        regressions = [-diff for diff in diffs]
        ci = paired_bootstrap_ci(
            regressions,
            statistic="mean",
            n_bootstrap=int(boot.get("samples", 2000)),
            seed=int(boot.get("seed", 0)) + 2,
            confidence=float(boot.get("confidence", 0.95)),
        )
        struct_details[dataset] = ci.to_dict()
        if ci.high >= max_struct:
            if ci.low < max_struct:
                struct_status = "inconclusive" if struct_status != "fail" else struct_status
                struct_reason = "structure_regression_inconclusive"
            else:
                struct_status = "fail"
                struct_reason = "structure_regression"
    gates.append(
        GateResult(
            name="structure",
            passed=struct_status == "pass",
            status=struct_status,
            reason=struct_reason,
            detail={"datasets": struct_details, "max_regression": max_struct},
        )
    )
    gates[0].detail["per_dataset_mean_improvement"] = per_dataset_means
    gates[0].detail["mean_across_datasets"] = (
        float(sum(dataset_improvements) / len(dataset_improvements)) if dataset_improvements else 0.0
    )
    return gates


def _accuracy_regression_gates(fidelity_records: Sequence[dict[str, Any]], policy: dict[str, Any]) -> list[GateResult]:
    """Speed proposals still require reconstruction not to regress."""
    return [gate for gate in _accuracy_gates(fidelity_records, policy) if gate.name != "accuracy_mean"]


def _speed_gates(
    speed_records: Sequence[dict[str, Any]],
    policy: dict[str, Any],
    proposal: dict[str, Any],
) -> list[GateResult]:
    cfg = policy.get("speed", {})
    boot = policy.get("bootstrap", {})
    targets = proposal.get("targets") or []
    if isinstance(targets, dict):
        target_keys = [str(item) for item in targets.get("speed", targets.get("datasets", []))]
    else:
        target_keys = [str(item) for item in targets]

    paired_wall = _paired_metric(speed_records, "wall_s")
    paired_cpu = _paired_metric(speed_records, "cpu_s")
    paired_rss = _paired_metric(speed_records, "peak_rss_bytes")
    if not paired_wall:
        return [GateResult(name="speed_target", passed=False, status="fail", reason="no_speed_pairs")]

    min_imp = float(cfg.get("min_target_improvement", 0.05))
    target_hits = []
    target_detail = {}
    for key, pairs in paired_wall.items():
        case_id = f"{key[0]}@{key[1]}"
        if target_keys and case_id not in target_keys and key[0] not in target_keys:
            continue
        improvements = [(base - cand) / base for base, cand in pairs if base > 0]
        if not improvements:
            continue
        ci = paired_bootstrap_ci(
            improvements,
            statistic="median",
            n_bootstrap=int(boot.get("samples", 2000)),
            seed=int(boot.get("seed", 0)) + 3,
            confidence=float(boot.get("confidence", 0.95)),
        )
        target_detail[case_id] = ci.to_dict()
        if ci.low >= min_imp:
            target_hits.append(case_id)
        elif ci.high > min_imp > ci.low:
            target_hits.append(f"{case_id}:inconclusive")

    if any(not item.endswith(":inconclusive") for item in target_hits):
        target_gate = GateResult(
            name="speed_target",
            passed=True,
            status="pass",
            reason="target_improved",
            detail={"targets": target_detail, "hits": target_hits},
        )
    elif any(item.endswith(":inconclusive") for item in target_hits):
        target_gate = GateResult(
            name="speed_target",
            passed=False,
            status="inconclusive",
            reason="target_improvement_inconclusive",
            detail={"targets": target_detail, "hits": target_hits},
        )
    else:
        target_gate = GateResult(
            name="speed_target",
            passed=False,
            status="fail",
            reason="no_target_improvement",
            detail={"targets": target_detail, "required": min_imp},
        )

    max_wall = float(cfg.get("max_wall_regression", 0.10))
    max_cpu = float(cfg.get("max_cpu_regression", 0.10))
    max_rss_frac = float(cfg.get("max_rss_regression_frac", 0.05))
    max_rss_abs = float(cfg.get("max_rss_regression_abs_mib", 64.0)) * 1024 * 1024

    guard = _regression_guard(
        paired_wall,
        paired_cpu,
        paired_rss,
        max_wall=max_wall,
        max_cpu=max_cpu,
        max_rss_frac=max_rss_frac,
        max_rss_abs=max_rss_abs,
        boot=boot,
    )
    return [target_gate, guard]


def _regression_guard(
    paired_wall: dict[tuple[str, str], list[tuple[float, float]]],
    paired_cpu: dict[tuple[str, str], list[tuple[float, float]]],
    paired_rss: dict[tuple[str, str], list[tuple[float, float]]],
    *,
    max_wall: float,
    max_cpu: float,
    max_rss_frac: float,
    max_rss_abs: float,
    boot: dict[str, Any],
) -> GateResult:
    detail: dict[str, Any] = {}
    status: Literal["pass", "fail", "inconclusive"] = "pass"
    reason = "no_speed_regression"
    for key, pairs in paired_wall.items():
        case_id = f"{key[0]}@{key[1]}"
        wall_reg = [(cand - base) / base for base, cand in pairs if base > 0]
        cpu_pairs = paired_cpu.get(key, [])
        cpu_reg = [(cand - base) / base for base, cand in cpu_pairs if base > 0]
        rss_pairs = paired_rss.get(key, [])
        rss_reg = []
        for base, cand in rss_pairs:
            if base <= 0:
                continue
            rss_reg.append((cand - base) / max(base, max_rss_abs / max_rss_frac))
        case_detail: dict[str, Any] = {}
        for name, regs, limit in (
            ("wall", wall_reg, max_wall),
            ("cpu", cpu_reg, max_cpu),
            ("rss", rss_reg, max_rss_frac),
        ):
            if not regs:
                continue
            ci = paired_bootstrap_ci(
                regs,
                statistic="median",
                n_bootstrap=int(boot.get("samples", 2000)),
                seed=int(boot.get("seed", 0)) + 4,
                confidence=float(boot.get("confidence", 0.95)),
            )
            case_detail[name] = ci.to_dict()
            if ci.high > limit:
                if ci.low < limit:
                    status = "inconclusive" if status != "fail" else status
                    reason = f"{name}_regression_inconclusive"
                else:
                    status = "fail"
                    reason = f"{name}_regression"
        if rss_pairs:
            abs_deltas = [cand - base for base, cand in rss_pairs]
            if median(abs_deltas) > max_rss_abs and status != "fail":
                status = "fail"
                reason = "rss_absolute_regression"
                case_detail["rss_abs_median"] = float(median(abs_deltas))
        detail[case_id] = case_detail
    return GateResult(
        name="speed_guards",
        passed=status == "pass",
        status=status,
        reason=reason,
        detail=detail,
    )


def _improvement_gate(name: str, ci: BootstrapCI, min_improvement: float, ci_floor: float) -> GateResult:
    detail = {"ci": ci.to_dict(), "min_improvement": min_improvement, "ci_floor": ci_floor}
    # Accept when mean improvement >= threshold and the CI lower bound clears the floor.
    if ci.point >= min_improvement and ci.low > ci_floor:
        return GateResult(name=name, passed=True, status="pass", reason="accuracy_improved", detail=detail)
    if ci.crosses_zero() or ci.low <= ci_floor < ci.high:
        return GateResult(
            name=name, passed=False, status="inconclusive", reason="accuracy_crossing_interval", detail=detail
        )
    return GateResult(name=name, passed=False, status="fail", reason="accuracy_insufficient", detail=detail)


def _combine(gates: Sequence[GateResult]) -> Verdict:
    if any(gate.status == "fail" for gate in gates):
        return "rejected"
    if any(gate.status == "inconclusive" for gate in gates):
        return "inconclusive"
    if all(gate.passed for gate in gates):
        return "accepted"
    return "inconclusive"


def _sample_key(record: dict[str, Any]) -> str:
    parts = [str(record.get("kind")), str(record.get("role")), str(record.get("dataset")), str(record.get("tier"))]
    if "sample_index" in record:
        parts.append(str(record["sample_index"]))
    if "seed" in record:
        parts.append(f"seed-{record['seed']}")
    return "/".join(parts)


def _paired_relative(records: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
    by: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for record in records:
        if record.get("kind") != "fidelity" or not record.get("ok"):
            continue
        if record.get("relative_reconstruction") is None:
            continue
        key = (str(record["dataset"]), int(record["seed"]))
        by[key][str(record["role"])] = float(record["relative_reconstruction"])
    paired: dict[str, list[float]] = defaultdict(list)
    for (dataset, _seed), roles in sorted(by.items()):
        if "baseline" in roles and "candidate" in roles:
            paired[dataset].append(roles["candidate"] - roles["baseline"])
    return dict(paired)


def _paired_structure(records: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
    by: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for record in records:
        if record.get("kind") != "fidelity" or not record.get("ok"):
            continue
        key = (str(record["dataset"]), int(record["seed"]))
        by[key][str(record["role"])] = float(record["structure"])
    paired: dict[str, list[float]] = defaultdict(list)
    for (dataset, _seed), roles in sorted(by.items()):
        if "baseline" in roles and "candidate" in roles:
            paired[dataset].append(roles["candidate"] - roles["baseline"])
    return dict(paired)


def _paired_metric(records: Sequence[dict[str, Any]], metric: str) -> dict[tuple[str, str], list[tuple[float, float]]]:
    by: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    for record in records:
        if record.get("kind") != "speed" or not record.get("ok"):
            continue
        if record.get(metric) is None:
            continue
        key = (str(record["dataset"]), str(record["tier"]), int(record["sample_index"]))
        by[key][str(record["role"])] = float(record[metric])
    paired: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for (dataset, tier, _idx), roles in sorted(by.items()):
        if "baseline" in roles and "candidate" in roles:
            paired[(dataset, tier)].append((roles["baseline"], roles["candidate"]))
    return dict(paired)
