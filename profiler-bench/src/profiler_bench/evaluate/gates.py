"""Policy gates over paired fidelity and speed differences.

Each gate reports a two-sided percentile confidence interval at the policy confidence
level (default 95%). Intervals are not adjusted for multiplicity: there is no
family-wise or simultaneous coverage guarantee across gates or datasets.

Real datasets (`run["real_datasets"]`) are a holdout suite: their samples are kept out of
every other gate and checked only by the `real_holdout` gate (see `_real_holdout`).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from profiler_bench.evaluate.bootstrap import BootstrapCI, paired_bootstrap_ci, stratified_dataset_mean_ci


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
    run: dict[str, Any],
) -> dict[str, Any]:
    """Applies the frozen policy and returns a verdict with per-gate reasons."""
    proposal_type = str(proposal.get("type", "accuracy"))
    if proposal_type == "data_volume":
        from profiler_bench.evaluate.data_volume import evaluate_data_volume_gates  # noqa: PLC0415

        return evaluate_data_volume_gates(
            fidelity_records=fidelity_records,
            policy=policy,
            proposal=proposal,
            run=run,
        )

    real = set(run.get("real_datasets") or [])
    real_speed = [record for record in speed_records if record.get("dataset") in real]
    real_fidelity = [record for record in fidelity_records if record.get("dataset") in real]
    speed_records = [record for record in speed_records if record.get("dataset") not in real]
    fidelity_records = [record for record in fidelity_records if record.get("dataset") not in real]

    gates: list[GateResult] = []
    expected = expected_sample_keys(run)
    gates.append(_sample_completeness(speed_records, fidelity_records, expected))
    gates.append(_privacy_guard(fidelity_records, expected))
    gates.append(_control_guard(run))

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

    if run.get("suite", "synthetic") != "synthetic":
        gates.append(_real_holdout(real_speed, real_fidelity, policy, run))

    verdict = _combine(gates)
    return {
        "verdict": verdict,
        "proposal_type": proposal_type,
        "gates": [gate.to_dict() for gate in gates],
        "reasons": [gate.reason for gate in gates if gate.status != "pass"],
        "interval_note": (
            "Each gate uses a two-sided percentile CI at the policy confidence level; "
            "there is no family-wise coverage guarantee across gates."
        ),
    }


def expected_sample_keys(run: dict[str, Any]) -> dict[str, set[str]]:
    """Derives the expected speed and fidelity sample keys from `run.json`.

    Speed counts come from the recorded `speed_samples_effective` (reduced for accuracy
    proposals); runs recorded before that field fall back to the policy's `speed_samples`.
    """
    policy = run.get("policy") or {}
    datasets = [str(name) for name in run.get("datasets") or []]
    speed_tiers = [str(tier) for tier in run.get("speed_tiers") or policy.get("speed_tiers") or ["s"]]
    fidelity_tier = str(run.get("fidelity_tier") or policy.get("fidelity_tier") or "s")
    speed_n = run.get("speed_samples_effective") or policy.get("speed_samples") or {}
    fidelity_k = int(policy.get("fidelity_seeds", 5))
    roles = ("baseline", "candidate")
    speed: set[str] = set()
    for dataset in datasets:
        for tier in speed_tiers:
            n = int(speed_n[tier])
            for role in roles:
                for sample_index in range(n):
                    speed.add(f"speed/{role}/{dataset}/{tier}/{sample_index}")
    fidelity: set[str] = set()
    for dataset in datasets:
        for role in roles:
            for seed in range(fidelity_k):
                fidelity.add(f"fidelity/{role}/{dataset}/{fidelity_tier}/seed-{seed}")
    return {"speed": speed, "fidelity": fidelity}


def real_sample_keys(run: dict[str, Any]) -> set[str]:
    """Returns the expected speed and fidelity sample keys of the scored real datasets."""
    policy = run.get("policy") or {}
    fidelity_tier = str(run.get("fidelity_tier") or policy.get("fidelity_tier") or "s")
    speed_n = run.get("speed_samples_effective") or policy.get("speed_samples") or {}
    seeds = range(int(policy.get("fidelity_seeds", 5)))
    return {
        key
        for dataset in run.get("real_datasets") or []
        for role in ("baseline", "candidate")
        for key in (
            *(
                f"speed/{role}/{dataset}/{tier}/{index}"
                for tier in run.get("real_speed_tiers") or [fidelity_tier]
                for index in range(int(speed_n[tier]))
            ),
            *(f"fidelity/{role}/{dataset}/{fidelity_tier}/seed-{seed}" for seed in seeds),
        )
    }


def _real_holdout(
    speed_records: Sequence[dict[str, Any]],
    fidelity_records: Sequence[dict[str, Any]],
    policy: dict[str, Any],
    run: dict[str, Any],
) -> GateResult:
    """Holdout check on the real datasets; it never counts towards acceptance.

    Reports, per real dataset, each build's relative reconstruction and the paired regression with CIs, whether
    privacy passed, the privacy checks left unmeasured (too few independent baseline rows: recorded, neither a pass
    nor a fail), the copy checks that were underpowered (a 5% verbatim copy could not fail them; also unmeasured),
    the copy checks of key-only tables that used the shuffled control, and the holdout's overlap with the profiled
    part per table. Fails (rejects the candidate) when a candidate sample fails privacy. Otherwise it
    is inconclusive when a dataset's regression CI reaches `accuracy.max_dataset_regression`
    (`real_holdout_regression`) or when samples are failed or missing (`real_holdout_incomplete`).
    """
    boot = policy.get("bootstrap", {})
    n_bootstrap, seed = int(boot.get("samples", 2000)), int(boot.get("seed", 0))
    confidence = float(boot.get("confidence", 0.95))
    max_reg = float(policy.get("accuracy", {}).get("max_dataset_regression", 0.01))
    ok = {_sample_key(record): record for record in [*speed_records, *fidelity_records] if record.get("ok")}
    incomplete = sorted(real_sample_keys(run) - set(ok))
    privacy_failed = sorted(
        key for key, record in ok.items() if "/candidate/" in key and record.get("privacy_passed") is False
    )
    paired = _paired_relative(fidelity_records)
    datasets: dict[str, Any] = {}
    regressed = []
    for dataset in run.get("real_datasets") or []:
        relative = {
            role: [
                float(r["relative_reconstruction"])
                for r in fidelity_records
                if r.get("ok") and r.get("dataset") == dataset and r.get("role") == role
            ]
            for role in ("baseline", "candidate")
        }
        detail: dict[str, Any] = {
            f"{role}_relative_reconstruction": paired_bootstrap_ci(
                values, n_bootstrap=n_bootstrap, seed=seed + 5, confidence=confidence
            ).to_dict()
            for role, values in relative.items()
            if values
        }
        detail["candidate_privacy_passed"] = not any(f"/candidate/{dataset}/" in key for key in privacy_failed)
        candidate = [
            r for r in fidelity_records if r.get("ok") and r.get("dataset") == dataset and r.get("role") == "candidate"
        ]
        detail["candidate_privacy_unmeasured"] = sorted(
            {n for r in candidate for n in r.get("privacy_unmeasured") or []}
        )
        checks = [c for r in candidate for c in ((r.get("detail") or {}).get("privacy") or {}).get("checks", [])]
        for label, flag in (("shuffled_control", "shuffled_control"), ("underpowered", "underpowered")):
            detail[f"candidate_privacy_{label}"] = sorted(
                {c["name"] for c in checks if (c.get("detail") or {}).get(flag)}
            )
        overlap = next(
            (r["detail"]["holdout_overlap"] for r in candidate if "holdout_overlap" in (r.get("detail") or {})), None
        )
        if overlap is not None:
            detail["holdout_overlap"] = overlap
        if paired.get(dataset):
            ci = paired_bootstrap_ci(
                [-diff for diff in paired[dataset]], n_bootstrap=n_bootstrap, seed=seed + 1, confidence=confidence
            )
            detail["regression"] = ci.to_dict()
            if ci.high >= max_reg:
                regressed.append(dataset)
        datasets[dataset] = detail
    status: Literal["pass", "fail", "inconclusive"] = "pass"
    reason = "real_holdout_passed" if run.get("real_datasets") else "real_holdout_skipped"
    if privacy_failed:
        status, reason = "fail", "real_holdout_privacy_failed"
    elif regressed:
        status, reason = "inconclusive", "real_holdout_regression"
    elif incomplete:
        status, reason = "inconclusive", "real_holdout_incomplete"
    return GateResult(
        name="real_holdout",
        passed=status == "pass",
        status=status,
        reason=reason,
        detail={
            "datasets": datasets,
            "max_regression": max_reg,
            "regressed": regressed,
            "privacy_failed": privacy_failed,
            "incomplete": incomplete,
            "skipped": run.get("real_skipped") or {},
            "split": run.get("real_split"),
            "structure": "not_applicable",
            "note": "holdout check: excluded from the mean and per-dataset accuracy gates",
        },
    )


def _sample_completeness(
    speed_records: Sequence[dict[str, Any]],
    fidelity_records: Sequence[dict[str, Any]],
    expected: dict[str, set[str]],
) -> GateResult:
    """Checks raw records first, then duplicates, unexpected cells, and missing cells."""
    raw = [*speed_records, *fidelity_records]
    # Any failed record anywhere blocks acceptance, including cells outside the matrix and
    # failures that a later duplicate success would otherwise overwrite if keyed first.
    failed = [_sample_key(record) for record in raw if not record.get("ok")]
    if failed:
        return GateResult(
            name="sample_completeness",
            passed=False,
            status="fail",
            reason="failed_or_missing_samples",
            detail={"failed": failed},
        )

    expected_keys = expected["speed"] | expected["fidelity"]
    counts: dict[str, int] = defaultdict(int)
    for record in raw:
        counts[_sample_key(record)] += 1

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
    unexpected = sorted(present - expected_keys)
    if unexpected:
        return GateResult(
            name="sample_completeness",
            passed=False,
            status="fail",
            reason="unexpected_sample",
            detail={"unexpected": unexpected},
        )

    missing = sorted(expected_keys - present)
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
        detail={"expected_speed": len(expected["speed"]), "expected_fidelity": len(expected["fidelity"])},
    )


def _privacy_guard(fidelity_records: Sequence[dict[str, Any]], expected: dict[str, set[str]]) -> GateResult:
    """Hard privacy guard: every expected candidate fidelity cell must explicitly pass."""
    by_key = {_sample_key(record): record for record in fidelity_records}
    failures: list[str] = []
    for key in sorted(expected["fidelity"]):
        if "/candidate/" not in key:
            continue
        record = by_key.get(key)
        if record is None or not record.get("ok") or record.get("privacy_passed") is not True:
            failures.append(key)
    if failures:
        return GateResult(
            name="privacy",
            passed=False,
            status="fail",
            reason="privacy_guard_failed",
            detail={"failed": failures},
        )
    return GateResult(name="privacy", passed=True, status="pass", reason="privacy_passed")


def _control_guard(run: dict[str, Any]) -> GateResult:
    """Control runs and identical profiler builds can never be accepted."""
    baseline = run.get("baseline") or {}
    candidate = run.get("candidate") or {}
    same_hash = baseline.get("profiler_src_hash") is not None and baseline.get("profiler_src_hash") == candidate.get(
        "profiler_src_hash"
    )
    if run.get("control") or same_hash:
        return GateResult(
            name="control",
            passed=False,
            status="inconclusive",
            reason="control",
            detail={
                "control": bool(run.get("control")),
                "same_profiler_src_hash": bool(same_hash),
                "baseline_hash": baseline.get("profiler_src_hash"),
                "candidate_hash": candidate.get("profiler_src_hash"),
            },
        )
    return GateResult(name="control", passed=True, status="pass", reason="distinct_builds")


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
    per_dataset_means = {dataset: float(sum(diffs) / len(diffs)) for dataset, diffs in paired.items()}
    mean_ci = stratified_dataset_mean_ci(
        paired,
        n_bootstrap=int(boot.get("samples", 2000)),
        seed=int(boot.get("seed", 0)),
        confidence=float(boot.get("confidence", 0.95)),
    )
    min_improvement = float(cfg.get("min_mean_improvement", 0.005))
    ci_floor = float(cfg.get("ci_lower_must_exceed", 0.0))
    gates = [_improvement_gate("accuracy_mean", mean_ci, min_improvement, ci_floor)]
    gates[0].detail["estimand"] = (
        "equally weighted mean across datasets of per-dataset mean paired differences; "
        "stratified bootstrap resamples seeds within each dataset (datasets are fixed strata). "
        "A hierarchical bootstrap that also resamples datasets is an unused alternative."
    )
    gates[0].detail["per_dataset_mean_improvement"] = per_dataset_means
    gates[0].detail["mean_across_datasets"] = float(sum(per_dataset_means.values()) / len(per_dataset_means))

    max_reg = float(cfg.get("max_dataset_regression", 0.01))
    reg_details = {}
    reg_status: Literal["pass", "fail", "inconclusive"] = "pass"
    reg_reason = "no_reconstruction_regression"
    for dataset, diffs in paired.items():
        regressions = [-diff for diff in diffs]
        ci = paired_bootstrap_ci(
            regressions,
            statistic="mean",
            n_bootstrap=int(boot.get("samples", 2000)),
            seed=int(boot.get("seed", 0)) + 1,
            confidence=float(boot.get("confidence", 0.95)),
        )
        reg_details[dataset] = ci.to_dict()
        if ci.high >= max_reg:
            if ci.low < max_reg:
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
    # RSS bound is max(frac * baseline, abs_bytes): compare (cand-base) / max(base, abs/frac) to frac.
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
            if name == "rss":
                case_detail["rss_bound"] = "max(frac * baseline_rss, abs_mib)"
            if ci.high > limit:
                if ci.low < limit:
                    status = "inconclusive" if status != "fail" else status
                    reason = f"{name}_regression_inconclusive"
                else:
                    status = "fail"
                    reason = f"{name}_regression"
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
