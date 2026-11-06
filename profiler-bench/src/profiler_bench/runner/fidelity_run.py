"""Fidelity samples: paired seeds across baseline and candidate profiles."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from profiler_bench.runner.scorer import FidelityScores, Scorer


def paired_seeds(k: int, *, start: int = 0) -> list[int]:
    """Returns K sampler seeds shared by baseline and candidate."""
    return list(range(start, start + k))


def run_fidelity_samples(
    *,
    speed_records: Sequence[dict[str, Any]],
    scorer_baseline: Scorer,
    scorer_candidate: Scorer,
    datasets: Sequence[str],
    policy: dict[str, Any],
    samples_dir: Path,
    dataset_seed: int = 0,
) -> list[dict[str, Any]]:
    """Scores one representative profile per build/dataset at the fidelity tier with paired seeds."""
    tier = str(policy.get("fidelity_tier", "s"))
    k = int(policy.get("fidelity_seeds", 5))
    seeds = paired_seeds(k)
    samples_dir.mkdir(parents=True, exist_ok=True)
    profiles = _representative_profiles(speed_records, datasets, tier)
    records: list[dict[str, Any]] = []
    ceilings: dict[str, float] = {}
    for dataset in datasets:
        if dataset not in profiles.get("baseline", {}) or dataset not in profiles.get("candidate", {}):
            for role in ("baseline", "candidate"):
                record = {
                    "kind": "fidelity",
                    "role": role,
                    "dataset": dataset,
                    "tier": tier,
                    "ok": False,
                    "error": "missing speed profile for fidelity scoring",
                    "seeds": seeds,
                }
                path = samples_dir / f"fidelity-{role}-{dataset}-{tier}-missing.json"
                path.write_text(json.dumps(record, sort_keys=True))
                records.append(record)
            continue
        if dataset not in ceilings:
            ceilings[dataset] = float(scorer_baseline.ceiling(dataset, tier, structure_seed=dataset_seed))
        for seed in seeds:
            for role, scorer in (("baseline", scorer_baseline), ("candidate", scorer_candidate)):
                record = _one_seed(
                    role=role,
                    dataset=dataset,
                    tier=tier,
                    seed=seed,
                    profile=profiles[role][dataset],
                    scorer=scorer,
                    samples_dir=samples_dir,
                    dataset_seed=dataset_seed,
                    ceiling=ceilings[dataset],
                )
                records.append(record)
    return records


def _representative_profiles(
    speed_records: Sequence[dict[str, Any]],
    datasets: Sequence[str],
    tier: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Picks the first successful speed profile per (role, dataset) at `tier`."""
    out: dict[str, dict[str, dict[str, Any]]] = {"baseline": {}, "candidate": {}}
    for record in speed_records:
        if record.get("kind") != "speed" or not record.get("ok"):
            continue
        if record.get("tier") != tier or record.get("dataset") not in datasets:
            continue
        role = str(record["role"])
        dataset = str(record["dataset"])
        if dataset in out[role]:
            continue
        profile = record.get("profile")
        if isinstance(profile, dict):
            out[role][dataset] = profile
    return out


def _one_seed(
    *,
    role: str,
    dataset: str,
    tier: str,
    seed: int,
    profile: dict[str, Any],
    scorer: Scorer,
    samples_dir: Path,
    dataset_seed: int,
    ceiling: float,
) -> dict[str, Any]:
    path = samples_dir / f"fidelity-{role}-{dataset}-{tier}-seed-{seed}.json"
    try:
        scores: FidelityScores = scorer.score(
            profile,
            dataset=dataset,
            tier=tier,
            seed=seed,
            structure_seed=dataset_seed,
            sample_seed=dataset_seed,
        )
        relative = scores.relative_reconstruction
        if relative is None and ceiling > 0:
            relative = min(1.0, scores.reconstruction / ceiling)
        record: dict[str, Any] = {
            "kind": "fidelity",
            "role": role,
            "dataset": dataset,
            "tier": tier,
            "seed": seed,
            "ok": True,
            "reconstruction": scores.reconstruction,
            "structure": scores.structure,
            "privacy_passed": scores.privacy_passed,
            "ceiling": ceiling,
            "relative_reconstruction": relative,
            "detail": scores.detail,
            "metrics_config_hash": scores.metrics_config_hash,
        }
    except Exception as error:
        record = {
            "kind": "fidelity",
            "role": role,
            "dataset": dataset,
            "tier": tier,
            "seed": seed,
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
        }
    path.write_text(json.dumps(record, sort_keys=True, default=str))
    return record
