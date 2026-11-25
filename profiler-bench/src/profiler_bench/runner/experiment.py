"""Orchestrates an RSI experiment: identities, speed, fidelity, and artifacts."""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from profiler_bench.catalog import DATASETS, REAL_DATASETS, Unavailable
from profiler_bench.catalog.real import PARTS, SPLIT, SPLIT_REPORT, materialize_part, split_dir
from profiler_bench.evaluate.evaluate import evaluate_experiment
from profiler_bench.runner.baseline_cache import config_hash, dataset_identity, environment, load_entry, store_entry
from profiler_bench.runner.fidelity_run import DEFAULT_FIDELITY_CELL_TIMEOUT_S, paired_seeds, run_fidelity_samples
from profiler_bench.runner.identity import (
    BuildIdentity,
    HarnessIdentity,
    identify_build,
    identify_harness,
    machine_info,
)
from profiler_bench.runner.measure import check_profile_determinism, run_speed_samples
from profiler_bench.runner.policy import effective_speed_samples, load_policy
from profiler_bench.runner.scorer import FakeScorer, Scorer, try_real_scorer
from profiler_bench.runner.spark_conf import (
    DEFAULT_FIDELITY_DRIVER_MEMORY,
    EXPERIMENT_FIDELITY_MASTER,
    fidelity_spark_config,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Inputs that define one baseline-vs-candidate experiment.

    Attributes:
        experiment_id: Directory name under `profiler/rsi/experiments/`.
        baseline_worktree: Path containing the baseline `profiler/src`.
        candidate_worktree: Path containing the candidate `profiler/src`.
        policy_path: Frozen policy JSON.
        datasets: Catalog datasets to measure; defaults to every registered dataset.
        control: When true, the candidate path is ignored and the baseline is compared to itself.
        scorer: Injected scorer; defaults to the real one when importable.
        repo_root: Repository root that owns `profiler/rsi/experiments`.
        cache_dir: Optional catalog cache directory for workers.
        proposal: Optional proposal metadata written as `proposal.json`.
        speed_tiers: Override for speed tiers from the policy.
        fidelity_driver_memory: Driver heap for fidelity workers only (speed policy spark unchanged).
        fidelity_cell_timeout_s: Wall-clock timeout for one (build, dataset) fidelity worker.
        baseline_cache: Reuse and store baseline-build samples across experiments.
        baseline_cache_dir: Baseline cache location; defaults to `profiler/rsi/experiments/.cache/baseline`.
        suite: `synthetic` (the catalog generators), `real` (the real datasets, a holdout check), or `all`.
    """

    experiment_id: str
    baseline_worktree: Path
    candidate_worktree: Path
    policy_path: Path
    datasets: Sequence[str] | None = None
    control: bool = False
    scorer: Scorer | None = None
    repo_root: Path | None = None
    cache_dir: Path | None = None
    proposal: dict[str, Any] | None = None
    speed_tiers: Sequence[str] | None = None
    command_line: Sequence[str] | None = None
    fidelity_driver_memory: str = DEFAULT_FIDELITY_DRIVER_MEMORY
    fidelity_cell_timeout_s: float = DEFAULT_FIDELITY_CELL_TIMEOUT_S
    baseline_cache: bool = True
    baseline_cache_dir: Path | None = None
    suite: str = "synthetic"


SUITES = ("synthetic", "real", "all")


def suite_datasets(suite: str, datasets: Sequence[str] | None) -> tuple[list[str], list[str]]:
    """Returns the `(synthetic, real)` datasets of a suite, restricted to `datasets` when given.

    Raises:
        ValueError: If the suite is unknown or `datasets` names a dataset outside it.
    """
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; expected one of {list(SUITES)}")
    synthetic = sorted(DATASETS) if suite != "real" else []
    real = sorted(REAL_DATASETS) if suite != "synthetic" else []
    if datasets is None:
        return synthetic, real
    outside = [name for name in datasets if name not in synthetic and name not in real]
    if outside:
        raise ValueError(f"datasets {outside} are not in the {suite} suite")
    return [n for n in datasets if n in synthetic], [n for n in datasets if n in real]


def prepare_real(
    names: Sequence[str], tier: str, root: Path | None
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """Materializes every benchmark part of each real dataset and returns `(ready, skipped, split_reports)`.

    A dataset is skipped, with the reason, when its export is neither cached nor exportable (no warehouse
    configured, rejected credentials, offline), fails its pinned checks, or has no tier `tier`. The split report
    of each ready dataset gives its full-tier rows, shares, and copies per table and part (see `real.split_report`).
    """
    ready, skipped, reports = [], {}, {}
    for name in names:
        try:
            for part in PARTS:
                materialize_part(REAL_DATASETS[name], tier, part, root)
        except (Unavailable, ValueError, KeyError) as reason:
            skipped[name] = f"{type(reason).__name__}: {reason}"
        else:
            ready.append(name)
            report = split_dir(REAL_DATASETS[name], root) / SPLIT_REPORT
            reports[name] = json.loads(report.read_text()) if report.is_file() else None
    return ready, skipped, reports


def experiment_dir(experiment_id: str, repo_root: Path | None = None) -> Path:
    """Returns `profiler/rsi/experiments/<experiment_id>` under the repo root."""
    root = repo_root or Path(__file__).resolve().parents[4]
    return root / "profiler" / "rsi" / "experiments" / experiment_id


def _recorded_path(path: str | Path, repo_root: Path) -> str:
    """Returns a portable path for persisted experiment metadata."""
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(repo_root).as_posix()
        return "<repo>" if relative == "." else f"<repo>/{relative}"
    except ValueError:
        return f"<external>/{resolved.name}"


def _recorded_identity(identity: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Returns an identity with portable filesystem paths."""
    recorded = dict(identity)
    for field in ("worktree", "harness_root"):
        if field in recorded:
            recorded[field] = _recorded_path(recorded[field], repo_root)
    if "hashed_roots" in recorded:
        recorded["hashed_roots"] = [_recorded_path(path, repo_root) for path in recorded["hashed_roots"]]
    return recorded


def _recorded_command(command: Sequence[str], repo_root: Path) -> list[str]:
    """Returns command arguments with absolute paths made portable."""
    return [_recorded_path(arg, repo_root) if Path(arg).is_absolute() else arg for arg in command]


def run_experiment(config: ExperimentConfig) -> Path:
    """Runs the experiment and returns the experiment directory path."""
    policy = load_policy(config.policy_path)
    repo_root = (config.repo_root or Path(__file__).resolve().parents[4]).resolve()
    out = experiment_dir(config.experiment_id, repo_root)
    if out.exists():
        shutil.rmtree(out)
    samples_dir = out / "samples"
    samples_dir.mkdir(parents=True)

    baseline = identify_build("baseline", config.baseline_worktree)
    candidate_path = config.baseline_worktree if config.control else config.candidate_worktree
    candidate = identify_build("candidate", candidate_path)
    harness = identify_harness(policy=policy)
    recording_root = Path(harness.harness_root)

    datasets, real_requested = suite_datasets(config.suite, config.datasets)
    speed_tiers = (
        list(config.speed_tiers) if config.speed_tiers is not None else list(policy.get("speed_tiers", ["s", "m"]))
    )
    fidelity_tier = str(policy.get("fidelity_tier", "s"))
    if fidelity_tier not in speed_tiers:
        speed_tiers = [*speed_tiers, fidelity_tier]

    proposal = config.proposal or {"type": "accuracy", "targets": datasets}
    (out / "proposal.json").write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n")
    proposal_type = str(proposal.get("type", "accuracy"))
    all_speed_n = effective_speed_samples(policy, proposal_type)
    speed_n = {tier: all_speed_n[tier] for tier in speed_tiers}
    if speed_n[fidelity_tier] < 1:
        raise ValueError(f"fidelity tier {fidelity_tier!r} needs at least one speed sample for its profile")
    dataset_seed = int(policy.get("dataset_seed", 0))
    fidelity_parallelism = int(policy.get("fidelity_parallelism", 1))
    real_root = config.cache_dir
    real_datasets, real_skipped, real_reports = prepare_real(real_requested, fidelity_tier, real_root)
    real_speed_n = {fidelity_tier: speed_n[fidelity_tier]}

    fidelity_spark = fidelity_spark_config(
        driver_memory=config.fidelity_driver_memory,
        master=EXPERIMENT_FIDELITY_MASTER,
    )
    scorer = config.scorer or try_real_scorer(
        cache_dir=out / "cache",
        harness_hash=harness.harness_hash,
        metrics_config_hash=harness.metrics_config_hash,
        real_root=real_root,
    )
    if scorer is None:
        raise RuntimeError("no Scorer available: inject ExperimentConfig.scorer or merge sampler/fidelity modules")
    scorer_baseline, scorer_candidate = _role_scorers(scorer)

    cache_root = (config.baseline_cache_dir or out.parent / ".cache" / "baseline") if config.baseline_cache else None
    speed_keys, fidelity_keys = (
        _merged_keys(
            [
                _cache_keys(
                    baseline=baseline,
                    harness=harness,
                    policy=policy,
                    datasets=names,
                    speed_n=tiers,
                    fidelity_tier=fidelity_tier,
                    fidelity_spark=fidelity_spark,
                    scorer=scorer_baseline,
                    cache_speed=proposal_type == "accuracy",
                    catalog_root=config.cache_dir,
                )
                for names, tiers in ((datasets, speed_n), (real_datasets, real_speed_n))
            ]
        )
        if cache_root is not None
        else ({}, {})
    )
    speed_hits = _cache_hits(cache_root, speed_keys)
    fidelity_hits = _cache_hits(cache_root, fidelity_keys)
    hits = [
        {"case": f"{kind}/{dataset}/{tier}", "source": entry["source_experiment_id"]}
        for kind, found in (("speed", speed_hits), ("fidelity", fidelity_hits))
        for (dataset, tier), entry in found.items()
    ]

    real_meta = {
        "real_datasets": real_datasets,
        "real_skipped": real_skipped,
        "real_split": SPLIT,
        "real_split_report": real_reports,
        "real_speed_tiers": [fidelity_tier],
    }
    run_meta = {
        "suite": config.suite,
        **(real_meta if config.suite != "synthetic" else {}),
        "experiment_id": config.experiment_id,
        "control": config.control,
        "datasets": datasets,
        "speed_tiers": speed_tiers,
        "speed_samples_effective": speed_n,
        "fidelity_tier": fidelity_tier,
        "fidelity_spark": fidelity_spark,
        "fidelity_parallelism": fidelity_parallelism,
        "fidelity_cell_timeout_s": float(config.fidelity_cell_timeout_s),
        "fidelity_model_source": "harness",
        "baseline_cache_enabled": cache_root is not None,
        "baseline_cache_parts": {
            "speed_cache_allowed": proposal_type == "accuracy",
            "speed_cached": bool(speed_hits),
            "fidelity_cached": bool(fidelity_hits),
        },
        "baseline_cached": bool(hits),
        "baseline_cache_sources": sorted({hit["source"] for hit in hits}),
        "baseline_cache_hits": hits,
        "baseline": _recorded_identity(baseline.to_dict(), recording_root),
        "candidate": _recorded_identity(candidate.to_dict(), recording_root),
        "harness": _recorded_identity(harness.to_dict(), recording_root),
        "policy": policy,
        "policy_path": _recorded_path(config.policy_path, recording_root),
        "command_line": _recorded_command(config.command_line or sys.argv, recording_root),
        "machine": machine_info(),
        "not_measured": policy.get("not_measured", []),
    }
    (out / "run.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True, default=str) + "\n")

    speed_records = [
        record
        for names, tiers in ((datasets, speed_n), (real_datasets, real_speed_n))
        for record in run_speed_samples(
            baseline=baseline,
            candidate=candidate,
            datasets=names,
            tiers=list(tiers),
            policy=policy,
            samples_dir=samples_dir,
            dataset_seed=dataset_seed,
            cache_dir=config.cache_dir,
            speed_n=tiers,
            cached_baseline={case: entry["records"] for case, entry in speed_hits.items()},
        )
    ]
    determinism = check_profile_determinism(speed_records)
    (samples_dir / "profile_determinism.json").write_text(json.dumps(determinism, indent=2, sort_keys=True) + "\n")
    _store_misses(cache_root, speed_keys, speed_hits, speed_records, config.experiment_id)

    fidelity_records = run_fidelity_samples(
        speed_records=speed_records,
        scorer_baseline=scorer_baseline,
        scorer_candidate=scorer_candidate,
        datasets=[*datasets, *real_datasets],
        policy=policy,
        samples_dir=samples_dir,
        dataset_seed=dataset_seed,
        fidelity_spark=fidelity_spark,
        cache_dir=out / "cache",
        harness_hash=harness.harness_hash,
        metrics_config_hash=harness.metrics_config_hash,
        cell_timeout_s=float(config.fidelity_cell_timeout_s),
        parallelism=fidelity_parallelism,
        cached_baseline={dataset: entry["records"] for (dataset, _), entry in fidelity_hits.items()},
        real_root=real_root,
    )
    _store_misses(cache_root, fidelity_keys, fidelity_hits, fidelity_records, config.experiment_id, speed_records)

    evaluate_experiment(out)
    return out


def _cache_keys(
    *,
    baseline: BuildIdentity,
    harness: HarnessIdentity,
    policy: dict[str, Any],
    datasets: Sequence[str],
    speed_n: dict[str, int],
    fidelity_tier: str,
    fidelity_spark: dict[str, Any],
    scorer: Scorer,
    cache_speed: bool,
    catalog_root: Path | None,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Returns baseline cache key components for every cacheable speed and fidelity case `(dataset, tier)`.

    Speed cases are cacheable only when `cache_speed` is set (accuracy proposals); other proposal
    types time both builds fresh in the same run.
    """
    dataset_seed = int(policy.get("dataset_seed", 0))
    common = {
        "baseline_profiler_src_hash": baseline.profiler_src_hash,
        "harness_hash": harness.harness_hash,
        "policy_hash": _sampling_policy_hash(policy),
        "metrics_config_hash": harness.metrics_config_hash,
        "dataset_seed": dataset_seed,
    }
    speed = (
        {
            (dataset, tier): {
                **common,
                **dataset_identity(dataset, tier, dataset_seed, catalog_root),
                "kind": "speed",
                "dataset": dataset,
                "tier": tier,
                "samples": n,
                "spark_hash": config_hash(policy.get("spark", {})),
                "machine": machine_info(),
                "environment": environment(java=True),
            }
            for dataset in datasets
            for tier, n in speed_n.items()
            if n > 0
        }
        if cache_speed
        else {}
    )
    fidelity = {
        (dataset, fidelity_tier): {
            **common,
            **dataset_identity(dataset, fidelity_tier, dataset_seed, catalog_root),
            "kind": "fidelity",
            "dataset": dataset,
            "tier": fidelity_tier,
            "sampler_seeds": paired_seeds(int(policy.get("fidelity_seeds", 5))),
            "spark_hash": config_hash(fidelity_spark),
            "scorer_hash": config_hash(_scorer_fingerprint(scorer)),
            "environment": environment(java=False),
        }
        for dataset in datasets
    }
    return speed, fidelity


def _merged_keys(
    parts: Sequence[tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Returns the union of several `(speed, fidelity)` cache key sets."""
    return {k: v for speed, _ in parts for k, v in speed.items()}, {k: v for _, fid in parts for k, v in fid.items()}


def _sampling_policy_hash(policy: dict[str, Any]) -> str:
    """Hashes only policy inputs that change which baseline samples are produced."""
    sampling_keys = (
        "speed_samples",
        "speed_samples_accuracy",
        "fidelity_seeds",
        "fidelity_tier",
        "speed_tiers",
        "dataset_seed",
        "spark",
    )
    return config_hash({key: policy.get(key) for key in sampling_keys})


def _role_scorers(scorer: Scorer) -> tuple[Scorer, Scorer]:
    """Returns `(baseline, candidate)` scorers; fake scorers are bound to their role."""
    if isinstance(scorer, FakeScorer):
        return scorer.with_role("baseline"), scorer.with_role("candidate")
    return scorer, scorer


def _scorer_fingerprint(scorer: Scorer) -> dict[str, Any]:
    """Returns what identifies a scorer's results beyond the harness hash."""
    return scorer.to_spec() if isinstance(scorer, FakeScorer) else {"kind": "real"}


def _cache_hits(root: Path | None, keys: dict[Any, dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    """Returns the valid cached entries for `keys`."""
    if root is None:
        return {}
    entries = {case: load_entry(root, key) for case, key in keys.items()}
    return {case: entry for case, entry in entries.items() if entry is not None}


def _store_misses(
    root: Path | None,
    keys: dict[tuple[str, str], dict[str, Any]],
    hits: dict[tuple[str, str], dict[str, Any]],
    records: Sequence[dict[str, Any]],
    experiment_id: str,
    speed_records: Sequence[dict[str, Any]] | None = None,
) -> None:
    """Stores freshly measured baseline cases; fidelity entries also keep the scored profile."""
    if root is None:
        return
    for (dataset, tier), key in keys.items():
        if (dataset, tier) in hits:
            continue
        profile = next(
            (
                record.get("profile")
                for record in _baseline_records(speed_records or [], dataset, tier)
                if record.get("ok")
            ),
            None,
        )
        store_entry(
            root,
            key,
            source_experiment_id=experiment_id,
            records=_baseline_records(records, dataset, tier),
            profile=profile,
        )


def _baseline_records(records: Sequence[dict[str, Any]], dataset: str, tier: str) -> list[dict[str, Any]]:
    """Returns baseline records for one case, in run order."""
    return [
        record
        for record in records
        if record.get("role") == "baseline" and record.get("dataset") == dataset and record.get("tier") == tier
    ]
