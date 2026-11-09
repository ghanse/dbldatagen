"""Orchestrates an RSI experiment: identities, speed, fidelity, and artifacts."""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from profiler_bench.catalog import DATASETS
from profiler_bench.evaluate.evaluate import evaluate_experiment
from profiler_bench.runner.fidelity_run import run_fidelity_samples
from profiler_bench.runner.identity import identify_build, identify_harness, machine_info
from profiler_bench.runner.measure import check_profile_determinism, run_speed_samples
from profiler_bench.runner.policy import load_policy
from profiler_bench.runner.scorer import FakeScorer, Scorer, try_real_scorer


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
        fidelity_only_tier: When set, also collect speed samples at this tier for fidelity profiles.
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

    datasets = list(config.datasets) if config.datasets is not None else sorted(DATASETS)
    speed_tiers = (
        list(config.speed_tiers) if config.speed_tiers is not None else list(policy.get("speed_tiers", ["s", "m"]))
    )
    fidelity_tier = str(policy.get("fidelity_tier", "s"))
    if fidelity_tier not in speed_tiers:
        speed_tiers = [*speed_tiers, fidelity_tier]

    proposal = config.proposal or {"type": "accuracy", "targets": datasets}
    (out / "proposal.json").write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n")

    run_meta = {
        "experiment_id": config.experiment_id,
        "control": config.control,
        "datasets": datasets,
        "speed_tiers": speed_tiers,
        "fidelity_tier": fidelity_tier,
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

    speed_records = run_speed_samples(
        baseline=baseline,
        candidate=candidate,
        datasets=datasets,
        tiers=speed_tiers,
        policy=policy,
        samples_dir=samples_dir,
        dataset_seed=int(policy.get("dataset_seed", 0)),
        cache_dir=config.cache_dir,
    )
    determinism = check_profile_determinism(speed_records)
    (samples_dir / "profile_determinism.json").write_text(json.dumps(determinism, indent=2, sort_keys=True) + "\n")

    scorer = config.scorer or try_real_scorer(
        cache_dir=out / "cache",
        harness_hash=harness.harness_hash,
        metrics_config_hash=harness.metrics_config_hash,
    )
    if scorer is None:
        raise RuntimeError("no Scorer available: inject ExperimentConfig.scorer or merge sampler/fidelity modules")
    scorer_baseline: Scorer
    scorer_candidate: Scorer
    if isinstance(scorer, FakeScorer):
        scorer_baseline = scorer.with_role("baseline")
        scorer_candidate = scorer.with_role("candidate")
    else:
        scorer_baseline = scorer
        scorer_candidate = scorer

    fidelity_records = run_fidelity_samples(
        speed_records=speed_records,
        scorer_baseline=scorer_baseline,
        scorer_candidate=scorer_candidate,
        datasets=datasets,
        policy=policy,
        samples_dir=samples_dir,
        dataset_seed=int(policy.get("dataset_seed", 0)),
    )
    del fidelity_records  # samples are on disk; evaluate reloads them

    evaluate_experiment(out)
    return out
