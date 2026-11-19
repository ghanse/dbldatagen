"""Data-volume RSI experiment: profile harness subsets and score against full data."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from profiler_bench.catalog import DATASETS
from profiler_bench.evaluate.evaluate import evaluate_experiment
from profiler_bench.runner.fidelity_run import DEFAULT_FIDELITY_CELL_TIMEOUT_S
from profiler_bench.runner.identity import BuildIdentity, identify_build, identify_harness, machine_info
from profiler_bench.runner.policy import load_policy
from profiler_bench.runner.scorer import FakeScorer, Scorer, try_real_scorer
from profiler_bench.runner.spark_conf import DEFAULT_FIDELITY_DRIVER_MEMORY, fidelity_spark_config
from profiler_bench.runner.subset import DEFAULT_FRACTIONS, DEFAULT_MAX_CLOSURE_ITERATIONS


@dataclass(frozen=True)
class DataVolumeConfig:
    """Inputs for one data-volume baseline-vs-candidate experiment."""

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
    command_line: Sequence[str] | None = None
    fidelity_driver_memory: str = DEFAULT_FIDELITY_DRIVER_MEMORY
    fidelity_cell_timeout_s: float = DEFAULT_FIDELITY_CELL_TIMEOUT_S
    fractions: Sequence[float] | None = None
    subset_seeds: int | None = None
    fidelity_seeds: int | None = None


def experiment_dir(experiment_id: str, repo_root: Path | None = None) -> Path:
    """Returns `profiler/rsi/experiments/<experiment_id>` under the repo root."""
    root = repo_root or Path(__file__).resolve().parents[4]
    return root / "profiler" / "rsi" / "experiments" / experiment_id


def run_data_volume_experiment(config: DataVolumeConfig) -> Path:
    """Runs a data-volume experiment and returns the experiment directory."""
    from profiler_bench.runner.experiment import (  # noqa: PLC0415
        _recorded_command,
        _recorded_identity,
        _recorded_path,
    )

    policy = load_policy(config.policy_path)
    repo_root = (config.repo_root or Path(__file__).resolve().parents[4]).resolve()
    datasets = list(config.datasets) if config.datasets is not None else sorted(DATASETS)
    from profiler_bench.evaluate.data_volume import require_calibrated_targets  # noqa: PLC0415

    require_calibrated_targets(policy, datasets)

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
    fidelity_tier = str(policy.get("fidelity_tier", "s"))
    fractions = (
        [float(value) for value in config.fractions]
        if config.fractions is not None
        else [float(value) for value in policy.get("fractions", DEFAULT_FRACTIONS)]
    )
    subset_seeds = int(config.subset_seeds if config.subset_seeds is not None else policy.get("subset_seeds", 1))
    fidelity_seeds = int(
        config.fidelity_seeds if config.fidelity_seeds is not None else policy.get("fidelity_seeds", 1)
    )

    proposal = config.proposal or {"type": "data_volume", "targets": datasets}
    if proposal.get("type") != "data_volume":
        proposal = {**proposal, "type": "data_volume"}
    (out / "proposal.json").write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n")

    fidelity_spark = fidelity_spark_config(driver_memory=config.fidelity_driver_memory)
    max_closure_iterations = int(
        (policy.get("data_volume") or {}).get("max_closure_iterations", DEFAULT_MAX_CLOSURE_ITERATIONS)
    )
    run_meta = {
        "experiment_id": config.experiment_id,
        "mode": "data-volume",
        "control": config.control,
        "datasets": datasets,
        "fractions": fractions,
        "subset_seeds": subset_seeds,
        "fidelity_seeds": fidelity_seeds,
        "fidelity_tier": fidelity_tier,
        "fidelity_spark": fidelity_spark,
        "fidelity_cell_timeout_s": float(config.fidelity_cell_timeout_s),
        "fidelity_model_source": "harness",
        "subset_method": "fk_preserving_uniform",
        "max_closure_iterations": max_closure_iterations,
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

    scorer = config.scorer or try_real_scorer(
        cache_dir=out / "cache",
        harness_hash=harness.harness_hash,
        metrics_config_hash=harness.metrics_config_hash,
    )
    if scorer is None:
        raise RuntimeError("no Scorer available: inject DataVolumeConfig.scorer or merge sampler/fidelity modules")

    builds = {"baseline": baseline, "candidate": candidate}
    scorers = {
        "baseline": scorer.with_role("baseline") if isinstance(scorer, FakeScorer) else scorer,
        "candidate": scorer.with_role("candidate") if isinstance(scorer, FakeScorer) else scorer,
    }

    score_cache = config.cache_dir or (out / "cache")
    records = run_data_volume_samples(
        builds=builds,
        scorers=scorers,
        datasets=datasets,
        tier=fidelity_tier,
        fractions=fractions,
        subset_seeds=subset_seeds,
        fidelity_seeds=fidelity_seeds,
        policy=policy,
        samples_dir=samples_dir,
        dataset_seed=int(policy.get("dataset_seed", 0)),
        cache_dir=score_cache,
        use_worker=not isinstance(scorer, FakeScorer),
        fidelity_spark=fidelity_spark,
        harness_hash=harness.harness_hash,
        metrics_config_hash=harness.metrics_config_hash,
    )
    del records
    evaluate_experiment(out)
    return out


def run_data_volume_samples(
    *,
    builds: Mapping[str, BuildIdentity],
    scorers: Mapping[str, Scorer],
    datasets: Sequence[str],
    tier: str,
    fractions: Sequence[float],
    subset_seeds: int,
    fidelity_seeds: int,
    policy: dict[str, Any],
    samples_dir: Path,
    dataset_seed: int = 0,
    cache_dir: Path | None = None,
    use_worker: bool = True,
    python_exe: str | None = None,
    fidelity_spark: dict[str, Any] | None = None,
    harness_hash: str = "harness",
    metrics_config_hash: str = "metrics",
) -> list[dict[str, Any]]:
    """Profiles each subset and scores reconstruction against the full tables.

    Real scoring runs in a fidelity_worker subprocess with its own SparkSession so
    the parent process (and any session-scoped pytest Spark fixture) stays clean.
    """
    samples_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    exe = python_exe or sys.executable
    spark_cfg = fidelity_spark or fidelity_spark_config()
    with tempfile.TemporaryDirectory(prefix="rsi-dv-policy-") as tmp:
        policy_path = Path(tmp) / "policy.json"
        policy_path.write_text(json.dumps(policy, sort_keys=True))
        for dataset in datasets:
            for role, build in builds.items():
                for fraction in fractions:
                    for subset_seed in range(subset_seeds):
                        profile_payload = _profile_subset(
                            build=build,
                            role=role,
                            dataset=dataset,
                            tier=tier,
                            fraction=float(fraction),
                            subset_seed=subset_seed,
                            dataset_seed=dataset_seed,
                            policy=policy,
                            policy_path=policy_path,
                            cache_dir=cache_dir,
                            python_exe=exe,
                            use_worker=use_worker,
                            scorer=scorers[role],
                        )
                        if use_worker:
                            batch = _score_subset_via_worker(
                                role=role,
                                dataset=dataset,
                                tier=tier,
                                fraction=float(fraction),
                                subset_seed=subset_seed,
                                fidelity_seeds=fidelity_seeds,
                                dataset_seed=dataset_seed,
                                profile_payload=profile_payload,
                                scorer=scorers[role],
                                samples_dir=samples_dir,
                                cache_dir=cache_dir,
                                fidelity_spark=spark_cfg,
                                harness_hash=harness_hash,
                                metrics_config_hash=metrics_config_hash,
                                python_exe=exe,
                            )
                            records.extend(batch)
                        else:
                            for seed in range(fidelity_seeds):
                                record = _score_subset_profile(
                                    role=role,
                                    dataset=dataset,
                                    tier=tier,
                                    fraction=float(fraction),
                                    subset_seed=subset_seed,
                                    seed=seed,
                                    dataset_seed=dataset_seed,
                                    profile_payload=profile_payload,
                                    scorer=scorers[role],
                                    samples_dir=samples_dir,
                                )
                                records.append(record)
    return records


def _profile_subset(
    *,
    build: BuildIdentity,
    role: str,
    dataset: str,
    tier: str,
    fraction: float,
    subset_seed: int,
    dataset_seed: int,
    policy: dict[str, Any],
    policy_path: Path,
    cache_dir: Path | None,
    python_exe: str,
    use_worker: bool,
    scorer: Scorer,
) -> dict[str, Any]:
    if not use_worker:
        # FakeScorer path: invent a minimal profile and synthetic subset meta.
        del scorer
        kept = max(1, int(1000 * fraction))
        return {
            "ok": True,
            "profile": {"tables": [{"name": "t", "row_count": kept}]},
            "subset": {
                "fraction": fraction,
                "actual_fractions": {"t": kept / 1000.0},
                "overall_actual_fraction": kept / 1000.0,
                "subset_seed": subset_seed,
                "method": "fk_preserving_uniform",
                "full_row_counts": {"t": 1000},
                "subset_row_counts": {"t": kept},
                "scales": {"t": 1000.0 / kept},
            },
            "wall_s": 0.01,
            "role": role,
        }

    with tempfile.TemporaryDirectory(prefix="rsi-dv-out-") as tmp:
        out = Path(tmp) / "worker.json"
        command = [
            python_exe,
            "-m",
            "profiler_bench.runner.worker",
            "--worktree",
            str(build.worktree),
            "--dataset",
            dataset,
            "--tier",
            tier,
            "--seed",
            str(dataset_seed),
            "--policy-json",
            str(policy_path),
            "--out",
            str(out),
            "--fraction",
            str(fraction),
            "--subset-seed",
            str(subset_seed),
        ]
        if cache_dir is not None:
            command.extend(["--cache-dir", str(cache_dir)])
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if out.is_file():
            payload = json.loads(out.read_text())
        else:
            payload = {
                "ok": False,
                "error": "worker produced no output",
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        payload["returncode"] = completed.returncode
        payload["role"] = role
        return cast(dict[str, Any], payload)


def _score_subset_via_worker(
    *,
    role: str,
    dataset: str,
    tier: str,
    fraction: float,
    subset_seed: int,
    fidelity_seeds: int,
    dataset_seed: int,
    profile_payload: dict[str, Any],
    scorer: Scorer,
    samples_dir: Path,
    cache_dir: Path | None,
    fidelity_spark: dict[str, Any],
    harness_hash: str,
    metrics_config_hash: str,
    python_exe: str,
) -> list[dict[str, Any]]:
    """Scores one subset profile in a fidelity_worker subprocess (isolated Spark)."""
    from profiler_bench.evaluate.data_volume import fraction_token  # noqa: PLC0415
    from profiler_bench.runner.fidelity_run import _scorer_spec  # noqa: PLC0415
    from profiler_bench.runner.subset import apply_table_scales  # noqa: PLC0415

    base_meta = {
        "fraction": fraction,
        "subset_seed": subset_seed,
        "subset_method": "fk_preserving_uniform",
        "profile_wall_s": profile_payload.get("wall_s"),
    }
    if not profile_payload.get("ok"):
        failed = []
        for seed in range(fidelity_seeds):
            record = {
                "kind": "fidelity",
                "role": role,
                "dataset": dataset,
                "tier": tier,
                "seed": seed,
                **base_meta,
                "ok": False,
                "error_class": "SubsetProfileError",
                "error": profile_payload.get("error"),
                "privacy_passed": False,
            }
            _write_record(samples_dir, record)
            failed.append(record)
        return failed

    subset_meta = profile_payload.get("subset") or {}
    scales = {str(name): float(value) for name, value in dict(subset_meta.get("scales") or {}).items()}
    base_meta["subset"] = subset_meta
    base_meta["scales"] = scales
    scaled_profile = (
        apply_table_scales(dict(profile_payload.get("profile") or {}), scales)
        if scales
        else dict(profile_payload.get("profile") or {})
    )

    with tempfile.TemporaryDirectory(prefix="rsi-dv-score-") as tmp:
        tmp_path = Path(tmp)
        profile_path = tmp_path / "profile.json"
        scorer_path = tmp_path / "scorer.json"
        spark_path = tmp_path / "fidelity-spark.json"
        seeds_path = tmp_path / "seeds.json"
        out_path = tmp_path / "cell.json"
        profile_path.write_text(json.dumps(scaled_profile, sort_keys=True, default=str))
        scorer_path.write_text(json.dumps(_scorer_spec(scorer, cache_dir, harness_hash, metrics_config_hash)))
        spark_path.write_text(json.dumps(fidelity_spark, sort_keys=True))
        seeds_path.write_text(json.dumps(list(range(fidelity_seeds))))
        command = [
            python_exe,
            "-m",
            "profiler_bench.runner.fidelity_worker",
            "--profile-json",
            str(profile_path),
            "--scorer-json",
            str(scorer_path),
            "--fidelity-spark-json",
            str(spark_path),
            "--seeds-json",
            str(seeds_path),
            "--out",
            str(out_path),
            "--role",
            role,
            "--dataset",
            dataset,
            "--tier",
            tier,
            "--dataset-seed",
            str(dataset_seed),
        ]
        if cache_dir is not None:
            command.extend(["--cache-dir", str(cache_dir)])
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if not out_path.is_file():
            records = []
            for seed in range(fidelity_seeds):
                record = {
                    "kind": "fidelity",
                    "role": role,
                    "dataset": dataset,
                    "tier": tier,
                    "seed": seed,
                    **base_meta,
                    "ok": False,
                    "error_class": "WorkerOutputMissing",
                    "error": f"fidelity_worker produced no output (rc={completed.returncode})",
                    "privacy_passed": False,
                }
                _write_record(samples_dir, record)
                records.append(record)
            return records
        payload = json.loads(out_path.read_text())
        if not payload.get("ok"):
            records = []
            for seed in range(fidelity_seeds):
                record = {
                    "kind": "fidelity",
                    "role": role,
                    "dataset": dataset,
                    "tier": tier,
                    "seed": seed,
                    **base_meta,
                    "ok": False,
                    "error_class": str(payload.get("error_class") or "FidelityWorkerError"),
                    "error": payload.get("error"),
                    "privacy_passed": False,
                }
                _write_record(samples_dir, record)
                records.append(record)
            return records
        records = []
        for raw in payload.get("records") or []:
            record = dict(raw)
            record.update(base_meta)
            token = fraction_token(fraction)
            path = samples_dir / (
                f"fidelity-{role}-{dataset}-{tier}-f-{token}-subset-{subset_seed}-seed-{record['seed']}.json"
            )
            path.write_text(json.dumps(record, sort_keys=True, default=str) + "\n")
            records.append(record)
        return records


def _score_subset_profile(
    *,
    role: str,
    dataset: str,
    tier: str,
    fraction: float,
    subset_seed: int,
    seed: int,
    dataset_seed: int,
    profile_payload: dict[str, Any],
    scorer: Scorer,
    samples_dir: Path,
) -> dict[str, Any]:
    """In-process scoring for FakeScorer (no Spark)."""
    from profiler_bench.evaluate.data_volume import fraction_token  # noqa: PLC0415

    base = {
        "kind": "fidelity",
        "role": role,
        "dataset": dataset,
        "tier": tier,
        "fraction": fraction,
        "subset_seed": subset_seed,
        "seed": seed,
        "subset_method": "fk_preserving_uniform",
    }
    if not profile_payload.get("ok"):
        record = {
            **base,
            "ok": False,
            "error_class": "SubsetProfileError",
            "error": profile_payload.get("error"),
            "privacy_passed": False,
        }
        _write_record(samples_dir, record)
        return record

    profile = profile_payload.get("profile")
    subset_meta = profile_payload.get("subset") or {}
    scales = {str(name): float(value) for name, value in dict(subset_meta.get("scales") or {}).items()}
    try:
        if not isinstance(scorer, FakeScorer):
            raise TypeError("in-process data-volume scoring requires FakeScorer; real scoring uses fidelity_worker")
        scores = scorer.score(
            profile or {},
            dataset=dataset,
            tier=tier,
            seed=seed,
            structure_seed=dataset_seed,
            sample_seed=dataset_seed,
        )
        relative = scores.relative_reconstruction
        if relative is None:
            raise RuntimeError("fake scorer missing relative reconstruction")
        for key, value in scorer.relative_by_fraction.items():
            if abs(float(key) - fraction) <= 1e-12:
                relative = float(value)
                break
        reconstruction = relative * float(scores.ceiling or 1.0)
        record = {
            **base,
            "ok": True,
            "reconstruction": reconstruction,
            "structure": scores.structure,
            "privacy_passed": scores.privacy_passed,
            "privacy_unmeasured": scores.privacy_unmeasured,
            "ceiling": scores.ceiling,
            "relative_reconstruction": relative,
            "detail": scores.detail,
            "metrics_config_hash": scores.metrics_config_hash,
            "subset": subset_meta,
            "scales": scales,
            "profile_wall_s": profile_payload.get("wall_s"),
        }
    except Exception as error:
        record = {
            **base,
            "ok": False,
            "error_class": type(error).__name__,
            "error": str(error),
            "privacy_passed": False,
        }
    token = fraction_token(fraction)
    path = samples_dir / (f"fidelity-{role}-{dataset}-{tier}-f-{token}-subset-{subset_seed}-seed-{seed}.json")
    path.write_text(json.dumps(record, sort_keys=True, default=str) + "\n")
    return record


def _write_record(samples_dir: Path, record: dict[str, Any]) -> None:
    from profiler_bench.evaluate.data_volume import fraction_token  # noqa: PLC0415

    token = fraction_token(float(record["fraction"]))
    path = samples_dir / (
        f"fidelity-{record['role']}-{record['dataset']}-{record['tier']}"
        f"-f-{token}-subset-{record['subset_seed']}-seed-{record['seed']}.json"
    )
    path.write_text(json.dumps(record, sort_keys=True, default=str) + "\n")


def calibrate_targets(
    *,
    baseline_experiment: Path,
    output_policy: Path,
    base_policy: Path | None = None,
    margin: float = 0.01,
) -> dict[str, Any]:
    """Writes a proposed data-volume policy with ``T_d`` and frozen control noise widths.

    Expects ``baseline_experiment`` to be a same-build control scored at ``f = 1``
    over K sampler seeds x S subset seeds. Both ``T_d`` and the per-dataset noise
    width (half the f=1 mean CI width) are written with provenance (commit, harness
    hash, run id).

    Refuses (writes nothing) unless: ``run.control`` is true, baseline and candidate
    ``profiler_src_hash`` match, the f=1 K x S matrix is complete for every selected
    dataset with every record ``ok``, and privacy passed on every f=1 record.
    """
    from profiler_bench.evaluate.data_volume import calibrate_targets_from_records  # noqa: PLC0415
    from profiler_bench.evaluate.evaluate import load_samples  # noqa: PLC0415
    from profiler_bench.runner.policy import DEFAULT_POLICY, policy_path_hash  # noqa: PLC0415

    root = Path(baseline_experiment)
    run = json.loads((root / "run.json").read_text())
    _speed, fidelity = load_samples(root / "samples")
    _require_calibrate_provenance(run, fidelity)
    boot = (run.get("policy") or {}).get("bootstrap") or {}
    targets, noise = calibrate_targets_from_records(
        fidelity,
        role="baseline",
        fraction=1.0,
        margin=margin,
        n_bootstrap=int(boot.get("samples", 2000)),
        bootstrap_seed=int(boot.get("seed", 0)),
        confidence=float(boot.get("confidence", 0.95)),
    )
    if not targets:
        raise ValueError(
            "calibrate-targets found no ok baseline f=1 relative_reconstruction scores "
            f"for datasets={list(run.get('datasets') or [])}"
        )
    policy_path = base_policy or Path(__file__).resolve().parents[3] / "policies" / "data_volume.json"
    if policy_path.is_file():
        policy = json.loads(policy_path.read_text())
    else:
        policy = json.loads((base_policy or DEFAULT_POLICY).read_text())
    harness = run.get("harness") or {}
    baseline = run.get("baseline") or {}
    provenance = {
        "baseline_experiment": str(root),
        "run_id": run.get("experiment_id"),
        "commit": baseline.get("commit") or harness.get("commit"),
        "harness_hash": harness.get("harness_hash"),
        "profiler_src_hash": baseline.get("profiler_src_hash"),
        "control": bool(run.get("control")),
        "fraction": 1.0,
        "margin": margin,
        "note": ("Proposed by calibrate-targets from a same-build control at f=1; " "polly freezes the accepted file."),
    }
    policy["targets"] = targets
    policy["targets_provenance"] = provenance
    data_volume = dict(policy.get("data_volume") or {})
    data_volume["control_noise_width"] = noise
    data_volume["control_noise_width_provenance"] = provenance
    data_volume.setdefault("max_closure_iterations", DEFAULT_MAX_CLOSURE_ITERATIONS)
    data_volume["f1_regression_bound"] = {
        "source": "default.json",
        "path": "accuracy.max_dataset_regression",
        "default_policy_hash": policy_path_hash(DEFAULT_POLICY),
        "note": "Loaded by reference from default.json at gate time; not duplicated here.",
    }
    policy["data_volume"] = data_volume
    policy.pop("accuracy", None)
    output_policy.parent.mkdir(parents=True, exist_ok=True)
    output_policy.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    return cast(dict[str, Any], policy)


def _require_calibrate_provenance(run: Mapping[str, Any], fidelity: Sequence[dict[str, Any]]) -> None:
    """Raises ``ValueError`` with a clear reason when calibration provenance is unsafe."""
    import math  # noqa: PLC0415

    from profiler_bench.evaluate.data_volume import data_volume_sample_key, fraction_token  # noqa: PLC0415

    if not bool(run.get("control")):
        raise ValueError(
            "calibrate-targets refuses to freeze targets: run.control must be true "
            "(same-build control at f=1 required)"
        )
    baseline = run.get("baseline") or {}
    candidate = run.get("candidate") or {}
    base_hash = baseline.get("profiler_src_hash")
    cand_hash = candidate.get("profiler_src_hash")
    if not base_hash or base_hash != cand_hash:
        raise ValueError(
            "calibrate-targets refuses to freeze targets: baseline and candidate "
            f"profiler_src_hash must be identical; got baseline={base_hash!r} candidate={cand_hash!r}"
        )

    policy = run.get("policy") or {}
    datasets = [str(name) for name in run.get("datasets") or []]
    if not datasets:
        raise ValueError("calibrate-targets refuses to freeze targets: run.datasets is empty")
    tier = str(run.get("fidelity_tier") or policy.get("fidelity_tier") or "s")
    subset_seeds = int(run.get("subset_seeds") or policy.get("subset_seeds") or 1)
    fidelity_seeds = int(run.get("fidelity_seeds") or policy.get("fidelity_seeds") or 1)
    token = fraction_token(1.0)
    expected: set[str] = set()
    for dataset in datasets:
        for subset_seed in range(subset_seeds):
            for seed in range(fidelity_seeds):
                expected.add(f"fidelity/baseline/{dataset}/{tier}/f-{token}/subset-{subset_seed}/seed-{seed}")

    by_key = {data_volume_sample_key(record): record for record in fidelity}
    missing = sorted(expected - set(by_key))
    if missing:
        raise ValueError(
            "calibrate-targets refuses to freeze targets: incomplete f=1 baseline matrix "
            f"(need K={fidelity_seeds} x S={subset_seeds} per dataset); missing={missing[:8]}"
            + ("..." if len(missing) > 8 else "")
        )

    f1_records = [
        record
        for record in fidelity
        if record.get("kind") == "fidelity" and math.isclose(float(record.get("fraction", 1.0)), 1.0, abs_tol=1e-12)
    ]
    not_ok = [data_volume_sample_key(record) for record in f1_records if not record.get("ok")]
    if not_ok:
        raise ValueError(
            "calibrate-targets refuses to freeze targets: f=1 records must all be ok; "
            f"failed={not_ok[:8]}" + ("..." if len(not_ok) > 8 else "")
        )
    privacy_failed = [
        data_volume_sample_key(record) for record in f1_records if record.get("privacy_passed") is not True
    ]
    if privacy_failed:
        raise ValueError(
            "calibrate-targets refuses to freeze targets: privacy must pass on every f=1 record; "
            f"failed={privacy_failed[:8]}" + ("..." if len(privacy_failed) > 8 else "")
        )
    # Also require every expected baseline cell itself is ok + privacy-passed (defensive).
    for key in sorted(expected):
        record = by_key[key]
        if not record.get("ok") or record.get("privacy_passed") is not True:
            raise ValueError(
                "calibrate-targets refuses to freeze targets: expected baseline f=1 cell "
                f"{key} is not ok with privacy_passed=true"
            )
