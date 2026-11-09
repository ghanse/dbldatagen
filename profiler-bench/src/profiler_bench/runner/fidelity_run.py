"""Fidelity samples: paired seeds across baseline and candidate profiles."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from profiler_bench.runner.measure import rotate_order
from profiler_bench.runner.scorer import FakeScorer, FidelityScores, Scorer, real_scorer_spec
from profiler_bench.runner.spark_conf import DEFAULT_FIDELITY_DRIVER_MEMORY, fidelity_spark_config


DEFAULT_FIDELITY_CELL_TIMEOUT_S = 45 * 60
DEFAULT_WORKER_KILL_GRACE_S = 5.0


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
    fidelity_spark: dict[str, Any] | None = None,
    cache_dir: Path | None = None,
    harness_hash: str = "",
    metrics_config_hash: str = "",
    python_exe: str | None = None,
    cell_timeout_s: float = DEFAULT_FIDELITY_CELL_TIMEOUT_S,
    worker_prefix: Sequence[str] | None = None,
    kill_grace_s: float = DEFAULT_WORKER_KILL_GRACE_S,
) -> list[dict[str, Any]]:
    """Scores one representative profile per build/dataset in an isolated worker.

    Each (role, dataset) cell runs in a fresh subprocess with its own SparkSession.
    All K paired seeds for that cell share the process. A crash in one cell cannot
    affect any other cell.
    """
    tier = str(policy.get("fidelity_tier", "s"))
    k = int(policy.get("fidelity_seeds", 5))
    seeds = paired_seeds(k)
    samples_dir.mkdir(parents=True, exist_ok=True)
    spark_config = fidelity_spark or fidelity_spark_config(driver_memory=DEFAULT_FIDELITY_DRIVER_MEMORY)
    profiles = _representative_profiles(speed_records, datasets, tier)
    scorers = {"baseline": scorer_baseline, "candidate": scorer_candidate}
    records: list[dict[str, Any]] = []
    exe = python_exe or sys.executable
    for case_index, dataset in enumerate(datasets):
        order = rotate_order(0, case_index)
        if dataset not in profiles.get("baseline", {}) or dataset not in profiles.get("candidate", {}):
            for role in order:
                for seed in seeds:
                    records.append(
                        _failed_seed(
                            role,
                            dataset,
                            tier,
                            seed,
                            "MissingProfileError",
                            "missing speed profile for fidelity scoring",
                            samples_dir,
                        )
                    )
            continue
        for role in order:
            cell_records = _run_cell(
                role=role,
                dataset=dataset,
                tier=tier,
                seeds=seeds,
                profile=profiles[role][dataset],
                scorer=scorers[role],
                samples_dir=samples_dir,
                dataset_seed=dataset_seed,
                fidelity_spark=spark_config,
                cache_dir=cache_dir,
                harness_hash=harness_hash,
                metrics_config_hash=metrics_config_hash,
                python_exe=exe,
                cell_timeout_s=cell_timeout_s,
                worker_prefix=worker_prefix,
                kill_grace_s=kill_grace_s,
            )
            records.extend(cell_records)
    return records


def score_one_seed(
    *,
    role: str,
    dataset: str,
    tier: str,
    seed: int,
    profile: dict[str, Any],
    scorer: Scorer,
    dataset_seed: int,
) -> dict[str, Any]:
    """Scores one seed and returns the fidelity sample record (does not write files)."""
    try:
        scores: FidelityScores = scorer.score(
            profile,
            dataset=dataset,
            tier=tier,
            seed=seed,
            structure_seed=dataset_seed,
            sample_seed=dataset_seed,
        )
        if scores.ceiling is None or scores.relative_reconstruction is None:
            raise RuntimeError("scorer did not produce ceiling-relative reconstruction")
        return {
            "kind": "fidelity",
            "role": role,
            "dataset": dataset,
            "tier": tier,
            "seed": seed,
            "ok": True,
            "reconstruction": scores.reconstruction,
            "structure": scores.structure,
            "privacy_passed": scores.privacy_passed,
            "privacy_unmeasured": scores.privacy_unmeasured,
            "ceiling": scores.ceiling,
            "relative_reconstruction": scores.relative_reconstruction,
            "detail": scores.detail,
            "metrics_config_hash": scores.metrics_config_hash,
        }
    except Exception as error:
        return {
            "kind": "fidelity",
            "role": role,
            "dataset": dataset,
            "tier": tier,
            "seed": seed,
            "ok": False,
            "error_class": type(error).__name__,
            "error": str(error),
        }


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
) -> dict[str, Any]:
    """Scores one seed and writes the sample JSON (kept for unit tests)."""
    record = score_one_seed(
        role=role,
        dataset=dataset,
        tier=tier,
        seed=seed,
        profile=profile,
        scorer=scorer,
        dataset_seed=dataset_seed,
    )
    path = samples_dir / f"fidelity-{role}-{dataset}-{tier}-seed-{seed}.json"
    path.write_text(json.dumps(record, sort_keys=True, default=str))
    return record


def _run_cell(
    *,
    role: str,
    dataset: str,
    tier: str,
    seeds: Sequence[int],
    profile: dict[str, Any],
    scorer: Scorer,
    samples_dir: Path,
    dataset_seed: int,
    fidelity_spark: dict[str, Any],
    cache_dir: Path | None,
    harness_hash: str,
    metrics_config_hash: str,
    python_exe: str,
    cell_timeout_s: float,
    worker_prefix: Sequence[str] | None,
    kill_grace_s: float,
) -> list[dict[str, Any]]:
    """Runs one (role, dataset) cell in a fresh worker subprocess."""
    with tempfile.TemporaryDirectory(prefix="rsi-fidelity-") as tmp:
        tmp_path = Path(tmp)
        profile_path = tmp_path / "profile.json"
        scorer_path = tmp_path / "scorer.json"
        spark_path = tmp_path / "fidelity-spark.json"
        seeds_path = tmp_path / "seeds.json"
        out_path = tmp_path / "cell.json"
        profile_path.write_text(json.dumps(profile, sort_keys=True, default=str))
        scorer_path.write_text(json.dumps(_scorer_spec(scorer, cache_dir, harness_hash, metrics_config_hash)))
        spark_path.write_text(json.dumps(fidelity_spark, sort_keys=True))
        seeds_path.write_text(json.dumps(list(seeds)))
        prefix = (
            list(worker_prefix)
            if worker_prefix is not None
            else [python_exe, "-m", "profiler_bench.runner.fidelity_worker"]
        )
        command = [
            *prefix,
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
        completed = _run_worker_command(
            command,
            timeout_s=cell_timeout_s,
            kill_grace_s=kill_grace_s,
            log_dir=tmp_path,
        )
        return _records_from_worker(
            role=role,
            dataset=dataset,
            tier=tier,
            seeds=seeds,
            out_path=out_path,
            samples_dir=samples_dir,
            completed=completed,
        )


def _run_worker_command(
    command: Sequence[str],
    *,
    timeout_s: float,
    kill_grace_s: float,
    log_dir: Path,
) -> subprocess.CompletedProcess[str]:
    """Runs a worker in its own process group and enforces `timeout_s`.

    Worker stdout/stderr go to log files (not pipes) so a surviving descendant cannot
    hang the parent in ``communicate()``.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "worker.stdout"
    stderr_path = log_dir / "worker.stderr"
    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        proc = subprocess.Popen(
            list(command),
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(proc.pid, grace_s=kill_grace_s)
            try:
                proc.wait(timeout=max(kill_grace_s, 0.1) + 2.0)
            except subprocess.TimeoutExpired:
                _signal_process_group(proc.pid, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=2.0)
    result = subprocess.CompletedProcess(
        args=list(command),
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout=stdout_path.read_text() if stdout_path.is_file() else "",
        stderr=stderr_path.read_text() if stderr_path.is_file() else "",
    )
    result.timed_out = timed_out  # type: ignore[attr-defined]
    return result


def _terminate_process_group(pid: int, *, grace_s: float) -> None:
    """SIGTERM the worker process group, then SIGKILL if the group still exists."""
    _signal_process_group(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not _process_group_alive(pid):
            return
        time.sleep(0.05)
    if _process_group_alive(pid):
        _signal_process_group(pid, signal.SIGKILL)


def _signal_process_group(pid: int, sig: signal.Signals | int) -> bool:
    """Signals a process group; falls back to the leader pid. False if already gone."""
    try:
        os.killpg(pid, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        try:
            os.kill(pid, sig)
            return True
        except (ProcessLookupError, PermissionError):
            return False


def _process_group_alive(pgid: int) -> bool:
    """Returns whether any member of process group `pgid` still exists."""
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # macOS can raise PermissionError for signal 0 even when the group remains.
        return True


def _scorer_spec(
    scorer: Scorer,
    cache_dir: Path | None,
    harness_hash: str,
    metrics_config_hash: str,
) -> dict[str, Any]:
    if isinstance(scorer, FakeScorer):
        return scorer.to_spec()
    if cache_dir is None:
        raise ValueError("real fidelity scoring requires cache_dir")
    return real_scorer_spec(
        cache_dir=cache_dir,
        harness_hash=harness_hash,
        metrics_config_hash=metrics_config_hash,
    )


def _records_from_worker(
    *,
    role: str,
    dataset: str,
    tier: str,
    seeds: Sequence[int],
    out_path: Path,
    samples_dir: Path,
    completed: subprocess.CompletedProcess[str],
) -> list[dict[str, Any]]:
    failure = _worker_failure(
        out_path=out_path, completed=completed, role=role, dataset=dataset, tier=tier, seeds=seeds
    )
    if failure is not None:
        error_class, error = failure
        return _failed_cell(role, dataset, tier, seeds, error_class, error, samples_dir)

    payload, _ = _load_worker_payload(out_path)
    assert payload is not None  # narrowed by _worker_failure
    records = [dict(record) for record in cast(list[dict[str, Any]], payload["records"])]
    for record in records:
        _write_sample(samples_dir, record)
    return records


def _worker_failure(
    *,
    out_path: Path,
    completed: subprocess.CompletedProcess[str],
    role: str,
    dataset: str,
    tier: str,
    seeds: Sequence[int],
) -> tuple[str, str] | None:
    """Returns `(error_class, error)` when the worker result cannot be accepted."""
    error_class: str | None = None
    error: str | None = None
    if bool(getattr(completed, "timed_out", False)):
        error_class = "WorkerTimeout"
        error = f"fidelity worker exceeded cell timeout (returncode={completed.returncode})"
    else:
        payload, parse_error = _load_worker_payload(out_path)
        if completed.returncode != 0:
            detail = parse_error or (payload or {}).get("error") or completed.stderr[-2000:]
            error_class = "WorkerCrashed"
            error = f"fidelity worker exited {completed.returncode}: {detail}"
        elif payload is None:
            error_class = "WorkerOutputInvalid"
            error = parse_error or "fidelity worker produced missing or invalid JSON"
        elif not payload.get("ok"):
            error_class = "WorkerOutputInvalid"
            error = str(payload.get("error") or "fidelity worker payload missing ok records")
        else:
            raw_records = payload.get("records")
            shape_error = _records_shape_error(raw_records)
            if shape_error is not None:
                error_class = "WorkerOutputInvalid"
                error = shape_error
            else:
                records = [dict(record) for record in cast(list[dict[str, Any]], raw_records)]
                mismatch = _payload_mismatch(records, role=role, dataset=dataset, tier=tier, seeds=seeds)
                if mismatch is not None:
                    error_class = "WorkerOutputMismatch"
                    error = mismatch
    if error_class is None or error is None:
        return None
    return error_class, error


def _records_shape_error(raw_records: object) -> str | None:
    """Returns an error when `records` is not a list of dicts."""
    if not isinstance(raw_records, list):
        return f"records must be a list, got {type(raw_records).__name__}"
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            return f"records[{index}] must be an object, got {type(record).__name__}"
    return None


def _load_worker_payload(out_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not out_path.is_file():
        return None, "fidelity worker produced no output file"
    try:
        payload = json.loads(out_path.read_text())
    except json.JSONDecodeError as error:
        return None, f"truncated or invalid JSON: {error}"
    if not isinstance(payload, dict):
        return None, "worker output JSON must be an object"
    return payload, None


def _payload_mismatch(
    records: Sequence[dict[str, Any]],
    *,
    role: str,
    dataset: str,
    tier: str,
    seeds: Sequence[int],
) -> str | None:
    expected = list(seeds)
    actual_seeds = [record.get("seed") for record in records]
    if actual_seeds != expected:
        return f"seed set mismatch: expected {expected}, got {actual_seeds}"
    for record in records:
        if record.get("role") != role:
            return f"role mismatch: expected {role!r}, got {record.get('role')!r}"
        if record.get("dataset") != dataset:
            return f"dataset mismatch: expected {dataset!r}, got {record.get('dataset')!r}"
        if record.get("tier") != tier:
            return f"tier mismatch: expected {tier!r}, got {record.get('tier')!r}"
        if record.get("kind") != "fidelity":
            return f"kind mismatch: expected 'fidelity', got {record.get('kind')!r}"
    return None


def _failed_cell(
    role: str,
    dataset: str,
    tier: str,
    seeds: Sequence[int],
    error_class: str,
    error: str,
    samples_dir: Path,
) -> list[dict[str, Any]]:
    return [_failed_seed(role, dataset, tier, seed, error_class, error, samples_dir) for seed in seeds]


def _failed_seed(
    role: str,
    dataset: str,
    tier: str,
    seed: int,
    error_class: str,
    error: str,
    samples_dir: Path,
) -> dict[str, Any]:
    record = {
        "kind": "fidelity",
        "role": role,
        "dataset": dataset,
        "tier": tier,
        "seed": seed,
        "ok": False,
        "error_class": error_class,
        "error": error,
    }
    _write_sample(samples_dir, record)
    return record


def _write_sample(samples_dir: Path, record: dict[str, Any]) -> None:
    seed = record["seed"]
    samples_dir.mkdir(parents=True, exist_ok=True)
    path = samples_dir / f"fidelity-{record['role']}-{record['dataset']}-{record['tier']}-seed-{seed}.json"
    path.write_text(json.dumps(record, sort_keys=True, default=str))


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
