"""Fresh-process speed sampling with rotating build order."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from profiler_bench.runner.identity import BuildIdentity
from profiler_bench.runner.policy import DEFAULT_POLICY


def rotate_order(sample_index: int, case_index: int = 0) -> tuple[str, str]:
    """Returns `(first_role, second_role)` rotating A/B then B/A across cases."""
    if (sample_index + case_index) % 2 == 0:
        return ("baseline", "candidate")
    return ("candidate", "baseline")


def run_speed_samples(
    *,
    baseline: BuildIdentity,
    candidate: BuildIdentity,
    datasets: Sequence[str],
    tiers: Sequence[str],
    policy: dict[str, Any],
    samples_dir: Path,
    dataset_seed: int = 0,
    cache_dir: Path | None = None,
    python_exe: str | None = None,
) -> list[dict[str, Any]]:
    """Runs N fresh-process speed samples per (build, dataset, tier) with rotating order."""
    speed_n = policy.get("speed_samples", {})
    builds = {"baseline": baseline, "candidate": candidate}
    records: list[dict[str, Any]] = []
    case_index = 0
    with tempfile.TemporaryDirectory(prefix="rsi-policy-") as tmp:
        policy_path = Path(tmp) / "policy.json"
        policy_path.write_text(json.dumps(policy, sort_keys=True))
        for dataset in datasets:
            for tier in tiers:
                n = int(speed_n.get(tier, speed_n.get("s", 5)))
                for sample_index in range(n):
                    order = rotate_order(sample_index, case_index)
                    for role in order:
                        record = _one_sample(
                            build=builds[role],
                            role=role,
                            dataset=dataset,
                            tier=tier,
                            sample_index=sample_index,
                            order=order,
                            policy=policy,
                            policy_path=policy_path,
                            samples_dir=samples_dir,
                            dataset_seed=dataset_seed,
                            cache_dir=cache_dir,
                            python_exe=python_exe or sys.executable,
                        )
                        records.append(record)
                case_index += 1
    return records


def _one_sample(
    *,
    build: BuildIdentity,
    role: str,
    dataset: str,
    tier: str,
    sample_index: int,
    order: tuple[str, str],
    policy: dict[str, Any],
    policy_path: Path,
    samples_dir: Path,
    dataset_seed: int,
    cache_dir: Path | None,
    python_exe: str,
) -> dict[str, Any]:
    samples_dir.mkdir(parents=True, exist_ok=True)
    out = samples_dir / f"speed-{role}-{dataset}-{tier}-{sample_index}.json"
    command = [
        python_exe,
        "-m",
        "profiler_bench.runner.worker",
        "--worktree",
        build.worktree,
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
    record = {
        "kind": "speed",
        "role": role,
        "dataset": dataset,
        "tier": tier,
        "sample_index": sample_index,
        "order": list(order),
        "build_commit": build.commit,
        "build_profiler_src_hash": build.profiler_src_hash,
        "ok": bool(payload.get("ok")),
        "wall_s": payload.get("wall_s"),
        "cpu_s": payload.get("cpu_s"),
        "peak_rss_bytes": payload.get("peak_rss_bytes"),
        "jvm_rss_bytes": payload.get("jvm_rss_bytes"),
        "profile_hash": payload.get("profile_hash"),
        "profile": payload.get("profile"),
        "profiler_module_file": payload.get("profiler_module_file"),
        "error": payload.get("error"),
        "returncode": completed.returncode,
        "not_measured": payload.get("not_measured", policy.get("not_measured", [])),
        "policy_file": str(DEFAULT_POLICY.name),
    }
    out.write_text(json.dumps(record, sort_keys=True, default=str))
    return record


def check_profile_determinism(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Flags non-deterministic profiles across successful speed samples of the same case."""
    groups: dict[tuple[str, str, str], set[str]] = {}
    for record in records:
        if record.get("kind") != "speed" or not record.get("ok") or not record.get("profile_hash"):
            continue
        key = (str(record["role"]), str(record["dataset"]), str(record["tier"]))
        groups.setdefault(key, set()).add(str(record["profile_hash"]))
    flags = {
        f"{role}/{dataset}/{tier}": sorted(hashes)
        for (role, dataset, tier), hashes in groups.items()
        if len(hashes) > 1
    }
    return {"deterministic": not flags, "divergent": flags}
