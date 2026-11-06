"""Re-evaluates saved experiment samples without re-running the worker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from profiler_bench.evaluate.gates import evaluate_gates
from profiler_bench.evaluate.report import write_outputs


def evaluate_experiment(experiment_dir: Path) -> dict[str, Any]:
    """Loads samples under `experiment_dir`, applies the policy, and writes artifacts."""
    root = Path(experiment_dir)
    run = json.loads((root / "run.json").read_text())
    proposal_path = root / "proposal.json"
    proposal = json.loads(proposal_path.read_text()) if proposal_path.is_file() else {"type": "accuracy"}
    policy = run["policy"]
    speed_records, fidelity_records = load_samples(root / "samples")
    determinism_path = root / "samples" / "profile_determinism.json"
    determinism = json.loads(determinism_path.read_text()) if determinism_path.is_file() else None
    verdict = evaluate_gates(
        speed_records=speed_records,
        fidelity_records=fidelity_records,
        policy=policy,
        proposal=proposal,
    )
    write_outputs(
        root,
        run=run,
        speed_records=speed_records,
        fidelity_records=fidelity_records,
        verdict=verdict,
        determinism=determinism,
    )
    return verdict


def load_samples(samples_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns `(speed_records, fidelity_records)` from JSON files in `samples_dir`."""
    speed: list[dict[str, Any]] = []
    fidelity: list[dict[str, Any]] = []
    if not samples_dir.is_dir():
        return speed, fidelity
    for path in sorted(samples_dir.glob("*.json")):
        if path.name == "profile_determinism.json":
            continue
        record = json.loads(path.read_text())
        kind = record.get("kind")
        if kind == "speed":
            speed.append(record)
        elif kind == "fidelity":
            fidelity.append(record)
    return speed, fidelity
