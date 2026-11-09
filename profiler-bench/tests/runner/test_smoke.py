"""End-to-end smoke: baseline vs itself on xs with a fake Scorer."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pytest

from profiler_bench.runner.experiment import ExperimentConfig, run_experiment
from profiler_bench.runner.policy import DEFAULT_POLICY
from profiler_bench.runner.scorer import FakeScorer


REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def smoke_policy(tmp_path: Path) -> Path:
    policy = json.loads(DEFAULT_POLICY.read_text())
    policy["speed_samples"] = {"xs": 2, "s": 2, "m": 2}
    policy["fidelity_seeds"] = 2
    policy["fidelity_tier"] = "xs"
    policy["speed_tiers"] = ["xs"]
    path = tmp_path / "smoke-policy.json"
    path.write_text(json.dumps(policy))
    return path


def test_control_smoke_never_accepted(tmp_path: Path, smoke_policy: Path, catalog_cache: Path) -> None:
    experiment_id = "smoke-control-p1-6"
    out_root = tmp_path / "repo"
    (out_root / "profiler" / "rsi" / "experiments").mkdir(parents=True)
    try:
        out = run_experiment(
            ExperimentConfig(
                experiment_id=experiment_id,
                baseline_worktree=REPO,
                candidate_worktree=REPO,
                policy_path=smoke_policy,
                datasets=["numeric"],
                control=True,
                scorer=FakeScorer(reconstruction=0.8, structure=0.7, privacy_passed=True, ceiling_value=0.9),
                repo_root=out_root,
                cache_dir=catalog_cache,
                proposal={"type": "accuracy", "proposal_id": experiment_id, "targets": ["numeric"]},
                speed_tiers=["xs"],
            )
        )
        assert out == out_root / "profiler" / "rsi" / "experiments" / experiment_id
        for name in ("run.json", "summary.json", "verdict.json", "REPORT.md", "benchmark_entry.json", "proposal.json"):
            assert (out / name).is_file()
        samples = list((out / "samples").glob("speed-*.json"))
        assert len(samples) == 4  # 2 roles * 2 samples
        fidelity = list((out / "samples").glob("fidelity-*.json"))
        assert len(fidelity) == 4  # 2 roles * 2 seeds
        verdict = json.loads((out / "verdict.json").read_text())
        assert verdict["verdict"] != "accepted"
        run = json.loads((out / "run.json").read_text())
        assert run["baseline"]["commit"] == run["candidate"]["commit"]
        assert "not_measured" in run
        artifacts = [
            out / name for name in ("run.json", "summary.json", "verdict.json", "REPORT.md", "benchmark_entry.json")
        ]
        absolute_path = re.compile(r"/(?:Users|home|private|tmp|opt|var)/")
        for artifact in artifacts:
            text = artifact.read_text()
            assert os.path.expanduser("~") not in text
            assert absolute_path.search(text) is None
        orders = {tuple(json.loads(path.read_text())["order"]) for path in samples}
        assert ("baseline", "candidate") in orders
        assert ("candidate", "baseline") in orders
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


@pytest.mark.slow
def test_real_control_is_inconclusive(tmp_path: Path, smoke_policy: Path, catalog_cache: Path) -> None:
    experiment_id = "real-control-xs"
    out = run_experiment(
        ExperimentConfig(
            experiment_id=experiment_id,
            baseline_worktree=REPO,
            candidate_worktree=REPO,
            policy_path=smoke_policy,
            datasets=["retail", "numeric"],
            control=True,
            repo_root=tmp_path,
            cache_dir=catalog_cache,
            proposal={"type": "accuracy", "proposal_id": experiment_id, "targets": ["retail", "numeric"]},
            speed_tiers=["xs"],
        )
    )
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["verdict"] == "inconclusive"
    assert "control" in verdict["reasons"]
    fidelity = [json.loads(path.read_text()) for path in (out / "samples").glob("fidelity-*.json")]
    assert len(fidelity) == 8
    assert all(record["ok"] for record in fidelity)
    assert all(record["detail"]["relative_to_ceiling"] == record["relative_reconstruction"] for record in fidelity)
