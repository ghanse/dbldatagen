"""Fidelity cell isolation, recorded spark config, and subprocess round-trip."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from profiler_bench.runner.experiment import ExperimentConfig, run_experiment
from profiler_bench.runner.fidelity_run import run_fidelity_samples, score_one_seed
from profiler_bench.runner.policy import DEFAULT_POLICY
from profiler_bench.runner.scorer import FakeScorer
from profiler_bench.runner.spark_conf import fidelity_spark_config


REPO = Path(__file__).resolve().parents[3]


def _speed_profile(role: str, dataset: str, tier: str = "xs") -> dict[str, Any]:
    return {
        "kind": "speed",
        "role": role,
        "dataset": dataset,
        "tier": tier,
        "ok": True,
        "profile": {"tables": {dataset: {"role": role}}, "marker": f"{role}-{dataset}"},
    }


def test_cell_crash_leaves_other_cells_ok(tmp_path: Path) -> None:
    records = run_fidelity_samples(
        speed_records=[
            _speed_profile("baseline", "numeric"),
            _speed_profile("candidate", "numeric"),
            _speed_profile("baseline", "retail"),
            _speed_profile("candidate", "retail"),
        ],
        scorer_baseline=FakeScorer(crash_exit_code=99, fail_datasets=("numeric",)),
        scorer_candidate=FakeScorer(crash_exit_code=99, fail_datasets=("numeric",)),
        datasets=["numeric", "retail"],
        policy={"fidelity_tier": "xs", "fidelity_seeds": 2},
        samples_dir=tmp_path / "samples",
        fidelity_spark=fidelity_spark_config(driver_memory="8g"),
    )
    by_cell = {(record["role"], record["dataset"]): record for record in records}
    assert len(records) == 8
    assert all(not record["ok"] for record in records if record["dataset"] == "numeric")
    assert all(record["ok"] for record in records if record["dataset"] == "retail")
    assert all(record["error_class"] == "FidelityWorkerCrash" for record in records if record["dataset"] == "numeric")
    assert by_cell[("baseline", "retail")]["reconstruction"] == 0.8


def test_cell_raise_leaves_other_cells_ok(tmp_path: Path) -> None:
    records = run_fidelity_samples(
        speed_records=[
            _speed_profile("baseline", "numeric"),
            _speed_profile("candidate", "numeric"),
            _speed_profile("baseline", "retail"),
            _speed_profile("candidate", "retail"),
        ],
        scorer_baseline=FakeScorer(raise_message="boom", fail_datasets=("numeric",)),
        scorer_candidate=FakeScorer(raise_message="boom", fail_datasets=("numeric",)),
        datasets=["numeric", "retail"],
        policy={"fidelity_tier": "xs", "fidelity_seeds": 2},
        samples_dir=tmp_path / "samples",
        fidelity_spark=fidelity_spark_config(driver_memory="8g"),
    )
    assert all(
        not record["ok"] and record["error_class"] == "RuntimeError"
        for record in records
        if record["dataset"] == "numeric"
    )
    assert all(record["ok"] for record in records if record["dataset"] == "retail")


def test_subprocess_round_trip_preserves_score_json(tmp_path: Path) -> None:
    scorer = FakeScorer(
        reconstruction=0.81,
        structure=0.72,
        privacy_passed=True,
        ceiling_value=0.9,
        metrics_config_hash="round-trip",
        by_seed={1: {"reconstruction": 0.88}},
    )
    expected = [
        score_one_seed(
            role="baseline",
            dataset="numeric",
            tier="xs",
            seed=seed,
            profile={"keep": True},
            scorer=scorer,
            dataset_seed=0,
        )
        for seed in (0, 1)
    ]
    records = run_fidelity_samples(
        speed_records=[_speed_profile("baseline", "numeric"), _speed_profile("candidate", "numeric")],
        scorer_baseline=scorer.with_role("baseline"),
        scorer_candidate=scorer.with_role("candidate"),
        datasets=["numeric"],
        policy={"fidelity_tier": "xs", "fidelity_seeds": 2},
        samples_dir=tmp_path / "samples",
        fidelity_spark=fidelity_spark_config(driver_memory="8g"),
    )
    baseline = [record for record in records if record["role"] == "baseline"]
    assert baseline == expected
    for record in baseline:
        saved = json.loads(
            (tmp_path / "samples" / f"fidelity-baseline-numeric-xs-seed-{record['seed']}.json").read_text()
        )
        assert saved == record


def test_cells_run_in_deterministic_rotated_order(tmp_path: Path) -> None:
    records = run_fidelity_samples(
        speed_records=[
            _speed_profile("baseline", "numeric"),
            _speed_profile("candidate", "numeric"),
            _speed_profile("baseline", "retail"),
            _speed_profile("candidate", "retail"),
        ],
        scorer_baseline=FakeScorer(role="baseline"),
        scorer_candidate=FakeScorer(role="candidate"),
        datasets=["numeric", "retail"],
        policy={"fidelity_tier": "xs", "fidelity_seeds": 2},
        samples_dir=tmp_path / "samples",
        fidelity_spark=fidelity_spark_config(driver_memory="8g"),
    )
    assert [(record["role"], record["dataset"], record["seed"]) for record in records] == [
        ("baseline", "numeric", 0),
        ("baseline", "numeric", 1),
        ("candidate", "numeric", 0),
        ("candidate", "numeric", 1),
        ("candidate", "retail", 0),
        ("candidate", "retail", 1),
        ("baseline", "retail", 0),
        ("baseline", "retail", 1),
    ]


def test_run_json_records_fidelity_spark(tmp_path: Path, catalog_cache: Path) -> None:
    policy = json.loads(DEFAULT_POLICY.read_text())
    policy["speed_samples"] = {"xs": 1}
    policy["fidelity_seeds"] = 1
    policy["fidelity_tier"] = "xs"
    policy["speed_tiers"] = ["xs"]
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    out = run_experiment(
        ExperimentConfig(
            experiment_id="fidelity-spark-record",
            baseline_worktree=REPO,
            candidate_worktree=REPO,
            policy_path=policy_path,
            datasets=["numeric"],
            control=True,
            scorer=FakeScorer(),
            repo_root=tmp_path / "repo",
            cache_dir=catalog_cache,
            speed_tiers=["xs"],
            fidelity_driver_memory="8g",
        )
    )
    run = json.loads((out / "run.json").read_text())
    assert run["fidelity_spark"] == fidelity_spark_config(driver_memory="8g")
    assert run["policy"]["spark"]["configs"]["spark.driver.memory"] == "2g"
    assert run["fidelity_spark"]["configs"]["spark.driver.memory"] == "8g"
