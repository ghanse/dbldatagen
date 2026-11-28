"""P1.6f: screening stage, real subset cap, and Spark session reuse."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from pyspark.sql import SparkSession

from dbldatagen_profiler.engines.spark import profile_tables
from profiler_bench.catalog import load
from profiler_bench.catalog.real import REAL_SUBSET_CAP, RealDataset, RealTable, subset
from profiler_bench.evaluate.evaluate import evaluate_experiment
from profiler_bench.evaluate.gates import evaluate_gates
from profiler_bench.rsi import __main__ as rsi_cli
from profiler_bench.runner import fidelity_worker
from profiler_bench.runner.experiment import ExperimentConfig, run_experiment
from profiler_bench.runner.fidelity_run import run_fidelity_samples, score_one_seed
from profiler_bench.runner.policy import (
    DEFAULT_POLICY,
    apply_screen_overrides,
    load_policy,
    screened_accuracy_thresholds,
)
from profiler_bench.runner.scorer import FakeScorer, FidelityScores, try_real_scorer
from profiler_bench.runner.spark_conf import fidelity_spark_config


REPO = Path(__file__).resolve().parents[3]


def _speed(role: str, dataset: str, tier: str, index: int = 0) -> dict[str, Any]:
    return {
        "kind": "speed",
        "role": role,
        "dataset": dataset,
        "tier": tier,
        "sample_index": index,
        "ok": True,
        "wall_s": 1.0,
        "cpu_s": 1.0,
        "peak_rss_bytes": 100,
        "profile": {"tables": {dataset: {"role": role}}},
    }


def _fidelity(
    role: str,
    dataset: str,
    tier: str,
    seed: int,
    *,
    relative: float = 0.9,
    structure: float = 0.7,
    privacy: bool = True,
    ok: bool = True,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": "fidelity",
        "role": role,
        "dataset": dataset,
        "tier": tier,
        "seed": seed,
        "ok": ok,
        "reconstruction": relative * 0.9,
        "structure": structure,
        "relative_reconstruction": relative,
        "discriminator_gap": 1.0 - relative,
        "privacy_passed": privacy,
        "ceiling": 0.9,
    }
    if not ok:
        record["error_class"] = "RuntimeError"
        record["error"] = "boom"
    return record


def _run_meta(*, stage: str = "official", tier: str = "s", seeds: int = 5, datasets: list[str] | None = None) -> dict:
    names = datasets or ["numeric"]
    policy = load_policy(DEFAULT_POLICY)
    if stage == "screen":
        policy = apply_screen_overrides(policy)
        tier = str(policy["fidelity_tier"])
    return {
        "experiment_id": "screen-test",
        "control": False,
        "stage": stage,
        "datasets": names,
        "speed_tiers": [tier],
        "speed_samples_effective": {tier: 2},
        "fidelity_tier": tier,
        "baseline": {"profiler_src_hash": "aaaa"},
        "candidate": {"profiler_src_hash": "bbbb"},
        "policy": policy,
    }


def test_screen_thresholds_apply_margin() -> None:
    policy = load_policy(DEFAULT_POLICY)
    screened = screened_accuracy_thresholds(policy)
    assert screened["max_dataset_regression"] == pytest.approx(0.015)
    assert screened["max_structure_regression"] == pytest.approx(0.03)
    assert screened["ci_lower_must_exceed"] == 0.0
    discriminator = screened_accuracy_thresholds(policy, section="discriminator")
    assert discriminator["min_mean_improvement"] == pytest.approx(0.005)
    assert discriminator["min_target_improvement"] == pytest.approx(0.01)
    assert discriminator["max_dataset_regression"] == pytest.approx(0.03)
    assert discriminator["max_group_regression"] == pytest.approx(0.03)


def test_apply_screen_overrides_sets_tier_and_seeds() -> None:
    policy = apply_screen_overrides(load_policy(DEFAULT_POLICY))
    assert policy["fidelity_tier"] == "r"
    assert policy["fidelity_seeds"] == 3
    assert policy["speed_tiers"] == ["r"]
    assert policy["speed_samples_accuracy"]["r"] == 2
    assert policy["discriminator"]["min_mean_improvement"] == pytest.approx(0.005)


def test_screen_pass_never_accepted() -> None:
    run = _run_meta(stage="screen")
    policy = run["policy"]
    speed = [_speed(role, "numeric", "r", i) for role in ("baseline", "candidate") for i in range(2)]
    fidelity = []
    for seed in range(3):
        fidelity.append(_fidelity("baseline", "numeric", "r", seed, relative=0.80))
        fidelity.append(_fidelity("candidate", "numeric", "r", seed, relative=0.85))
    verdict = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=policy,
        proposal={"type": "accuracy"},
        run=run,
    )
    assert verdict["verdict"] == "screen_pass"
    assert verdict["stage"] == "screen"
    assert verdict["verdict"] != "accepted"


def test_screen_fail_on_insufficient_accuracy() -> None:
    run = _run_meta(stage="screen")
    policy = run["policy"]
    speed = [_speed(role, "numeric", "r", i) for role in ("baseline", "candidate") for i in range(2)]
    fidelity = []
    for seed in range(3):
        fidelity.append(_fidelity("baseline", "numeric", "r", seed, relative=0.90))
        fidelity.append(_fidelity("candidate", "numeric", "r", seed, relative=0.9001))
    verdict = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=policy,
        proposal={"type": "accuracy"},
        run=run,
    )
    assert verdict["verdict"] == "screen_fail"


def test_screen_privacy_failure_is_screen_fail() -> None:
    run = _run_meta(stage="screen")
    policy = run["policy"]
    speed = [_speed(role, "numeric", "r", i) for role in ("baseline", "candidate") for i in range(2)]
    fidelity = []
    for seed in range(3):
        fidelity.append(_fidelity("baseline", "numeric", "r", seed, relative=0.80))
        fidelity.append(_fidelity("candidate", "numeric", "r", seed, relative=0.90, privacy=False))
    verdict = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=policy,
        proposal={"type": "accuracy"},
        run=run,
    )
    assert verdict["verdict"] == "screen_fail"
    assert "privacy_guard_failed" in verdict["reasons"]


def test_screen_margin_lets_borderline_through() -> None:
    """Official discriminator threshold rejects +0.007; screen margin 0.5 accepts it as screen_pass."""
    run = _run_meta(stage="screen")
    policy = run["policy"]
    speed = [_speed(role, "numeric", "r", i) for role in ("baseline", "candidate") for i in range(2)]
    fidelity = []
    for seed in range(3):
        fidelity.append(_fidelity("baseline", "numeric", "r", seed, relative=0.80))
        fidelity.append(_fidelity("candidate", "numeric", "r", seed, relative=0.807))
    screen = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=policy,
        proposal={"type": "accuracy"},
        run=run,
    )
    official_run = dict(run)
    official_run["stage"] = "official"
    official_policy = load_policy(DEFAULT_POLICY)
    official = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=official_policy,
        proposal={"type": "accuracy"},
        run=official_run,
    )
    assert official["verdict"] == "rejected"
    assert screen["verdict"] == "screen_pass"


def test_old_run_reevaluates_under_recorded_policy(tmp_path: Path) -> None:
    experiment = tmp_path / "old-run"
    samples = experiment / "samples"
    samples.mkdir(parents=True)
    old_policy = load_policy(DEFAULT_POLICY)
    old_policy["version"] = "2.1.0"
    del old_policy["screen"]
    run = {
        "experiment_id": "old-run",
        "control": True,
        "datasets": ["numeric"],
        "speed_tiers": ["s"],
        "speed_samples_effective": {"s": 2},
        "fidelity_tier": "s",
        "baseline": {"profiler_src_hash": "same"},
        "candidate": {"profiler_src_hash": "same"},
        "policy": old_policy,
    }
    (experiment / "run.json").write_text(json.dumps(run))
    (experiment / "proposal.json").write_text(json.dumps({"type": "accuracy"}))
    for index in range(2):
        for role in ("baseline", "candidate"):
            (samples / f"speed-{role}-numeric-s-{index}.json").write_text(
                json.dumps(_speed(role, "numeric", "s", index))
            )
    for seed in range(5):
        for role in ("baseline", "candidate"):
            (samples / f"fidelity-{role}-numeric-s-seed-{seed}.json").write_text(
                json.dumps(_fidelity(role, "numeric", "s", seed))
            )
    verdict = evaluate_experiment(experiment)
    assert verdict["verdict"] != "accepted"
    assert "screen" not in verdict


def test_screen_cli_and_stage_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[ExperimentConfig] = []

    def fake_run(config: ExperimentConfig) -> Path:
        calls.append(config)
        out = tmp_path / config.experiment_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "verdict.json").write_text(json.dumps({"verdict": "screen_pass"}))
        return out

    monkeypatch.setattr("profiler_bench.rsi.__main__.run_experiment", fake_run)
    assert (
        rsi_cli.main(
            [
                "screen",
                "--baseline",
                str(tmp_path),
                "--candidate",
                str(tmp_path),
                "--experiment-id",
                "via-screen",
            ]
        )
        == 0
    )
    assert calls[-1].stage == "screen"
    assert (
        rsi_cli.main(
            [
                "run",
                "--stage",
                "screen",
                "--baseline",
                str(tmp_path),
                "--candidate",
                str(tmp_path),
                "--experiment-id",
                "via-stage",
            ]
        )
        == 0
    )
    assert calls[-1].stage == "screen"


def test_screen_experiment_records_stage(tmp_path: Path) -> None:
    policy = load_policy(DEFAULT_POLICY)
    policy["screen"] = {
        "tier": "xs",
        "fidelity_seeds": 3,
        "margin": 0.5,
        "speed_samples": {"xs": 2},
        "note": "test screen uses xs for speed",
    }
    policy["speed_samples"]["xs"] = 2
    policy["speed_samples_accuracy"]["xs"] = 2
    policy_path = tmp_path / "screen-policy.json"
    policy_path.write_text(json.dumps(policy))
    out_root = tmp_path / "repo"
    (out_root / "profiler" / "rsi" / "experiments").mkdir(parents=True)
    out = run_experiment(
        ExperimentConfig(
            experiment_id="screen-exp",
            baseline_worktree=REPO,
            candidate_worktree=REPO,
            policy_path=policy_path,
            datasets=["numeric"],
            control=True,
            scorer=FakeScorer(),
            repo_root=out_root,
            cache_dir=tmp_path / "catalog",
            stage="screen",
            baseline_cache=False,
        )
    )
    run = json.loads((out / "run.json").read_text())
    verdict = json.loads((out / "verdict.json").read_text())
    assert run["stage"] == "screen"
    assert run["fidelity_tier"] == "xs"
    assert run["policy"]["fidelity_seeds"] == 3
    assert run["speed_samples_effective"]["xs"] == 2
    assert verdict["verdict"] in {"screen_pass", "screen_fail"}
    assert verdict["verdict"] != "accepted"


def test_real_subset_cap_is_50k() -> None:
    assert REAL_SUBSET_CAP == 50_000
    dataset = RealDataset(
        name="cap",
        description="cap",
        license="none",
        terms="none",
        tables=(RealTable("t", "x.t", 200_000, "id bigint", ("id",)),),
    )
    tables = {"t": pa.table({"id": list(range(200_000))})}
    kept = subset(dataset, tables, REAL_SUBSET_CAP)
    assert kept["t"].num_rows == pytest.approx(REAL_SUBSET_CAP, rel=0.05)
    assert kept["t"].num_rows < 75_000


def _speed_profile(role: str, dataset: str, tier: str = "xs") -> dict[str, Any]:
    return {
        "kind": "speed",
        "role": role,
        "dataset": dataset,
        "tier": tier,
        "ok": True,
        "profile": {"tables": {dataset: {"role": role}}, "marker": f"{role}-{dataset}"},
    }


def test_session_reuse_matches_fresh_session_scores(tmp_path: Path) -> None:
    scorer = FakeScorer(
        reconstruction=0.81,
        structure=0.72,
        by_seed={0: {"reconstruction": 0.80}, 1: {"reconstruction": 0.82}},
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
    reused = run_fidelity_samples(
        speed_records=[_speed_profile("baseline", "numeric"), _speed_profile("candidate", "numeric")],
        scorer_baseline=scorer.with_role("baseline"),
        scorer_candidate=scorer.with_role("candidate"),
        datasets=["numeric"],
        policy={"fidelity_tier": "xs", "fidelity_seeds": 2},
        samples_dir=tmp_path / "reuse",
        fidelity_spark=fidelity_spark_config(driver_memory="8g"),
    )
    fresh = run_fidelity_samples(
        speed_records=[_speed_profile("baseline", "numeric"), _speed_profile("candidate", "numeric")],
        scorer_baseline=scorer.with_role("baseline"),
        scorer_candidate=scorer.with_role("candidate"),
        datasets=["numeric"],
        policy={"fidelity_tier": "xs", "fidelity_seeds": 2},
        samples_dir=tmp_path / "fresh",
        fidelity_spark=fidelity_spark_config(driver_memory="8g"),
        worker_prefix=[sys.executable, "-c", _fresh_worker_launcher()],
    )
    baseline_reused = [record for record in reused if record["role"] == "baseline"]
    baseline_fresh = [record for record in fresh if record["role"] == "baseline"]
    assert baseline_reused == expected
    assert baseline_fresh == expected


def _fresh_worker_launcher() -> str:
    return (
        "import os, runpy, sys;"
        "os.environ['PROFILER_BENCH_FIDELITY_SESSION']='fresh';"
        "sys.argv=['profiler_bench.runner.fidelity_worker', *sys.argv[1:]];"
        "runpy.run_module('profiler_bench.runner.fidelity_worker', run_name='__main__')"
    )


def test_seed_raise_still_records_per_seed_failures(tmp_path: Path) -> None:
    boom = FakeScorer(raise_message="seed-boom", raise_seeds=(1,))
    records = run_fidelity_samples(
        speed_records=[_speed_profile("baseline", "numeric"), _speed_profile("candidate", "numeric")],
        scorer_baseline=boom.with_role("baseline"),
        scorer_candidate=boom.with_role("candidate"),
        datasets=["numeric"],
        policy={"fidelity_tier": "xs", "fidelity_seeds": 2},
        samples_dir=tmp_path / "samples",
        fidelity_spark=fidelity_spark_config(driver_memory="8g"),
    )
    baseline = [record for record in records if record["role"] == "baseline"]
    assert len(baseline) == 2
    assert baseline[0]["ok"] is True
    assert baseline[1]["ok"] is False
    assert baseline[1]["error_class"] == "RuntimeError"
    assert "seed-boom" in baseline[1]["error"]


def test_reused_session_restores_non_timezone_conf(spark: SparkSession) -> None:
    """A seed that mutates runtime SQL conf must not leak those settings into the next seed."""
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    spark.conf.set("spark.sql.shuffle.partitions", "2")
    original_shuffle = spark.conf.get("spark.sql.shuffle.partitions")
    marker_key = "spark.sql.profiler_bench.test.marker"
    with contextlib.suppress(Exception):
        spark.conf.unset(marker_key)
    assert spark.conf.get(marker_key, None) is None

    seen: dict[str, str | None] = {}

    class ConfMutatingScorer:
        def score(
            self,
            profile: dict[str, Any],
            *,
            dataset: str,
            tier: str,
            seed: int,
            structure_seed: int,
            sample_seed: int,
        ) -> FidelityScores:
            del profile, dataset, tier, structure_seed, sample_seed
            active = SparkSession.getActiveSession()
            assert active is not None
            if seed == 0:
                active.conf.set("spark.sql.shuffle.partitions", "99")
                active.conf.set(marker_key, "leaked")
            else:
                seen["shuffle"] = active.conf.get("spark.sql.shuffle.partitions")
                seen["marker"] = active.conf.get(marker_key, None)
            return FidelityScores(
                reconstruction=0.8,
                structure=0.7,
                privacy_passed=True,
                ceiling=0.9,
                relative_reconstruction=0.8 / 0.9,
                privacy_unmeasured=[],
                detail={},
                metrics_config_hash="conf-mutate",
            )

        def ceiling(self, dataset: str, tier: str, *, structure_seed: int = 0) -> float:
            del dataset, tier, structure_seed
            return 0.9

    records, session_ids, _held = fidelity_worker._score_seeds(
        role="baseline",
        dataset="numeric",
        tier="xs",
        seeds=[0, 1],
        profile={"tables": {}},
        scorer=ConfMutatingScorer(),
        dataset_seed=0,
        fidelity_spark=fidelity_spark_config(driver_memory="8g"),
        use_spark=True,
        session_mode=fidelity_worker.SESSION_REUSE,
    )
    assert len(records) == 2 and all(record["ok"] for record in records)
    assert len(set(session_ids)) == 1
    assert seen["shuffle"] == original_shuffle
    assert seen["marker"] is None
    assert spark.conf.get("spark.sql.shuffle.partitions") == original_shuffle
    assert spark.conf.get(marker_key, None) is None


def _numeric_xs_profile(spark: SparkSession) -> dict[str, Any]:
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    real = load("numeric", "xs", seed=0, spark=spark, sample_seed=0)
    dumped = profile_tables(real, detect_pii=False).model_dump(mode="json")
    assert isinstance(dumped, dict)
    return dumped


def test_real_spark_session_reuse_matches_fresh_score_json(tmp_path: Path, spark: SparkSession) -> None:
    """Reuse vs fresh must be byte-identical with a real scorer (Spark exercised, cache cleared between seeds)."""
    profile = _numeric_xs_profile(spark)
    cache = tmp_path / "ceilings"
    scorer = try_real_scorer(cache_dir=cache, harness_hash="screen-reuse", metrics_config_hash="screen")
    assert scorer is not None
    speed = [
        {"kind": "speed", "role": role, "dataset": "numeric", "tier": "xs", "ok": True, "profile": profile}
        for role in ("baseline", "candidate")
    ]
    policy = {"fidelity_tier": "xs", "fidelity_seeds": 3}
    spark_conf = fidelity_spark_config(driver_memory="8g")
    reused = run_fidelity_samples(
        speed_records=speed,
        scorer_baseline=scorer,
        scorer_candidate=scorer,
        datasets=["numeric"],
        policy=policy,
        samples_dir=tmp_path / "reuse",
        fidelity_spark=spark_conf,
        cache_dir=cache,
        harness_hash="screen-reuse",
        metrics_config_hash="screen",
        parallelism=1,
    )
    fresh = run_fidelity_samples(
        speed_records=speed,
        scorer_baseline=scorer,
        scorer_candidate=scorer,
        datasets=["numeric"],
        policy=policy,
        samples_dir=tmp_path / "fresh",
        fidelity_spark=spark_conf,
        cache_dir=cache,
        harness_hash="screen-reuse",
        metrics_config_hash="screen",
        parallelism=1,
        worker_prefix=[sys.executable, "-c", _fresh_worker_launcher()],
    )
    baseline_reused = [record for record in reused if record["role"] == "baseline"]
    baseline_fresh = [record for record in fresh if record["role"] == "baseline"]
    assert len(baseline_reused) == 3 and len(baseline_fresh) == 3
    assert all(record["ok"] for record in baseline_reused)
    assert all(record["ok"] for record in baseline_fresh)
    assert json.dumps(baseline_reused, sort_keys=True, separators=(",", ":")) == json.dumps(
        baseline_fresh, sort_keys=True, separators=(",", ":")
    )


def test_real_spark_seed_raise_continues_remaining_seeds(
    tmp_path: Path, spark: SparkSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One seed raising under a reused real Spark session must not drop the other seeds' records."""
    profile = _numeric_xs_profile(spark)
    scorer = try_real_scorer(cache_dir=tmp_path / "ceilings", harness_hash="screen-raise", metrics_config_hash="screen")
    assert scorer is not None
    real_score = scorer.score

    def score_or_raise(profile_arg: dict[str, Any], **kwargs: Any) -> Any:  # noqa: ANN401
        if kwargs.get("seed") == 1:
            raise RuntimeError("seed-boom-real")
        return real_score(profile_arg, **kwargs)

    monkeypatch.setattr(scorer, "score", score_or_raise)
    # Reuse the session-scoped Spark via getOrCreate; do not stop it.
    records, session_ids, _held = fidelity_worker._score_seeds(
        role="baseline",
        dataset="numeric",
        tier="xs",
        seeds=[0, 1, 2],
        profile=profile,
        scorer=scorer,
        dataset_seed=0,
        fidelity_spark=fidelity_spark_config(driver_memory="8g"),
        use_spark=True,
        session_mode=fidelity_worker.SESSION_REUSE,
    )
    assert len(records) == 3
    assert len(set(session_ids)) == 1
    assert records[0]["ok"] is True
    assert records[1]["ok"] is False
    assert records[1]["error_class"] == "RuntimeError"
    assert "seed-boom-real" in records[1]["error"]
    assert records[2]["ok"] is True
