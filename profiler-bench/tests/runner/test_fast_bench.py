"""P1.6c: accuracy smoke speed N, baseline cache, parallel fidelity cells, and policy 2.0.0."""

from __future__ import annotations

import itertools
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from profiler_bench.catalog import REAL_DATASETS
from profiler_bench.catalog.real import SPLIT
from profiler_bench.evaluate.evaluate import evaluate_experiment
from profiler_bench.evaluate.gates import evaluate_gates, expected_sample_keys
from profiler_bench.rsi import __main__ as rsi_cli
from profiler_bench.runner import baseline_cache, experiment, fidelity_run, measure
from profiler_bench.runner.experiment import ExperimentConfig, run_experiment
from profiler_bench.runner.fidelity_run import run_fidelity_samples
from profiler_bench.runner.identity import BuildIdentity
from profiler_bench.runner.policy import DEFAULT_POLICY, effective_speed_samples, load_policy
from profiler_bench.runner.scorer import FakeScorer
from profiler_bench.runner.spark_conf import fidelity_spark_config


DATASETS = ["numeric", "retail"]


def test_default_policy_is_2_1_0_with_fast_defaults() -> None:
    policy = load_policy(DEFAULT_POLICY)
    assert policy["version"] == "2.1.0"
    assert policy["accuracy"]["min_target_improvement"] == 0.01
    assert policy["spark"]["master"] == "local[8]"
    assert policy["spark"]["configs"] == {
        "spark.sql.shuffle.partitions": "2",
        "spark.sql.session.timeZone": "UTC",
        "spark.ui.enabled": "false",
        "spark.driver.memory": "8g",
    }
    assert policy["speed_samples"] == {"xs": 2, "s": 15, "m": 15}
    assert policy["speed_samples_accuracy"] == {"xs": 2, "s": 2, "m": 0}
    assert policy["fidelity_parallelism"] == 3


def test_effective_speed_samples_by_proposal_type() -> None:
    policy = load_policy(DEFAULT_POLICY)
    assert effective_speed_samples(policy, "accuracy") == {"xs": 2, "s": 2, "m": 0}
    assert effective_speed_samples(policy, "speed") == {"xs": 2, "s": 15, "m": 15}
    assert effective_speed_samples(policy, "data_volume") == {"xs": 2, "s": 15, "m": 15}
    del policy["speed_samples_accuracy"]
    assert effective_speed_samples(policy, "accuracy") == {"xs": 2, "s": 15, "m": 15}


def _speed(role: str, dataset: str, tier: str, index: int) -> dict[str, Any]:
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
    }


def _fidelity(role: str, dataset: str, tier: str, seed: int) -> dict[str, Any]:
    return {
        "kind": "fidelity",
        "role": role,
        "dataset": dataset,
        "tier": tier,
        "seed": seed,
        "ok": True,
        "reconstruction": 0.8,
        "structure": 0.7,
        "relative_reconstruction": 0.9,
        "privacy_passed": True,
    }


def _matrix(n_speed: int, k: int = 2) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    roles = ("baseline", "candidate")
    speed = [_speed(role, "numeric", "s", index) for role in roles for index in range(n_speed)]
    fidelity = [_fidelity(role, "numeric", "s", seed) for role in roles for seed in range(k)]
    return speed, fidelity


def _completeness(run: dict[str, Any], proposal_type: str, n_speed: int) -> dict[str, Any]:
    speed, fidelity = _matrix(n_speed, int(run["policy"]["fidelity_seeds"]))
    verdict = evaluate_gates(
        speed_records=speed,
        fidelity_records=fidelity,
        policy=run["policy"],
        proposal={"type": proposal_type, "targets": ["numeric"]},
        run=run,
    )
    return next(gate for gate in verdict["gates"] if gate["name"] == "sample_completeness")


def _run_json(policy: dict[str, Any], proposal_type: str | None) -> dict[str, Any]:
    run: dict[str, Any] = {
        "policy": {**policy, "fidelity_seeds": 2, "bootstrap": {"samples": 50, "seed": 0, "confidence": 0.95}},
        "datasets": ["numeric"],
        "speed_tiers": ["s", "m"],
        "fidelity_tier": "s",
    }
    if proposal_type is not None:
        run["speed_samples_effective"] = effective_speed_samples(policy, proposal_type)
    return run


def test_accuracy_completeness_passes_with_smoke_speed_n() -> None:
    run = _run_json(load_policy(DEFAULT_POLICY), "accuracy")
    assert run["speed_samples_effective"]["s"] == 2
    assert _completeness(run, "accuracy", 2)["status"] == "pass"


def test_speed_proposal_still_needs_full_n() -> None:
    run = _run_json(load_policy(DEFAULT_POLICY), "speed")
    gate = _completeness(run, "speed", 2)
    assert gate["status"] == "fail"
    assert gate["reason"] == "failed_or_missing_samples"
    assert "speed/candidate/numeric/s/14" in gate["detail"]["missing"]


def test_old_1_1_0_run_json_keeps_its_recorded_policy_n() -> None:
    old_policy = load_policy(DEFAULT_POLICY)
    old_policy["version"] = "1.1.0"
    del old_policy["speed_samples_accuracy"]
    run = _run_json(old_policy, None)
    expected = expected_sample_keys(run)
    assert len(expected["speed"]) == 2 * (15 + 15)
    assert _completeness(run, "accuracy", 2)["status"] == "fail"


def test_old_1_1_0_experiment_re_evaluates_with_its_n(tmp_path: Path) -> None:
    old_policy = load_policy(DEFAULT_POLICY)
    old_policy.update(version="1.1.0", speed_samples={"s": 3}, speed_tiers=["s"], fidelity_seeds=2)
    old_policy["bootstrap"]["samples"] = 50
    del old_policy["speed_samples_accuracy"]
    root = tmp_path / "old"
    (root / "samples").mkdir(parents=True)
    run = {"policy": old_policy, "datasets": ["numeric"], "speed_tiers": ["s"], "fidelity_tier": "s"}
    (root / "run.json").write_text(json.dumps(run))
    (root / "proposal.json").write_text(json.dumps({"type": "accuracy", "targets": ["numeric"]}))
    speed, fidelity = _matrix(3)
    for index, record in enumerate([*speed, *fidelity]):
        (root / "samples" / f"{index}.json").write_text(json.dumps(record))
    verdict = evaluate_experiment(root)
    gate = next(gate for gate in verdict["gates"] if gate["name"] == "sample_completeness")
    assert gate["status"] == "pass"
    assert gate["detail"]["expected_speed"] == 6


# Baseline cache: unit level.


def _components(**overrides: object) -> dict[str, Any]:
    components: dict[str, Any] = {
        "kind": "fidelity",
        "baseline_profiler_src_hash": "src",
        "harness_hash": "harness",
        "policy_hash": "policy",
        "metrics_config_hash": "metrics",
        "dataset": "numeric",
        "tier": "s",
        "dataset_seed": 0,
        "spark_hash": "spark",
        "sampler_seeds": [0, 1],
        "scorer_hash": "scorer",
        "catalog_fingerprint": "fingerprint",
        "catalog_content_hash": "content",
        "real_pinned_hashes": {"t": "pinned"},
        "environment": {
            "python": "3.10.20",
            "pyspark": "4.1.1",
            "numpy": "2.0",
            "pandas": "2.2",
            "scikit-learn": "1.5",
            "pyarrow": "17",
            "java": "17",
        },
    }
    components.update(overrides)
    return components


def _store(root: Path, components: dict[str, Any]) -> None:
    baseline_cache.store_entry(
        root,
        components,
        source_experiment_id="exp-1",
        records=[_fidelity("baseline", "numeric", "s", seed) for seed in (0, 1)],
        profile={"p": 1},
    )


def test_cache_round_trip(tmp_path: Path) -> None:
    _store(tmp_path, _components())
    entry = baseline_cache.load_entry(tmp_path, _components())
    assert entry is not None
    assert entry["source_experiment_id"] == "exp-1"
    assert entry["profile"] == {"p": 1}
    assert [record["seed"] for record in entry["records"]] == [0, 1]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline_profiler_src_hash", "other"),
        ("harness_hash", "other"),
        ("policy_hash", "other"),
        ("metrics_config_hash", "other"),
        ("dataset_seed", 1),
        ("spark_hash", "other"),
        ("scorer_hash", "other"),
        ("sampler_seeds", [0, 2]),
        ("catalog_fingerprint", "other"),
        ("catalog_content_hash", "other"),
        ("real_pinned_hashes", {"t": "repinned"}),
        *[
            (f"environment.{name}", "other")
            for name in ("python", "pyspark", "numpy", "pandas", "scikit-learn", "pyarrow", "java")
        ],
    ],
)
def test_cache_misses_on_each_key_component(tmp_path: Path, field: str, value: object) -> None:
    _store(tmp_path, _components())
    changed = _components()
    if field.startswith("environment."):
        changed["environment"] = {**changed["environment"], field.split(".", 1)[1]: value}
    else:
        changed[field] = value
    assert baseline_cache.load_entry(tmp_path, changed) is None


def test_environment_records_python_libraries_and_java() -> None:
    assert set(baseline_cache.environment(java=False)) == {
        "python",
        "pyspark",
        "numpy",
        "pandas",
        "scikit-learn",
        "pyarrow",
    }
    assert "java" in baseline_cache.environment(java=True)


def test_dataset_identity_uses_catalog_hashes_and_real_pins(tmp_path: Path) -> None:
    synthetic = baseline_cache.dataset_identity("numeric", "xs", 0, tmp_path)
    assert set(synthetic) == {"catalog_fingerprint", "catalog_content_hash"}
    assert baseline_cache.dataset_identity("numeric", "xs", 1, tmp_path) != synthetic
    real = next(iter(REAL_DATASETS.values()))
    assert baseline_cache.dataset_identity(real.name, "s", 0, None) == {
        "real_pinned_hashes": dict(real.hashes),
        "real_split": SPLIT,
    }


def test_cache_misses_on_dataset_and_tier(tmp_path: Path) -> None:
    _store(tmp_path, _components())
    assert baseline_cache.load_entry(tmp_path, _components(dataset="retail")) is None
    assert baseline_cache.load_entry(tmp_path, _components(tier="m")) is None


def test_cache_rejects_corrupt_and_tampered_entries(tmp_path: Path) -> None:
    components = _components()
    _store(tmp_path, components)
    path = baseline_cache.entry_path(tmp_path, components)
    entry = json.loads(path.read_text())
    entry["components"]["harness_hash"] = "tampered"
    path.write_text(json.dumps(entry))
    assert baseline_cache.load_entry(tmp_path, components) is None
    path.write_text('{"truncated": ')
    assert baseline_cache.load_entry(tmp_path, components) is None


def test_cache_never_stores_failed_or_incomplete_sets(tmp_path: Path) -> None:
    components = _components()
    failed = _fidelity("baseline", "numeric", "s", 1) | {"ok": False}
    baseline_cache.store_entry(
        tmp_path, components, source_experiment_id="x", records=[_fidelity("baseline", "numeric", "s", 0), failed]
    )
    baseline_cache.store_entry(
        tmp_path, components, source_experiment_id="x", records=[_fidelity("baseline", "numeric", "s", 0)]
    )
    assert not baseline_cache.entry_path(tmp_path, components).exists()


# Experiment level: fake speed samples (no Spark), FakeScorer fidelity workers.


@pytest.fixture
def speed_log(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    log: list[dict[str, Any]] = []

    def fake_one_sample(
        *,
        build: BuildIdentity,
        role: str,
        dataset: str,
        tier: str,
        sample_index: int,
        order: tuple[str, str],
        samples_dir: Path,
        **_: object,
    ) -> dict[str, Any]:
        start = time.monotonic()
        time.sleep(0.01)
        record = {
            **_speed(role, dataset, tier, sample_index),
            "order": list(order),
            "build_profiler_src_hash": build.profiler_src_hash,
            "profile_hash": f"{build.profiler_src_hash}-{dataset}",
            "profile": {"dataset": dataset, "build": build.profiler_src_hash},
        }
        measure._write_record(samples_dir, record)
        log.append({"role": role, "dataset": dataset, "start": start, "end": time.monotonic()})
        return record

    monkeypatch.setattr(measure, "_one_sample", fake_one_sample)
    return log


@pytest.fixture
def fidelity_log(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    log: list[dict[str, Any]] = []
    real = fidelity_run._run_worker_command

    def timed(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:  # noqa: ANN401
        start = time.monotonic()
        result = real(command, **kwargs)
        role = command[command.index("--role") + 1]
        dataset = command[command.index("--dataset") + 1]
        log.append({"role": role, "dataset": dataset, "start": start, "end": time.monotonic()})
        return result

    monkeypatch.setattr(fidelity_run, "_run_worker_command", timed)
    return log


def _worktree(path: Path, source: str) -> Path:
    (path / "profiler" / "src").mkdir(parents=True)
    (path / "profiler" / "src" / "mod.py").write_text(source)
    return path


@pytest.fixture
def setup(tmp_path: Path) -> dict[str, Path]:
    policy = load_policy(DEFAULT_POLICY)
    policy.update(fidelity_tier="xs", speed_tiers=["xs"], fidelity_seeds=2)
    policy["bootstrap"]["samples"] = 50
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    return {
        "policy": policy_path,
        "baseline": _worktree(tmp_path / "baseline", "A = 1\n"),
        "candidate": _worktree(tmp_path / "candidate", "A = 2\n"),
        "repo": tmp_path / "repo",
        "cache": tmp_path / "baseline-cache",
    }


def _run(setup: dict[str, Path], experiment_id: str, **overrides: object) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "experiment_id": experiment_id,
        "baseline_worktree": setup["baseline"],
        "candidate_worktree": setup["candidate"],
        "policy_path": setup["policy"],
        "datasets": DATASETS,
        "scorer": FakeScorer(),
        "repo_root": setup["repo"],
        "baseline_cache_dir": setup["cache"],
        "proposal": {"type": "accuracy", "targets": DATASETS},
    }
    fields.update(overrides)
    out = run_experiment(ExperimentConfig(**fields))
    run: dict[str, Any] = json.loads((out / "run.json").read_text())
    return run


def _roles(log: list[dict[str, Any]]) -> list[str]:
    return sorted({entry["role"] for entry in log})


def test_accuracy_run_records_effective_n_and_passes_completeness(
    setup: dict[str, Path], speed_log: list[dict[str, Any]], fidelity_log: list[dict[str, Any]]
) -> None:
    run = _run(setup, "acc")
    assert run["speed_samples_effective"] == {"xs": 2}
    assert run["fidelity_parallelism"] == 3
    assert run["fidelity_spark"]["master"] == "local[4]"
    assert run["fidelity_spark"]["configs"]["spark.driver.memory"] == "8g"
    assert run["baseline_cached"] is False
    assert len(speed_log) == 2 * 2 * len(DATASETS)
    verdict = json.loads((setup["repo"] / "profiler/rsi/experiments/acc/verdict.json").read_text())
    gate = next(gate for gate in verdict["gates"] if gate["name"] == "sample_completeness")
    assert gate["status"] == "pass"


def test_speed_never_overlaps_fidelity(
    setup: dict[str, Path], speed_log: list[dict[str, Any]], fidelity_log: list[dict[str, Any]]
) -> None:
    _run(setup, "overlap")
    assert len(fidelity_log) == 2 * len(DATASETS)
    assert max(entry["end"] for entry in speed_log) <= min(entry["start"] for entry in fidelity_log)
    ordered = sorted(speed_log, key=lambda entry: entry["start"])
    assert all(a["end"] <= b["start"] for a, b in itertools.pairwise(ordered))


def test_accuracy_cache_hit_reuses_baseline_speed_and_fidelity(
    setup: dict[str, Path], speed_log: list[dict[str, Any]], fidelity_log: list[dict[str, Any]]
) -> None:
    _run(setup, "first")
    first = {
        path.name: path.read_bytes()
        for path in (setup["repo"] / "profiler/rsi/experiments/first/samples").glob("*-baseline-*")
    }
    speed_log.clear()
    fidelity_log.clear()
    run = _run(setup, "second")
    assert run["baseline_cached"] is True
    assert run["baseline_cache_parts"] == {"speed_cache_allowed": True, "speed_cached": True, "fidelity_cached": True}
    assert run["baseline_cache_sources"] == ["first"]
    assert len(run["baseline_cache_hits"]) == 2 * len(DATASETS)
    assert _roles(speed_log) == ["candidate"]
    assert _roles(fidelity_log) == ["candidate"]
    second = {
        path.name: path.read_bytes()
        for path in (setup["repo"] / "profiler/rsi/experiments/second/samples").glob("*-baseline-*")
    }
    assert second == first


@pytest.mark.parametrize("proposal_type", ["speed", "data_volume"])
def test_non_accuracy_runs_time_fresh_paired_baselines_with_warm_cache(
    setup: dict[str, Path],
    speed_log: list[dict[str, Any]],
    fidelity_log: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    proposal_type: str,
) -> None:
    monkeypatch.setattr(experiment, "evaluate_experiment", lambda out: {})
    proposal = {"type": proposal_type, "targets": DATASETS}
    _run(setup, "warm-accuracy")
    _run(setup, "warm-same-type", proposal=proposal)
    speed_log.clear()
    fidelity_log.clear()
    run = _run(setup, "fresh-speed", proposal=proposal)
    assert run["baseline_cache_parts"] == {"speed_cache_allowed": False, "speed_cached": False, "fidelity_cached": True}
    assert all(hit["case"].startswith("fidelity/") for hit in run["baseline_cache_hits"])
    assert _roles(fidelity_log) == ["candidate"]
    samples = setup["repo"] / "profiler/rsi/experiments/fresh-speed/samples"
    for dataset in DATASETS:
        runs = [entry["role"] for entry in speed_log if entry["dataset"] == dataset]
        assert sorted(runs) == ["baseline", "baseline", "candidate", "candidate"]
        orders = {
            tuple(json.loads((samples / f"speed-candidate-{dataset}-xs-{index}.json").read_text())["order"])
            for index in range(2)
        }
        assert orders == {("baseline", "candidate"), ("candidate", "baseline")}
    assert not [path for path in setup["cache"].glob("speed-*.json") if "fresh-speed" in path.read_text()]


def test_environment_change_misses_cache(
    setup: dict[str, Path],
    speed_log: list[dict[str, Any]],
    fidelity_log: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run(setup, "first")
    real = baseline_cache.environment
    monkeypatch.setattr(experiment, "environment", lambda *, java: {**real(java=java), "numpy": "0.0"})
    speed_log.clear()
    run = _run(setup, "new-numpy")
    assert run["baseline_cached"] is False
    assert "baseline" in _roles(speed_log)


def test_catalog_content_change_misses_cache(
    setup: dict[str, Path],
    speed_log: list[dict[str, Any]],
    fidelity_log: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run(setup, "first")
    real = baseline_cache.dataset_identity
    monkeypatch.setattr(
        experiment,
        "dataset_identity",
        lambda *args: {**real(*args), "catalog_content_hash": "regenerated"},
    )
    assert _run(setup, "new-catalog")["baseline_cached"] is False


def test_baseline_change_misses_cache(
    setup: dict[str, Path], speed_log: list[dict[str, Any]], fidelity_log: list[dict[str, Any]], tmp_path: Path
) -> None:
    _run(setup, "first")
    speed_log.clear()
    run = _run(setup, "other-baseline", baseline_worktree=_worktree(tmp_path / "baseline2", "A = 3\n"))
    assert run["baseline_cached"] is False
    assert "baseline" in _roles(speed_log)


def test_policy_change_misses_cache(
    setup: dict[str, Path], speed_log: list[dict[str, Any]], fidelity_log: list[dict[str, Any]], tmp_path: Path
) -> None:
    _run(setup, "first")
    policy = json.loads(setup["policy"].read_text())
    policy["spark"]["configs"]["spark.driver.memory"] = "4g"
    changed = tmp_path / "changed-policy.json"
    changed.write_text(json.dumps(policy))
    assert _run(setup, "changed", policy_path=changed)["baseline_cached"] is False


def test_gate_threshold_change_hits_cache(
    setup: dict[str, Path], speed_log: list[dict[str, Any]], fidelity_log: list[dict[str, Any]], tmp_path: Path
) -> None:
    _run(setup, "first")
    policy = json.loads(setup["policy"].read_text())
    policy["accuracy"]["min_target_improvement"] = 0.02
    changed = tmp_path / "changed-gate-policy.json"
    changed.write_text(json.dumps(policy))
    run = _run(setup, "changed-gate", policy_path=changed)
    assert run["baseline_cache_parts"] == {
        "speed_cache_allowed": True,
        "speed_cached": True,
        "fidelity_cached": True,
    }


def test_sample_count_change_misses_cache(
    setup: dict[str, Path], speed_log: list[dict[str, Any]], fidelity_log: list[dict[str, Any]], tmp_path: Path
) -> None:
    _run(setup, "first")
    policy = json.loads(setup["policy"].read_text())
    policy["speed_samples_accuracy"]["s"] += 1
    changed = tmp_path / "changed-samples-policy.json"
    changed.write_text(json.dumps(policy))
    assert _run(setup, "changed-samples", policy_path=changed)["baseline_cached"] is False


def test_no_baseline_cache_recomputes(
    setup: dict[str, Path], speed_log: list[dict[str, Any]], fidelity_log: list[dict[str, Any]]
) -> None:
    _run(setup, "first")
    speed_log.clear()
    run = _run(setup, "fresh", baseline_cache=False)
    assert run["baseline_cache_enabled"] is False
    assert run["baseline_cached"] is False
    assert "baseline" in _roles(speed_log)


def test_control_never_reuses_cache_for_candidate(
    setup: dict[str, Path], speed_log: list[dict[str, Any]], fidelity_log: list[dict[str, Any]]
) -> None:
    _run(setup, "first", control=True)
    speed_log.clear()
    fidelity_log.clear()
    run = _run(setup, "control", control=True)
    assert run["baseline_cached"] is True
    assert _roles(speed_log) == ["candidate"]
    assert _roles(fidelity_log) == ["candidate"]
    assert len(speed_log) == 2 * len(DATASETS)


def test_corrupt_cache_entry_is_recomputed(
    setup: dict[str, Path], speed_log: list[dict[str, Any]], fidelity_log: list[dict[str, Any]]
) -> None:
    _run(setup, "first")
    entries = sorted(setup["cache"].glob("*.json"))
    assert entries
    for path in entries:
        path.write_text("{not json")
    speed_log.clear()
    run = _run(setup, "after-corrupt")
    assert run["baseline_cached"] is False
    assert "baseline" in _roles(speed_log)
    assert all(json.loads(path.read_text())["source_experiment_id"] == "after-corrupt" for path in entries)


def test_cli_no_baseline_cache_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[ExperimentConfig] = []

    def fake_run(config: ExperimentConfig) -> Path:
        seen.append(config)
        (tmp_path / "verdict.json").write_text(json.dumps({"verdict": "inconclusive"}))
        return tmp_path

    monkeypatch.setattr(rsi_cli, "run_experiment", fake_run)
    base = ["run", "--baseline", "a", "--candidate", "b", "--experiment-id", "x"]
    rsi_cli.main(base)
    rsi_cli.main([*base, "--no-baseline-cache"])
    assert [config.baseline_cache for config in seen] == [True, False]


# Parallel fidelity cells.


def _profiles(datasets: list[str]) -> list[dict[str, Any]]:
    return [
        {"kind": "speed", "role": role, "dataset": dataset, "tier": "xs", "ok": True, "profile": {"d": dataset}}
        for dataset in datasets
        for role in ("baseline", "candidate")
    ]


def _fidelity_run(samples_dir: Path, parallelism: int, scorer: FakeScorer) -> list[dict[str, Any]]:
    datasets = ["numeric", "retail", "strings"]
    return run_fidelity_samples(
        speed_records=_profiles(datasets),
        scorer_baseline=scorer.with_role("baseline"),
        scorer_candidate=scorer.with_role("candidate"),
        datasets=datasets,
        policy={"fidelity_tier": "xs", "fidelity_seeds": 2},
        samples_dir=samples_dir,
        fidelity_spark=fidelity_spark_config(),
        parallelism=parallelism,
    )


def test_parallel_fidelity_equals_serial(tmp_path: Path) -> None:
    scorer = FakeScorer(by_role_seed={("candidate", 1): {"reconstruction": 0.85}})
    serial = _fidelity_run(tmp_path / "serial", 1, scorer)
    parallel = _fidelity_run(tmp_path / "parallel", 3, scorer)
    assert parallel == serial
    serial_files = {path.name: path.read_bytes() for path in (tmp_path / "serial").iterdir()}
    parallel_files = {path.name: path.read_bytes() for path in (tmp_path / "parallel").iterdir()}
    assert len(serial_files) == 12
    assert parallel_files == serial_files


def test_parallel_crashed_cell_leaves_others_ok(tmp_path: Path) -> None:
    records = _fidelity_run(tmp_path / "samples", 3, FakeScorer(crash_exit_code=7, fail_datasets=("retail",)))
    assert len(records) == 12
    assert all(record["error_class"] == "WorkerCrashed" for record in records if record["dataset"] == "retail")
    assert all(record["ok"] for record in records if record["dataset"] != "retail")
