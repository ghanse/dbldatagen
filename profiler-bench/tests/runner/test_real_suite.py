"""The real-dataset suite in the runner: suite selection, offline skips, and real cells next to synthetic ones."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from profiler_bench.catalog import DATASETS, REAL_DATASETS
from profiler_bench.catalog.real import (
    REAL_SUBSET_CAP,
    REAL_SUBSET_METHOD,
    SPLIT,
    WAREHOUSE_ENV,
    RealDataset,
    RealTable,
    instance_dir,
)
from profiler_bench.catalog.store import write_instance
from profiler_bench.runner import baseline_cache, measure
from profiler_bench.runner.experiment import ExperimentConfig, prepare_real, run_experiment, suite_datasets
from profiler_bench.runner.identity import BuildIdentity
from profiler_bench.runner.policy import DEFAULT_POLICY, load_policy
from profiler_bench.runner.scorer import FakeScorer


REAL = "nyctaxi_trips"
TRIPS = RealDataset(
    name=REAL,
    description="A small stand-in for the trips table.",
    license="none",
    terms="none",
    tables=(RealTable("trips", "x.y.trips", 300, "fare double, zip bigint"),),
)


def test_suite_datasets() -> None:
    assert suite_datasets("synthetic", None) == (sorted(DATASETS), [])
    assert suite_datasets("real", None) == ([], sorted(REAL_DATASETS))
    assert suite_datasets("all", ["numeric", REAL]) == (["numeric"], [REAL])
    with pytest.raises(ValueError, match="not in the synthetic suite"):
        suite_datasets("synthetic", [REAL])
    with pytest.raises(ValueError, match="unknown suite"):
        suite_datasets("everything", None)


def test_uncached_real_datasets_offline_are_skipped_with_a_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(WAREHOUSE_ENV, raising=False)
    ready, skipped, reports = prepare_real(sorted(REAL_DATASETS), "s", tmp_path)
    assert (ready, reports) == ([], {})
    assert set(skipped) == set(REAL_DATASETS)
    assert all(reason.startswith("Unavailable:") and WAREHOUSE_ENV in reason for reason in skipped.values())
    assert prepare_real([REAL], "xs", tmp_path)[1][REAL].startswith("KeyError:")


def test_real_cache_key_includes_pins_and_split() -> None:
    identity = baseline_cache.dataset_identity(REAL, "s", 0, None)
    assert identity == {
        "real_pinned_hashes": dict(REAL_DATASETS[REAL].hashes),
        "real_split": SPLIT,
        "real_subset_cap": REAL_SUBSET_CAP,
        "real_subset_method": REAL_SUBSET_METHOD,
    }


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.delenv(WAREHOUSE_ENV, raising=False)
    monkeypatch.setitem(REAL_DATASETS, REAL, TRIPS)
    catalog = tmp_path / "catalog"
    rng = np.random.default_rng(0)
    trips = {"trips": pa.table({"fare": rng.normal(size=300), "zip": rng.integers(0, 50, 300)})}
    target = instance_dir(TRIPS, "full", catalog)
    target.mkdir(parents=True)
    write_instance(target, trips, TRIPS.manifest("full", trips), 0)

    speed: list[tuple[str, str]] = []

    def fake_one_sample(
        *, build: BuildIdentity, role: str, dataset: str, tier: str, sample_index: int, samples_dir: Path, **_: object
    ) -> dict[str, Any]:
        record = {
            "kind": "speed",
            "role": role,
            "dataset": dataset,
            "tier": tier,
            "sample_index": sample_index,
            "ok": True,
            "wall_s": 1.0,
            "cpu_s": 1.0,
            "peak_rss_bytes": 100,
            "profile_hash": f"{build.profiler_src_hash}-{dataset}",
            "profile": {"dataset": dataset},
        }
        measure._write_record(samples_dir, record)
        speed.append((dataset, tier))
        return record

    monkeypatch.setattr(measure, "_one_sample", fake_one_sample)
    policy = load_policy(DEFAULT_POLICY)
    policy.update(fidelity_tier="s", speed_tiers=["xs"], fidelity_seeds=2)
    policy["bootstrap"]["samples"] = 50
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    for name, source in (("baseline", "A = 1\n"), ("candidate", "A = 2\n")):
        (tmp_path / name / "profiler" / "src").mkdir(parents=True)
        (tmp_path / name / "profiler" / "src" / "mod.py").write_text(source)
    return {"tmp": tmp_path, "catalog": catalog, "speed": speed}


def _run(setup: dict[str, Any], experiment_id: str, **overrides: Any) -> tuple[Path, dict[str, Any]]:  # noqa: ANN401
    tmp = setup["tmp"]
    fields: dict[str, Any] = {
        "experiment_id": experiment_id,
        "baseline_worktree": tmp / "baseline",
        "candidate_worktree": tmp / "candidate",
        "policy_path": tmp / "policy.json",
        "scorer": FakeScorer(),
        "repo_root": tmp / "repo",
        "cache_dir": setup["catalog"],
        "baseline_cache_dir": tmp / "baseline-cache",
        "suite": "real",
        "datasets": [REAL],
    }
    fields.update(overrides)
    out = run_experiment(ExperimentConfig(**fields))
    return out, json.loads((out / "run.json").read_text())


def test_real_suite_scores_cached_dataset_as_a_holdout(setup: dict[str, Any]) -> None:
    out, run = _run(setup, "real")
    assert run["suite"] == "real"
    assert (run["datasets"], run["real_datasets"], run["real_skipped"]) == ([], [REAL], {})
    assert run["real_split"] == SPLIT
    assert run["real_split_report"][REAL]["trips"]["profiled"]["copied"] == 0
    assert set(setup["speed"]) == {(REAL, "s")}
    records = [json.loads(p.read_text()) for p in sorted((out / "samples").glob(f"fidelity-*-{REAL}-*.json"))]
    assert len(records) == 4
    assert all(r["ok"] and r["structure"] is None and r["structure_status"] == "not_applicable" for r in records)
    verdict = json.loads((out / "verdict.json").read_text())
    gate = next(g for g in verdict["gates"] if g["name"] == "real_holdout")
    assert gate["reason"] == "real_holdout_passed"
    assert verdict["verdict"] != "accepted"


def test_real_suite_skips_uncached_dataset_and_still_runs(setup: dict[str, Any]) -> None:
    out, run = _run(setup, "skip", datasets=["wanderbricks"])
    assert run["real_datasets"] == []
    assert run["real_skipped"]["wanderbricks"].startswith("Unavailable:")
    assert setup["speed"] == []
    verdict = json.loads((out / "verdict.json").read_text())
    assert next(g for g in verdict["gates"] if g["name"] == "real_holdout")["reason"] == "real_holdout_skipped"


def test_synthetic_suite_ignores_real_datasets(setup: dict[str, Any]) -> None:
    policy = json.loads((setup["tmp"] / "policy.json").read_text())
    policy.update(fidelity_tier="xs")
    (setup["tmp"] / "policy.json").write_text(json.dumps(policy))
    out, run = _run(setup, "synthetic", suite="synthetic", datasets=["numeric"])
    assert run["suite"] == "synthetic"
    assert not any(key.startswith("real_") for key in run)
    assert {dataset for dataset, _ in setup["speed"]} == {"numeric"}
    verdict = json.loads((out / "verdict.json").read_text())
    assert "real_holdout" not in [g["name"] for g in verdict["gates"]]
    cached = [json.loads(p.read_text())["components"] for p in (setup["tmp"] / "baseline-cache").glob("*.json")]
    assert cached and all("catalog_content_hash" in c and "real_split" not in c for c in cached)


def test_cli_passes_suite_through(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from profiler_bench.rsi import __main__ as cli  # noqa: PLC0415

    seen: list[str] = []

    def fake_run(config: ExperimentConfig) -> Path:
        seen.append(config.suite)
        (tmp_path / "verdict.json").write_text(json.dumps({"verdict": "inconclusive"}))
        return tmp_path

    monkeypatch.setattr(cli, "run_experiment", fake_run)
    base = ["run", "--baseline", ".", "--candidate", ".", "--experiment-id", "x"]
    assert cli.main([*base, "--suite", "real"]) == 0
    assert cli.main(base) == 0
    assert seen == ["real", "synthetic"]
    with pytest.raises(SystemExit):
        cli.main([*base, "--suite", "every"])
