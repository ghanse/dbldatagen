"""Data-volume runner: fake curves, completeness, and a small real xs smoke."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from profiler_bench.runner.data_volume import DataVolumeConfig, calibrate_targets, run_data_volume_experiment
from profiler_bench.runner.scorer import FakeScorer
from profiler_bench.runner.subset import apply_table_scales


REPO = Path(__file__).resolve().parents[3]


def test_run_fails_up_front_without_frozen_targets(tmp_path: Path) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "bootstrap": {"samples": 200, "seed": 0, "confidence": 0.95},
                "fractions": [1.0],
                "subset_seeds": 1,
                "fidelity_seeds": 1,
                "fidelity_tier": "xs",
                "targets": {"numeric": {"T_d": None, "provenance": "unset"}},
                "data_volume": {"control_noise_width": {}},
                "dataset_seed": 0,
                "spark": {"master": "local[1]", "configs": {"spark.ui.enabled": "false"}},
                "not_measured": [],
            }
        )
    )
    with pytest.raises(ValueError, match="missing frozen calibration"):
        run_data_volume_experiment(
            DataVolumeConfig(
                experiment_id="dv-missing-targets",
                baseline_worktree=REPO,
                candidate_worktree=REPO,
                policy_path=policy,
                datasets=["numeric"],
                control=True,
                scorer=FakeScorer(reconstruction=0.9, structure=0.5, privacy_passed=True, ceiling_value=1.0),
                repo_root=tmp_path,
                fractions=[1.0],
                subset_seeds=1,
                fidelity_seeds=1,
            )
        )


def test_calibrate_targets_freezes_noise_width(tmp_path: Path) -> None:
    experiment = tmp_path / "exp"
    samples = experiment / "samples"
    samples.mkdir(parents=True)
    (experiment / "run.json").write_text(
        json.dumps(
            {
                "experiment_id": "dv-control-f1",
                "control": True,
                "baseline": {"commit": "abc123", "profiler_src_hash": "prof"},
                "harness": {"commit": "harness1", "harness_hash": "hh"},
                "policy": {"bootstrap": {"samples": 200, "seed": 0, "confidence": 0.95}},
            }
        )
    )
    for seed in range(4):
        record = {
            "kind": "fidelity",
            "role": "baseline",
            "dataset": "numeric",
            "tier": "xs",
            "fraction": 1.0,
            "subset_seed": 0,
            "seed": seed,
            "ok": True,
            "relative_reconstruction": 0.9 + 0.01 * (seed % 2),
            "privacy_passed": True,
        }
        (samples / f"fidelity-baseline-numeric-xs-f-1-subset-0-seed-{seed}.json").write_text(json.dumps(record) + "\n")
    out = tmp_path / "proposed.json"
    policy = calibrate_targets(baseline_experiment=experiment, output_policy=out, margin=0.01)
    assert policy["targets"]["numeric"]["T_d"] is not None
    assert "numeric" in policy["data_volume"]["control_noise_width"]
    assert policy["data_volume"]["control_noise_width_provenance"]["run_id"] == "dv-control-f1"
    assert policy["data_volume"]["control_noise_width_provenance"]["harness_hash"] == "hh"
    assert policy["data_volume"]["f1_regression_bound"]["source"] == "default.json"
    assert "accuracy" not in policy


def test_nstar_on_fake_scorer_curve(tmp_path: Path) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "bootstrap": {"samples": 200, "seed": 0, "confidence": 0.95},
                "fractions": [0.1, 0.5, 1.0],
                "subset_seeds": 1,
                "fidelity_seeds": 2,
                "fidelity_tier": "xs",
                "targets": {"numeric": {"T_d": 0.80, "provenance": "test"}},
                "data_volume": {"control_noise_width": {"numeric": 0.0}},
                "accuracy": {"max_dataset_regression": 0.05},
                "dataset_seed": 0,
                "spark": {"master": "local[1]", "configs": {"spark.ui.enabled": "false"}},
                "not_measured": [],
            }
        )
    )
    # Both roles share a curve that reaches at 0.5; control-like, so not accepted.
    scorer = FakeScorer(
        reconstruction=0.7,
        structure=0.5,
        privacy_passed=True,
        ceiling_value=1.0,
        relative_by_fraction={0.1: 0.5, 0.5: 0.9, 1.0: 0.95},
    )
    out = run_data_volume_experiment(
        DataVolumeConfig(
            experiment_id="dv-fake-curve",
            baseline_worktree=REPO,
            candidate_worktree=REPO,
            policy_path=policy,
            datasets=["numeric"],
            control=True,
            scorer=scorer,
            repo_root=tmp_path,
            fractions=[0.1, 0.5, 1.0],
            subset_seeds=1,
            fidelity_seeds=2,
            proposal={"type": "data_volume", "targets": ["numeric"]},
        )
    )
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["verdict"] != "accepted"
    assert "control" in verdict["reasons"]
    assert verdict["n_star"]["baseline"]["numeric"]["n_star"] == 0.5
    samples = list((out / "samples").glob("fidelity-*.json"))
    assert len(samples) == 12  # 2 roles * 3 fractions * 1 subset * 2 seeds
    run = json.loads((out / "run.json").read_text())
    assert run["mode"] == "data-volume"
    assert run["subset_method"] == "fk_preserving_uniform"


def test_apply_table_scales_matches_sampler_contract() -> None:
    profile = {"tables": [{"name": "customers", "row_count": 50}, {"name": "orders", "row_count": 200}]}
    scales = {"customers": 20.0, "orders": 5.0}
    scaled = apply_table_scales(profile, scales)
    assert scaled["tables"][0]["row_count"] == 1000
    assert scaled["tables"][1]["row_count"] == 1000


@pytest.mark.slow
def test_data_volume_xs_smoke(tmp_path: Path, catalog_cache: Path) -> None:
    """Short real smoke: 2 datasets, short grid, K = S = 1."""
    policy = tmp_path / "policy.json"
    # Provisional T_d below xs control relative means so f=1 reaches; n* search still runs.
    policy.write_text(
        json.dumps(
            {
                "bootstrap": {"samples": 200, "seed": 0, "confidence": 0.95},
                "fractions": [0.2, 1.0],
                "subset_seeds": 1,
                "fidelity_seeds": 1,
                "fidelity_tier": "xs",
                "targets": {
                    "numeric": {
                        "T_d": 0.50,
                        "provenance": "smoke provisional: below xs control relative mean",
                    },
                    "retail": {
                        "T_d": 0.40,
                        "provenance": "smoke provisional: below xs control relative mean",
                    },
                },
                "data_volume": {"control_noise_width": {"numeric": 0.05, "retail": 0.05}},
                "accuracy": {"max_dataset_regression": 0.05},
                "dataset_seed": 0,
                "spark": {
                    "master": "local[2]",
                    "configs": {
                        "spark.sql.shuffle.partitions": "2",
                        "spark.sql.session.timeZone": "UTC",
                        "spark.ui.enabled": "false",
                        "spark.driver.memory": "2g",
                    },
                },
                "not_measured": [],
            }
        )
    )
    out = run_data_volume_experiment(
        DataVolumeConfig(
            experiment_id="dv-xs-smoke",
            baseline_worktree=REPO,
            candidate_worktree=REPO,
            policy_path=policy,
            datasets=["numeric", "retail"],
            control=True,
            repo_root=tmp_path,
            cache_dir=catalog_cache,
            fractions=[0.2, 1.0],
            subset_seeds=1,
            fidelity_seeds=1,
            proposal={"type": "data_volume", "targets": ["numeric", "retail"]},
        )
    )
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["verdict"] != "accepted"
    curves = verdict["n_star"]["baseline"]
    assert "numeric" in curves and "retail" in curves
    samples = [json.loads(path.read_text()) for path in (out / "samples").glob("fidelity-*.json")]
    assert len(samples) == 8  # 2 roles * 2 datasets * 2 fractions * 1 * 1
    assert all(record["ok"] for record in samples)
    assert all(record.get("scales") for record in samples)
    walls = {
        f"{record['dataset']}@f={record['fraction']}": record.get("profile_wall_s")
        for record in samples
        if record["role"] == "baseline"
    }
    for _dataset, result in curves.items():
        assert result["curve"]
        assert result["n_star"] in {0.2, 1.0}
    # Emit smoke artifacts for the delivery report (pytest log + on-disk copy).
    curves_json = json.dumps(curves, sort_keys=True, default=str)
    walls_json = json.dumps(walls, sort_keys=True, default=str)
    (tmp_path / "p3-2-smoke-curves.json").write_text(curves_json + "\n")
    (tmp_path / "p3-2-smoke-walls.json").write_text(walls_json + "\n")
    print("P3.2_SMOKE_CURVES=" + curves_json)
    print("P3.2_SMOKE_WALLS=" + walls_json)
