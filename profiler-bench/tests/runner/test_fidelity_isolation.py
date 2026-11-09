"""Fidelity cell isolation, timeouts, payload validation, and harness model source."""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

from profiler_bench.runner import fidelity_worker
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


def _write_fake_worker(path: Path, body: str) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            from __future__ import annotations
            import argparse
            import json
            import os
            import signal
            import subprocess
            import sys
            import time
            from pathlib import Path

            def _parse() -> argparse.Namespace:
                parser = argparse.ArgumentParser()
                parser.add_argument("--profile-json", type=Path, required=True)
                parser.add_argument("--scorer-json", type=Path, required=True)
                parser.add_argument("--fidelity-spark-json", type=Path, required=True)
                parser.add_argument("--out", type=Path, required=True)
                parser.add_argument("--role", required=True)
                parser.add_argument("--dataset", required=True)
                parser.add_argument("--tier", required=True)
                parser.add_argument("--seeds-json", type=Path, required=True)
                parser.add_argument("--dataset-seed", type=int, default=0)
                parser.add_argument("--cache-dir", type=Path, default=None)
                return parser.parse_args()

            """
        )
        + textwrap.dedent(body)
    )


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
    assert all(record["error_class"] == "WorkerCrashed" for record in records if record["dataset"] == "numeric")
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


def test_worker_timeout_marks_cell_and_kills_children(tmp_path: Path) -> None:
    worker = tmp_path / "sleep_worker.py"
    child_pid_file = tmp_path / "child.pid"
    _write_fake_worker(
        worker,
        f"""
        args = _parse()
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1000)"])
        Path({str(child_pid_file)!r}).write_text(str(child.pid))
        time.sleep(1000)
        """,
    )
    started = time.monotonic()
    records = run_fidelity_samples(
        speed_records=[_speed_profile("baseline", "numeric"), _speed_profile("candidate", "numeric")],
        scorer_baseline=FakeScorer(),
        scorer_candidate=FakeScorer(),
        datasets=["numeric"],
        policy={"fidelity_tier": "xs", "fidelity_seeds": 2},
        samples_dir=tmp_path / "samples",
        fidelity_spark=fidelity_spark_config(driver_memory="8g"),
        cell_timeout_s=0.4,
        kill_grace_s=0.2,
        worker_prefix=[sys.executable, str(worker)],
    )
    assert time.monotonic() - started < 5
    assert len(records) == 4
    assert all(not record["ok"] and record["error_class"] == "WorkerTimeout" for record in records)
    assert child_pid_file.is_file()
    child_pid = int(child_pid_file.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError(f"child process {child_pid} survived worker timeout")


def test_truncated_worker_json_is_worker_output_invalid(tmp_path: Path) -> None:
    worker = tmp_path / "truncated_worker.py"
    _write_fake_worker(
        worker,
        """
        args = _parse()
        args.out.write_text('{"ok": true, "records": [')
        raise SystemExit(0)
        """,
    )
    records = run_fidelity_samples(
        speed_records=[_speed_profile("baseline", "numeric"), _speed_profile("candidate", "numeric")],
        scorer_baseline=FakeScorer(),
        scorer_candidate=FakeScorer(),
        datasets=["numeric"],
        policy={"fidelity_tier": "xs", "fidelity_seeds": 1},
        samples_dir=tmp_path / "samples",
        fidelity_spark=fidelity_spark_config(driver_memory="8g"),
        worker_prefix=[sys.executable, str(worker)],
    )
    assert len(records) == 2
    assert all(record["error_class"] == "WorkerOutputInvalid" for record in records)


def test_sigkill_worker_is_worker_crashed(tmp_path: Path) -> None:
    worker = tmp_path / "sigkill_worker.py"
    _write_fake_worker(
        worker,
        """
        args = _parse()
        os.kill(os.getpid(), signal.SIGKILL)
        """,
    )
    records = run_fidelity_samples(
        speed_records=[_speed_profile("baseline", "numeric"), _speed_profile("candidate", "numeric")],
        scorer_baseline=FakeScorer(),
        scorer_candidate=FakeScorer(),
        datasets=["numeric"],
        policy={"fidelity_tier": "xs", "fidelity_seeds": 1},
        samples_dir=tmp_path / "samples",
        fidelity_spark=fidelity_spark_config(driver_memory="8g"),
        worker_prefix=[sys.executable, str(worker)],
    )
    assert len(records) == 2
    assert all(record["error_class"] == "WorkerCrashed" for record in records)


def test_payload_mismatch_does_not_overwrite_other_samples(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    samples.mkdir()
    other = samples / "fidelity-baseline-retail-xs-seed-0.json"
    other.write_text(json.dumps({"kind": "fidelity", "ok": True, "keep": True, "seed": 0}))
    worker = tmp_path / "mismatch_worker.py"
    _write_fake_worker(
        worker,
        """
        args = _parse()
        seeds = json.loads(args.seeds_json.read_text())
        payload = {
            "ok": True,
            "records": [
                {
                    "kind": "fidelity",
                    "role": "other-role",
                    "dataset": args.dataset,
                    "tier": args.tier,
                    "seed": seed,
                    "ok": True,
                    "reconstruction": 0.1,
                    "structure": 0.1,
                    "privacy_passed": True,
                    "privacy_unmeasured": [],
                    "ceiling": 1.0,
                    "relative_reconstruction": 0.1,
                    "detail": {},
                    "metrics_config_hash": "x",
                }
                for seed in seeds
            ],
        }
        args.out.write_text(json.dumps(payload))
        raise SystemExit(0)
        """,
    )
    records = run_fidelity_samples(
        speed_records=[_speed_profile("baseline", "numeric"), _speed_profile("candidate", "numeric")],
        scorer_baseline=FakeScorer(),
        scorer_candidate=FakeScorer(),
        datasets=["numeric"],
        policy={"fidelity_tier": "xs", "fidelity_seeds": 2},
        samples_dir=samples,
        fidelity_spark=fidelity_spark_config(driver_memory="8g"),
        worker_prefix=[sys.executable, str(worker)],
    )
    assert all(record["error_class"] == "WorkerOutputMismatch" for record in records)
    assert json.loads(other.read_text())["keep"] is True
    assert not (samples / "fidelity-other-role-numeric-xs-seed-0.json").exists()


def test_fidelity_worker_does_not_put_candidate_profiler_on_sys_path(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate-wt" / "profiler" / "src"
    planted = candidate / "dbldatagen_profiler"
    planted.mkdir(parents=True)
    (planted / "__init__.py").write_text("TOKEN = 'from-candidate-worktree'\n")
    (tmp_path / "profile.json").write_text("{}")
    (tmp_path / "scorer.json").write_text(json.dumps(FakeScorer().to_spec()))
    (tmp_path / "spark.json").write_text(json.dumps(fidelity_spark_config()))
    (tmp_path / "seeds.json").write_text("[0]")
    out = tmp_path / "cell.json"
    assert (
        fidelity_worker.main(
            [
                "--profile-json",
                str(tmp_path / "profile.json"),
                "--scorer-json",
                str(tmp_path / "scorer.json"),
                "--fidelity-spark-json",
                str(tmp_path / "spark.json"),
                "--seeds-json",
                str(tmp_path / "seeds.json"),
                "--out",
                str(out),
                "--role",
                "baseline",
                "--dataset",
                "numeric",
                "--tier",
                "xs",
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text())
    candidate_src = str(candidate.resolve())
    assert all(candidate_src not in entry for entry in payload["sys_path"])
    assert payload["profiler_module_file"] is not None
    assert candidate_src not in payload["profiler_module_file"]
    assert "from-candidate-worktree" not in Path(payload["profiler_module_file"]).read_text()


def test_run_json_records_fidelity_spark_timeout_and_model_source(tmp_path: Path, catalog_cache: Path) -> None:
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
            fidelity_cell_timeout_s=2700,
        )
    )
    run = json.loads((out / "run.json").read_text())
    assert run["fidelity_spark"] == fidelity_spark_config(driver_memory="8g")
    assert run["policy"]["spark"]["configs"]["spark.driver.memory"] == "2g"
    assert run["fidelity_spark"]["configs"]["spark.driver.memory"] == "8g"
    assert run["fidelity_cell_timeout_s"] == 2700
    assert run["fidelity_model_source"] == "harness"
