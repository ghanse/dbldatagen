"""`python -m profiler_bench.rsi` — run or evaluate an RSI experiment."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from profiler_bench.evaluate.evaluate import evaluate_experiment
from profiler_bench.runner.data_volume import DataVolumeConfig, calibrate_targets, run_data_volume_experiment
from profiler_bench.runner.experiment import ExperimentConfig, run_experiment
from profiler_bench.runner.policy import DEFAULT_POLICY


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatches `run`, `evaluate`, and `calibrate-targets` subcommands."""
    parser = argparse.ArgumentParser(prog="python -m profiler_bench.rsi")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Compare a baseline profiler build with a candidate.")
    run.add_argument("--baseline", type=Path, required=True, help="Worktree with baseline profiler/src.")
    run.add_argument("--candidate", type=Path, required=True, help="Worktree with candidate profiler/src.")
    run.add_argument("--policy", type=Path, default=DEFAULT_POLICY, help="Frozen policy JSON.")
    run.add_argument("--experiment-id", required=True, help="Directory name under profiler/rsi/experiments/.")
    run.add_argument("--datasets", nargs="+", default=None, help="Catalog datasets to measure.")
    run.add_argument("--control", action="store_true", help="Compare the baseline with itself.")
    run.add_argument("--repo-root", type=Path, default=None, help="Repository root for experiment output.")
    run.add_argument("--cache-dir", type=Path, default=None, help="Catalog cache directory.")
    run.add_argument("--proposal", type=Path, default=None, help="Optional proposal.json to copy.")
    run.add_argument("--speed-tiers", nargs="+", default=None, help="Override speed tiers.")
    run.add_argument(
        "--no-baseline-cache",
        action="store_true",
        help="Measure the baseline build fresh instead of reusing cached baseline samples.",
    )
    run.add_argument(
        "--mode",
        choices=("default", "data-volume"),
        default="default",
        help="RSI objective: default accuracy/speed, or data-volume (least input fraction).",
    )
    run.add_argument("--fractions", nargs="+", type=float, default=None, help="Data-volume fraction grid.")
    run.add_argument("--subset-seeds", type=int, default=None, help="Data-volume subset seed count S.")
    run.add_argument("--fidelity-seeds", type=int, default=None, help="Sampler seed count K.")
    run.set_defaults(handler=_run)

    evaluate = sub.add_parser("evaluate", help="Re-evaluate saved samples without re-running.")
    evaluate.add_argument("experiment_dir", type=Path, help="Path to profiler/rsi/experiments/<id>.")
    evaluate.set_defaults(handler=_evaluate)

    calibrate = sub.add_parser(
        "calibrate-targets",
        help="Compute T_d from a baseline run at f=1 and write a proposed data_volume policy.",
    )
    calibrate.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Experiment directory whose f=1 relative scores define T_d.",
    )
    calibrate.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the proposed policy JSON (polly freezes it).",
    )
    calibrate.add_argument(
        "--base-policy",
        type=Path,
        default=None,
        help="Policy template to copy before filling targets (default: policies/data_volume.json).",
    )
    calibrate.add_argument("--margin", type=float, default=0.01, help="T_d = mean_relative - margin.")
    calibrate.set_defaults(handler=_calibrate)

    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.handler(args))


def _run(args: argparse.Namespace) -> int:
    proposal = None
    if args.proposal is not None:
        proposal = json.loads(Path(args.proposal).read_text())
    if args.mode == "data-volume":
        path = run_data_volume_experiment(
            DataVolumeConfig(
                experiment_id=args.experiment_id,
                baseline_worktree=args.baseline,
                candidate_worktree=args.candidate,
                policy_path=args.policy,
                datasets=args.datasets,
                control=args.control,
                repo_root=args.repo_root,
                cache_dir=args.cache_dir,
                proposal=proposal,
                command_line=sys.argv,
                fractions=args.fractions,
                subset_seeds=args.subset_seeds,
                fidelity_seeds=args.fidelity_seeds,
            )
        )
    else:
        path = run_experiment(
            ExperimentConfig(
                experiment_id=args.experiment_id,
                baseline_worktree=args.baseline,
                candidate_worktree=args.candidate,
                policy_path=args.policy,
                datasets=args.datasets,
                control=args.control,
                repo_root=args.repo_root,
                cache_dir=args.cache_dir,
                proposal=proposal,
                speed_tiers=args.speed_tiers,
                command_line=sys.argv,
                baseline_cache=not args.no_baseline_cache,
            )
        )
    verdict = json.loads((path / "verdict.json").read_text())
    print(json.dumps({"experiment_dir": str(path), "verdict": verdict["verdict"]}, sort_keys=True))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    verdict = evaluate_experiment(args.experiment_dir)
    print(json.dumps({"experiment_dir": str(args.experiment_dir), "verdict": verdict["verdict"]}, sort_keys=True))
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    policy = calibrate_targets(
        baseline_experiment=args.baseline,
        output_policy=args.output,
        base_policy=args.base_policy,
        margin=float(args.margin),
    )
    print(
        json.dumps(
            {"output": str(args.output), "targets": {name: value["T_d"] for name, value in policy["targets"].items()}},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
