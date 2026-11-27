"""`python -m profiler_bench.rsi` — run or evaluate an RSI experiment."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from profiler_bench.evaluate.evaluate import evaluate_experiment
from profiler_bench.runner.data_volume import DataVolumeConfig, calibrate_targets, run_data_volume_experiment
from profiler_bench.runner.experiment import SUITES, ExperimentConfig, run_experiment
from profiler_bench.runner.policy import DEFAULT_POLICY


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatches `run`, `screen`, `evaluate`, and `calibrate-targets` subcommands."""
    parser = argparse.ArgumentParser(prog="python -m profiler_bench.rsi")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Compare a baseline profiler build with a candidate.")
    _add_compare_args(run)
    run.add_argument(
        "--stage",
        choices=("official", "screen"),
        default="official",
        help="official (default) or screen (tier-r screening; accuracy only; synthetic only).",
    )
    run.add_argument(
        "--suite",
        choices=SUITES,
        default="synthetic",
        help="Datasets to score: the synthetic catalog (default), the real holdout datasets, or all.",
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

    screen = sub.add_parser(
        "screen",
        help="Run the screening stage (tier r, 3 seeds, synthetic, accuracy only).",
    )
    _add_compare_args(screen)
    screen.set_defaults(handler=_screen)

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


def _add_compare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline", type=Path, required=True, help="Worktree with baseline profiler/src.")
    parser.add_argument("--candidate", type=Path, required=True, help="Worktree with candidate profiler/src.")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY, help="Frozen policy JSON.")
    parser.add_argument("--experiment-id", required=True, help="Directory name under profiler/rsi/experiments/.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Catalog datasets to measure.")
    parser.add_argument("--control", action="store_true", help="Compare the baseline with itself.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Repository root for experiment output.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Catalog cache directory.")
    parser.add_argument("--proposal", type=Path, default=None, help="Optional proposal.json to copy.")
    parser.add_argument("--speed-tiers", nargs="+", default=None, help="Override speed tiers.")
    parser.add_argument(
        "--no-baseline-cache",
        action="store_true",
        help="Measure the baseline build fresh instead of reusing cached baseline samples.",
    )


def _run(args: argparse.Namespace) -> int:
    proposal = _load_proposal(args.proposal)
    stage = getattr(args, "stage", "official")
    suite = getattr(args, "suite", "synthetic")
    if args.mode == "data-volume":
        if stage == "screen":
            raise SystemExit("screening stage does not support --mode data-volume")
        if suite != "synthetic":
            raise SystemExit("--suite applies only to the default mode")
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
        if stage == "screen" and suite != "synthetic":
            raise SystemExit("screening stage is synthetic-only")
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
                stage=stage,
                suite=suite,
            )
        )
    verdict = json.loads((path / "verdict.json").read_text())
    print(json.dumps({"experiment_dir": str(path), "verdict": verdict["verdict"]}, sort_keys=True))
    return 0


def _screen(args: argparse.Namespace) -> int:
    args.stage = "screen"
    args.suite = "synthetic"
    args.mode = "default"
    args.fractions = None
    args.subset_seeds = None
    args.fidelity_seeds = None
    return _run(args)


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


def _load_proposal(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"proposal must be a JSON object: {path}")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
