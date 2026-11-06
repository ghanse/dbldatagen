"""`python -m profiler_bench.rsi` — run or evaluate an RSI experiment."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from profiler_bench.evaluate.evaluate import evaluate_experiment
from profiler_bench.runner.experiment import ExperimentConfig, run_experiment
from profiler_bench.runner.policy import DEFAULT_POLICY


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatches `run` and `evaluate` subcommands."""
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
    run.set_defaults(handler=_run)

    evaluate = sub.add_parser("evaluate", help="Re-evaluate saved samples without re-running.")
    evaluate.add_argument("experiment_dir", type=Path, help="Path to profiler/rsi/experiments/<id>.")
    evaluate.set_defaults(handler=_evaluate)

    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.handler(args))


def _run(args: argparse.Namespace) -> int:
    proposal = None
    if args.proposal is not None:
        proposal = json.loads(Path(args.proposal).read_text())
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
        )
    )
    verdict = json.loads((path / "verdict.json").read_text())
    print(json.dumps({"experiment_dir": str(path), "verdict": verdict["verdict"]}, sort_keys=True))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    verdict = evaluate_experiment(args.experiment_dir)
    print(json.dumps({"experiment_dir": str(args.experiment_dir), "verdict": verdict["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
