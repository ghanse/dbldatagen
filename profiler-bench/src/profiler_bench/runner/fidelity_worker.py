"""Subprocess worker that scores one (build, dataset) fidelity cell."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

from profiler_bench.runner.fidelity_run import score_one_seed
from profiler_bench.runner.scorer import scorer_from_spec
from profiler_bench.runner.spark_conf import build_fidelity_spark


def main(argv: list[str] | None = None) -> int:
    """Scores every paired seed for one cell and writes a JSON result list."""
    parser = argparse.ArgumentParser(prog="profiler_bench.runner.fidelity_worker")
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
    args = parser.parse_args(argv)

    result: dict[str, Any] = {
        "ok": False,
        "role": args.role,
        "dataset": args.dataset,
        "tier": args.tier,
        "records": [],
    }
    spark = None
    try:
        if args.cache_dir is not None:
            os.environ["PROFILER_BENCH_CACHE"] = str(args.cache_dir)
        profile = json.loads(args.profile_json.read_text())
        scorer_spec = json.loads(args.scorer_json.read_text())
        fidelity_spark = json.loads(args.fidelity_spark_json.read_text())
        seeds = [int(seed) for seed in json.loads(args.seeds_json.read_text())]
        if not isinstance(profile, dict):
            raise TypeError("profile JSON must be an object")
        scorer = scorer_from_spec(scorer_spec)
        if str(scorer_spec.get("kind", "real")) == "real":
            spark = build_fidelity_spark(
                fidelity_spark,
                app_name=f"rsi-fidelity-{args.role}-{args.dataset}-{args.tier}",
            )
            spark.conf.set("spark.sql.session.timeZone", "UTC")
        records = [
            score_one_seed(
                role=args.role,
                dataset=args.dataset,
                tier=args.tier,
                seed=seed,
                profile=profile,
                scorer=scorer,
                dataset_seed=args.dataset_seed,
            )
            for seed in seeds
        ]
        result["records"] = records
        result["ok"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["error_class"] = type(error).__name__
        result["traceback"] = traceback.format_exc()
    finally:
        if spark is not None:
            spark.stop()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
