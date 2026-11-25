"""Subprocess worker that scores one (build, dataset) fidelity cell."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession

from profiler_bench.runner.fidelity_run import score_one_seed
from profiler_bench.runner.scorer import Scorer, scorer_from_spec
from profiler_bench.runner.spark_conf import build_fidelity_spark


SESSION_REUSE = "reuse"
SESSION_FRESH = "fresh"


def main(argv: list[str] | None = None) -> int:
    """Scores every paired seed for one cell and writes a JSON result list.

    By default all seeds share one SparkSession (`session_mode=reuse`). Set
    `--session-mode fresh` or `PROFILER_BENCH_FIDELITY_SESSION=fresh` to create a
    new session per seed (measurement / equivalence checks only). Speed samples
    are unchanged and always use a fresh process per sample.
    """
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
    parser.add_argument(
        "--session-mode",
        choices=(SESSION_REUSE, SESSION_FRESH),
        default=os.environ.get("PROFILER_BENCH_FIDELITY_SESSION", SESSION_REUSE),
        help="reuse one SparkSession across seeds (default) or create a fresh session per seed.",
    )
    args = parser.parse_args(argv)

    result: dict[str, Any] = {
        "ok": False,
        "role": args.role,
        "dataset": args.dataset,
        "tier": args.tier,
        "records": [],
        "session_mode": args.session_mode,
        "session_ids": [],
        "profiler_module_file": _harness_profiler_module_file(),
        "sys_path": list(sys.path),
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
        records, session_ids, spark = _score_seeds(
            role=args.role,
            dataset=args.dataset,
            tier=args.tier,
            seeds=seeds,
            profile=profile,
            scorer=scorer_from_spec(scorer_spec),
            dataset_seed=args.dataset_seed,
            fidelity_spark=fidelity_spark,
            use_spark=str(scorer_spec.get("kind", "real")) == "real",
            session_mode=args.session_mode,
        )
        result["records"] = records
        result["session_ids"] = session_ids
        result["ok"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["error_class"] = type(error).__name__
        result["traceback"] = traceback.format_exc()
    finally:
        if spark is not None:
            with contextlib.suppress(Exception):
                spark.stop()
        result["profiler_module_file"] = _harness_profiler_module_file()
        result["sys_path"] = list(sys.path)
        _atomic_write_json(args.out, result)
    return 0 if result.get("ok") else 1


def _score_seeds(
    *,
    role: str,
    dataset: str,
    tier: str,
    seeds: list[int],
    profile: dict[str, Any],
    scorer: Scorer,
    dataset_seed: int,
    fidelity_spark: dict[str, Any],
    use_spark: bool,
    session_mode: str,
) -> tuple[list[dict[str, Any]], list[str], SparkSession | None]:
    """Scores every seed, reusing one SparkSession unless `session_mode` is fresh."""
    if use_spark and session_mode == SESSION_REUSE:
        spark = build_fidelity_spark(
            fidelity_spark,
            app_name=f"rsi-fidelity-{role}-{dataset}-{tier}",
        )
        spark.conf.set("spark.sql.session.timeZone", "UTC")
        records = [
            score_one_seed(
                role=role,
                dataset=dataset,
                tier=tier,
                seed=seed,
                profile=profile,
                scorer=scorer,
                dataset_seed=dataset_seed,
            )
            for seed in seeds
        ]
        return records, [str(id(spark))] * len(seeds), spark

    records = []
    session_ids: list[str] = []
    for seed in seeds:
        seed_spark = None
        try:
            if use_spark:
                seed_spark = build_fidelity_spark(
                    fidelity_spark,
                    app_name=f"rsi-fidelity-{role}-{dataset}-{tier}-seed-{seed}",
                )
                seed_spark.conf.set("spark.sql.session.timeZone", "UTC")
                session_ids.append(str(id(seed_spark)))
            else:
                session_ids.append("none")
            records.append(
                score_one_seed(
                    role=role,
                    dataset=dataset,
                    tier=tier,
                    seed=seed,
                    profile=profile,
                    scorer=scorer,
                    dataset_seed=dataset_seed,
                )
            )
        finally:
            if seed_spark is not None:
                with contextlib.suppress(Exception):
                    seed_spark.stop()
    return records, session_ids, None


def _harness_profiler_module_file() -> str | None:
    """Returns the frozen-harness dbldatagen_profiler module path when importable."""
    try:
        import dbldatagen_profiler  # noqa: PLC0415
    except ImportError:
        return None
    if dbldatagen_profiler.__file__ is None:
        return None
    return str(Path(dbldatagen_profiler.__file__).resolve())


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Writes JSON via a temp file + os.replace so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, sort_keys=True, default=str))
    os.replace(tmp_path, path)


if __name__ == "__main__":
    raise SystemExit(main())
