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
            with contextlib.suppress(Exception):
                spark.stop()
        result["profiler_module_file"] = _harness_profiler_module_file()
        result["sys_path"] = list(sys.path)
        _atomic_write_json(args.out, result)
    return 0 if result.get("ok") else 1


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
