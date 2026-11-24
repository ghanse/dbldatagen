"""Subprocess worker that profiles one catalog dataset and reports timings."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    """Profiles one dataset in a fresh process and writes a JSON result."""
    parser = argparse.ArgumentParser(prog="profiler_bench.runner.worker")
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Optional data-volume input fraction; when set, profile a harness subset only.",
    )
    parser.add_argument("--subset-seed", type=int, default=0, help="Seed for data-volume subset draws.")
    args = parser.parse_args(argv)
    result: dict[str, Any] = {
        "ok": False,
        "dataset": args.dataset,
        "tier": args.tier,
        "seed": args.seed,
        "sample_seed": args.sample_seed if args.sample_seed is not None else args.seed,
        "worktree": str(args.worktree.resolve()),
    }
    if args.fraction is not None:
        result["fraction"] = args.fraction
        result["subset_seed"] = args.subset_seed
    try:
        policy = json.loads(args.policy_json.read_text())
        _prepare_profiler_import(args.worktree.resolve())
        if args.cache_dir is not None:
            os.environ["PROFILER_BENCH_CACHE"] = str(args.cache_dir)
        result.update(_profile_once(args, policy))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, default=str))
    return 0 if result.get("ok") else 1


def _profile_once(args: argparse.Namespace, policy: dict[str, Any]) -> dict[str, Any]:
    """Runs warmup + timed profile after the profiler import path is prepared."""
    import dbldatagen_profiler  # noqa: PLC0415
    from dbldatagen_profiler.engines.spark.engine import profile_tables  # noqa: PLC0415
    from dbldatagen_profiler.model import dump_profile, load_profile  # noqa: PLC0415
    from profiler_bench.catalog import REAL_DATASETS, load, materialize  # noqa: PLC0415
    from profiler_bench.catalog.real import load_part  # noqa: PLC0415
    from profiler_bench.runner.identity import profile_content_hash  # noqa: PLC0415
    from profiler_bench.runner.spark_conf import build_spark  # noqa: PLC0415

    profiler_module_file = str(Path(dbldatagen_profiler.__file__).resolve())
    spark = build_spark(policy, app_name=f"rsi-{args.dataset}-{args.tier}")
    sample_seed = args.sample_seed if args.sample_seed is not None else args.seed
    if args.dataset in REAL_DATASETS:
        if args.fraction is not None:
            raise ValueError("data-volume subsets of real datasets are not supported")
        tables = load_part(REAL_DATASETS[args.dataset], args.tier, "profiled", spark, args.cache_dir)
    else:
        tables = load(args.dataset, args.tier, seed=args.seed, spark=spark, sample_seed=sample_seed)
    subset_meta = None
    if args.fraction is not None:
        from profiler_bench.runner.subset import (  # noqa: PLC0415
            DEFAULT_MAX_CLOSURE_ITERATIONS,
            subset_tables,
        )

        manifest = materialize(args.dataset, args.tier, seed=args.seed, sample_seed=sample_seed).manifest
        max_closure = int(
            (policy.get("data_volume") or {}).get("max_closure_iterations", DEFAULT_MAX_CLOSURE_ITERATIONS)
        )
        subset = subset_tables(
            tables,
            manifest,
            fraction=float(args.fraction),
            subset_seed=int(args.subset_seed),
            max_closure_iterations=max_closure,
        )
        tables = subset.tables
        subset_meta = subset.to_meta()
    for frame in tables.values():
        frame.count()
    profile_tables(tables)
    gc.collect()
    cpu_before = time.process_time()
    wall_before = time.perf_counter()
    profile = profile_tables(tables)
    wall_s = time.perf_counter() - wall_before
    cpu_s = time.process_time() - cpu_before
    peak_rss = _peak_rss_bytes()
    profile_obj = json.loads(dump_profile(profile))
    load_profile(profile_obj, strict=False)
    result = {
        "ok": True,
        "wall_s": wall_s,
        "cpu_s": cpu_s,
        "peak_rss_bytes": peak_rss,
        "jvm_rss_bytes": None,
        "profile": profile_obj,
        "profile_hash": profile_content_hash(profile_obj),
        "profiler_module_file": profiler_module_file,
        "not_measured": policy.get("not_measured", []),
    }
    if subset_meta is not None:
        result["subset"] = subset_meta
    return result


def _prepare_profiler_import(worktree: Path) -> None:
    """Puts `<worktree>/profiler/src` first on sys.path and clears prior imports."""
    profiler_src = str(worktree / "profiler" / "src")
    if not Path(profiler_src).is_dir():
        raise FileNotFoundError(f"missing profiler/src under {worktree}")
    for name in list(sys.modules):
        if name == "dbldatagen_profiler" or name.startswith("dbldatagen_profiler."):
            del sys.modules[name]
    sys.path = [profiler_src, *[path for path in sys.path if path != profiler_src]]


def _peak_rss_bytes() -> int:
    """Returns peak resident set size of this process in bytes."""
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports kilobytes.
    return peak if sys.platform == "darwin" else peak * 1024


if __name__ == "__main__":
    raise SystemExit(main())
