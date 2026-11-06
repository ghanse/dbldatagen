"""Command line for the dataset catalog: `python -m profiler_bench.catalog list|materialize|describe`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from profiler_bench.catalog.datasets import DATASETS
from profiler_bench.catalog.public import PUBLIC_DATASETS
from profiler_bench.catalog.registry import DEFAULT_TIER, TIERS
from profiler_bench.catalog.store import materialize


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m profiler_bench.catalog", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list catalog datasets")
    for name, help_text, many in (
        ("materialize", "write datasets to the parquet cache and print their content hashes", True),
        ("describe", "print a dataset's ground-truth manifest as JSON", False),
    ):
        command = commands.add_parser(name, help=help_text)
        if many:
            command.add_argument(
                "names", nargs="*", metavar="name", help=f"datasets (default: all of {sorted(DATASETS)})"
            )
        else:
            command.add_argument("name", choices=sorted(DATASETS))
        command.add_argument("--tier", choices=sorted(TIERS), default=DEFAULT_TIER)
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--cache-dir", type=Path, default=None)
        command.add_argument(
            "--skip-hash-check", action="store_true", help="trust a cached instance without re-hashing its files"
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the catalog command line and returns the exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    unknown = sorted(set(getattr(args, "names", [])) - set(DATASETS))
    if unknown:
        parser.error(f"unknown datasets {unknown}; choose from {sorted(DATASETS)}")
    if args.command == "list":
        for name, dataset in DATASETS.items():
            print(f"{name:<10} {dataset.description}")
        for name, public in PUBLIC_DATASETS.items():
            print(f"{name:<10} [public] {public.description}")
        print(f"tiers: {', '.join(f'{t}={n:,}' for t, n in TIERS.items())} base rows")
    elif args.command == "materialize":
        for name in args.names or list(DATASETS):
            instance = materialize(name, args.tier, args.seed, args.cache_dir, not args.skip_hash_check)
            print(f"{name:<10} {args.tier:<3} seed={args.seed} {instance.content_hash} {instance.path}")
    else:
        instance = materialize(args.name, args.tier, args.seed, args.cache_dir, not args.skip_hash_check)
        print(instance.manifest.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
