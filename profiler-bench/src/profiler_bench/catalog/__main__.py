"""Command line for the dataset catalog: `python -m profiler_bench.catalog list|materialize|describe|real`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from profiler_bench.catalog import real
from profiler_bench.catalog.datasets import DATASETS
from profiler_bench.catalog.public import PUBLIC_DATASETS
from profiler_bench.catalog.real_datasets import REAL_DATASETS
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
        command.add_argument("--seed", type=int, default=0, help="structure seed (and sample seed by default)")
        command.add_argument("--sample-seed", type=int, default=None, help="seed of the rows (default: --seed)")
        command.add_argument("--cache-dir", type=Path, default=None)
        command.add_argument(
            "--skip-hash-check", action="store_true", help="trust a cached instance without re-hashing its files"
        )
    command = commands.add_parser("real", help="export real datasets and print each table's rows, bytes, and hash")
    command.add_argument("names", nargs="*", metavar="name", help=f"datasets (default: all of {sorted(REAL_DATASETS)})")
    command.add_argument("--tier", choices=real.REAL_TIERS, default="s")
    command.add_argument("--cache-dir", type=Path, default=None)
    command.add_argument("--skip-hash-check", action="store_true", help="trust a cached tier without re-hashing it")
    return parser


def _real(names: Sequence[str], tier: str, root: Path | None, check_hash: bool) -> None:
    for name in names or list(REAL_DATASETS):
        try:
            instance = real.materialize(REAL_DATASETS[name], tier, root, check_hash)
        except real.Unavailable as reason:
            print(f"skipped {name}: {reason}")
            continue
        sizes = real.table_sizes(instance)
        for table in instance.manifest.tables:
            row = f"{name:<17} {tier:<4} {table.name:<22} {table.row_count:>10,} {sizes[table.name]:>12,}"
            print(f"{row} {instance.table_hashes[table.name]}")


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the catalog command line and returns the exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    known = sorted(REAL_DATASETS) if args.command == "real" else sorted(DATASETS)
    unknown = sorted(set(getattr(args, "names", [])) - set(known))
    if unknown:
        parser.error(f"unknown datasets {unknown}; choose from {known}")
    if args.command == "list":
        for name, dataset in DATASETS.items():
            print(f"{name:<10} {dataset.description}")
        for name, public in PUBLIC_DATASETS.items():
            print(f"{name:<10} [public] {public.description}")
        for name, real_dataset in REAL_DATASETS.items():
            print(f"{name:<10} [real] {real_dataset.description}")
        print(f"tiers: {', '.join(f'{t}={n:,}' for t, n in TIERS.items())} base rows")
    elif args.command == "real":
        _real(args.names, args.tier, args.cache_dir, not args.skip_hash_check)
    elif args.command == "materialize":
        for name in args.names or list(DATASETS):
            instance = materialize(
                name, args.tier, args.seed, args.cache_dir, not args.skip_hash_check, args.sample_seed
            )
            seeds = f"seed={instance.structure_seed} sample={instance.sample_seed}"
            print(f"{name:<10} {args.tier:<3} {seeds} {instance.content_hash} {instance.path}")
    else:
        instance = materialize(
            args.name, args.tier, args.seed, args.cache_dir, not args.skip_hash_check, args.sample_seed
        )
        print(instance.manifest.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
