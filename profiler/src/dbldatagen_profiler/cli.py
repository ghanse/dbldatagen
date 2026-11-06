"""Command-line interface for the dbldatagen profiler.

Provides the `dbldatagen-profile` command with `validate` and `schema`
subcommands. The `profile` subcommand is reserved for the profiling engine and
is not implemented yet.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dbldatagen_profiler.model import load_profile, render_json_schema


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the `dbldatagen-profile` command.

    Args:
        argv: Optional command-line arguments (default `sys.argv[1:]`).

    Returns:
        Process exit code: 0 on success, 1 for an invalid profile, 2 for usage errors and unimplemented commands.
    """
    args = _build_parser().parse_args(argv)
    return int(args.handler(args))


def _build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser with one sub-parser per subcommand."""
    parser = argparse.ArgumentParser(prog="dbldatagen-profile", description="Profile datasets for dbldatagen.")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate a profile JSON file against the spec.")
    validate.add_argument("path", type=Path, help="Path to the profile JSON file.")
    validate.add_argument("--strict", action="store_true", help="Reject fields that the spec does not define.")
    validate.set_defaults(handler=_validate)

    schema = commands.add_parser("schema", help="Print or write the profile JSON Schema.")
    schema.add_argument("-o", "--output", type=Path, help="Write the schema to this file instead of stdout.")
    schema.set_defaults(handler=_schema)

    profile = commands.add_parser("profile", help="Profile a dataset (not implemented yet).")
    profile.set_defaults(handler=_profile)
    return parser


def _validate(args: argparse.Namespace) -> int:
    """Validates the profile at `args.path` and reports the result."""
    try:
        profile = load_profile(args.path.read_text(encoding="utf-8"), strict=args.strict)
    except (OSError, ValueError) as error:
        print(f"{args.path}: invalid profile\n{error}", file=sys.stderr)
        return 1
    print(
        f"{args.path}: valid profile (spec {profile.spec_version}, "
        f"{len(profile.tables)} tables, {len(profile.relationships or [])} relationships)"
    )
    return 0


def _schema(args: argparse.Namespace) -> int:
    """Writes the JSON Schema to `args.output`, or to stdout when no output is given."""
    text = render_json_schema()
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0


def _profile(args: argparse.Namespace) -> int:
    """Reports that profiling is not implemented yet."""
    print("dbldatagen-profile profile: not implemented yet", file=sys.stderr)
    return 2
