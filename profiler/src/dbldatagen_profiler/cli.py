"""Command-line interface for the dbldatagen profiler.

Provides commands for profiling datasets, validating profiles, and rendering
the profile JSON Schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path

from dbldatagen_profiler.model import SourceInfo, dump_profile, load_profile, render_json_schema


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the `dbldatagen-profile` command.

    Args:
        argv: Optional command-line arguments (default `sys.argv[1:]`).

    Returns:
        Process exit code: 0 on success, 1 for an invalid profile or profiling failure, and 2 for usage errors.
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

    profile = commands.add_parser("profile", help="Profile one or more datasets with local Spark.")
    profile.add_argument(
        "--table",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named input table; repeat for multiple tables.",
    )
    profile.add_argument("--format", choices=("parquet", "csv", "delta"), default="parquet", help="Input format.")
    profile.add_argument("-o", "--output", type=Path, required=True, help="Output profile JSON file.")
    profile.set_defaults(handler=_profile)

    pii = commands.add_parser("pii", help="Detect or substitute personally identifying data.")
    pii_commands = pii.add_subparsers(dest="pii_command", required=True)
    detect = pii_commands.add_parser("detect", help="Detect PII semantics.")
    _add_input_arguments(detect)
    detect.add_argument("-o", "--output", type=Path, required=True, help="Output semantic JSON file.")
    detect.set_defaults(handler=_pii_detect)
    substitute = pii_commands.add_parser("substitute", help="Substitute PII using a profile and key.")
    _add_input_arguments(substitute)
    substitute.add_argument("--profile", type=Path, required=True, help="Profile JSON file.")
    substitute.add_argument("--key-file", type=Path, required=True, help="File containing the substitution key.")
    substitute.add_argument("-o", "--output", type=Path, required=True, help="Output directory.")
    substitute.set_defaults(handler=_pii_substitute)
    return parser


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds common named-table and input-format arguments."""
    parser.add_argument("--table", action="append", required=True, metavar="NAME=PATH", help="Named input table.")
    parser.add_argument("--format", choices=("parquet", "csv", "delta"), default="parquet", help="Input format.")


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
    """Profiles the input tables and writes a validated JSON profile."""
    spark_errors = import_module("pyspark.errors")
    py4j_errors = import_module("py4j.protocol")
    try:
        parsed = [_parse_table(value) for value in args.table]
        inputs = dict(parsed)
        if len(inputs) != len(parsed):
            raise ValueError("duplicate table name")
        SparkSession = import_module("pyspark.sql").SparkSession
        profile_tables = import_module("dbldatagen_profiler.engines.spark").profile_tables
        spark = (
            SparkSession.builder.master("local[*]")
            .appName("dbldatagen-profiler")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
        tables = {name: spark.read.format(args.format).load(path) for name, path in inputs.items()}
        profile = profile_tables(tables)
        profile.source = SourceInfo(name=", ".join(inputs), format=args.format)
        load_profile(profile.model_dump(mode="json"), strict=True)
        args.output.write_text(dump_profile(profile) + "\n", encoding="utf-8")
    except (OSError, ValueError, spark_errors.PySparkException, py4j_errors.Py4JError) as error:
        print(f"dbldatagen-profile profile: {error}", file=sys.stderr)
        return 1
    return 0


def _spark_inputs(args: argparse.Namespace) -> tuple[object, dict[str, object]]:
    """Creates a local UTC Spark session and reads CLI table arguments."""
    parsed = [_parse_table(value) for value in args.table]
    inputs = dict(parsed)
    if len(inputs) != len(parsed):
        raise ValueError("duplicate table name")
    SparkSession = import_module("pyspark.sql").SparkSession
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("dbldatagen-profiler")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    return spark, {name: spark.read.format(args.format).load(path) for name, path in inputs.items()}


def _pii_detect(args: argparse.Namespace) -> int:
    """Detects PII in input tables and writes semantic JSON."""
    try:
        _, tables = _spark_inputs(args)
        detect = import_module("dbldatagen_profiler.pii").detect
        result = detect(tables)
        payload = {
            table: {column: semantic.model_dump(mode="json") for column, semantic in columns.items()}
            for table, columns in result.items()
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"dbldatagen-profile pii detect: {error}", file=sys.stderr)
        return 1
    return 0


def _pii_substitute(args: argparse.Namespace) -> int:
    """Substitutes PII in input tables and writes Parquet outputs and a report."""
    try:
        _, tables = _spark_inputs(args)
        profile = load_profile(args.profile.read_text(encoding="utf-8"), strict=True)
        key = args.key_file.read_bytes()
        substitute = import_module("dbldatagen_profiler.pii").substitute
        result = substitute(tables, profile, key=key)
        args.output.mkdir(parents=True, exist_ok=True)
        for name, frame in result.tables.items():
            frame.write.mode("error").parquet(str(args.output / name))
        (args.output / "report.json").write_text(json.dumps(result.report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"dbldatagen-profile pii substitute: {error}", file=sys.stderr)
        return 1
    return 0


def _parse_table(value: str) -> tuple[str, str]:
    """Parses one `NAME=PATH` table argument."""
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"invalid --table {value!r}; expected NAME=PATH")
    return name, path
