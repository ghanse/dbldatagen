#!/usr/bin/env python3
"""Manages the append-only RSI audit log."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "log.jsonl"
DEFAULT_RENDER = ROOT / "LOG.md"
COMMON_FIELDS = {"seq", "ts", "prev_hash", "kind", "actor", "work_item", "summary"}
KIND_FIELDS = {
    "decision": set(),
    "proposal": {"proposal_id", "hypothesis", "mechanism", "targets", "expected_effect", "invariants"},
    "triage": {"proposal_id", "verdict", "reason"},
    "implementation": {"proposal_id", "branch", "pr_url", "commit", "worktree"},
    "benchmark": {
        "proposal_id",
        "experiment_id",
        "baseline_commit",
        "candidate_commit",
        "harness_hash",
        "policy",
        "results_path",
        "summary_metrics",
    },
    "disposition": {"proposal_id", "verdict", "reason", "evidence_path", "retained_patch"},
    "review": {"proposal_id", "reviewer", "verdict", "blocking_issues"},
    "benchmark_change": {"justification"},
}
VERDICTS = {
    "triage": {"accepted", "deferred", "rejected"},
    "disposition": {"accepted", "rejected", "inconclusive"},
}
JSON_FIELDS = {"targets", "invariants", "summary_metrics", "blocking_issues"}


class LogError(ValueError):
    """Represents an invalid RSI log."""


def canonical_line(entry: dict[str, Any]) -> str:
    """Returns an entry encoded in the canonical on-disk format."""
    return json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_log(path: Path) -> tuple[list[dict[str, Any]], list[bytes]]:
    """Reads entries and their exact serialized lines from a log."""
    if not path.exists():
        return [], []
    entries: list[dict[str, Any]] = []
    lines: list[bytes] = []
    for line_number, raw_line in enumerate(path.read_bytes().splitlines(), 1):
        if not raw_line.strip():
            raise LogError(f"line {line_number}: blank lines are not allowed")
        try:
            entry = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise LogError(f"line {line_number}: invalid JSON: {error}") from error
        if not isinstance(entry, dict):
            raise LogError(f"line {line_number}: entry must be a JSON object")
        entries.append(entry)
        lines.append(raw_line)
    return entries, lines


def _nonempty_string(entry: dict[str, Any], field: str, line_number: int) -> None:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise LogError(f"line {line_number}: {field} must be a non-empty string")


def _validate_entry(entry: dict[str, Any], line_number: int) -> None:
    missing_common = COMMON_FIELDS - entry.keys()
    if line_number == 1 and entry.get("seq") == 1:
        missing_common -= {"prev_hash"}
    if missing_common:
        raise LogError(f"line {line_number}: missing fields: {', '.join(sorted(missing_common))}")
    if type(entry.get("seq")) is not int or entry["seq"] < 1:
        raise LogError(f"line {line_number}: seq must be a positive integer")
    kind = entry.get("kind")
    if kind not in KIND_FIELDS:
        raise LogError(f"line {line_number}: unknown kind: {kind!r}")
    for field in ("ts", "actor", "work_item", "summary"):
        _nonempty_string(entry, field, line_number)
    try:
        parsed_ts = datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
    except ValueError as error:
        raise LogError(f"line {line_number}: ts must be an ISO 8601 timestamp") from error
    if parsed_ts.tzinfo is None or parsed_ts.utcoffset() != timezone.utc.utcoffset(parsed_ts):
        raise LogError(f"line {line_number}: ts must be UTC")
    required = KIND_FIELDS[kind]
    missing = required - entry.keys()
    if missing:
        raise LogError(f"line {line_number}: missing fields for {kind}: {', '.join(sorted(missing))}")
    for field in required - JSON_FIELDS:
        _nonempty_string(entry, field, line_number)
    if kind in VERDICTS and entry["verdict"] not in VERDICTS[kind]:
        choices = ", ".join(sorted(VERDICTS[kind]))
        raise LogError(f"line {line_number}: {kind} verdict must be one of: {choices}")
    if kind == "proposal" and not isinstance(entry["targets"], (list, dict)):
        raise LogError(f"line {line_number}: targets must be a JSON array or object")
    if kind == "proposal" and not isinstance(entry["invariants"], list):
        raise LogError(f"line {line_number}: invariants must be a JSON array")
    if kind == "benchmark" and not isinstance(entry["summary_metrics"], dict):
        raise LogError(f"line {line_number}: summary_metrics must be a JSON object")
    if kind == "review" and not isinstance(entry["blocking_issues"], list):
        raise LogError(f"line {line_number}: blocking_issues must be a JSON array")


def validate(entries: list[dict[str, Any]], lines: list[bytes]) -> None:
    """Validates entry schemas, ordering, hashes, and references."""
    proposals: set[str] = set()
    accepted: set[str] = set()
    implementations: set[str] = set()
    benchmarks: set[str] = set()
    for index, entry in enumerate(entries):
        line_number = index + 1
        _validate_entry(entry, line_number)
        expected_seq = line_number
        if entry["seq"] != expected_seq:
            raise LogError(f"line {line_number}: expected seq {expected_seq}, found {entry['seq']}")
        if index == 0:
            if entry.get("prev_hash") is not None:
                raise LogError("line 1: genesis prev_hash must be null or absent")
        else:
            expected_hash = hashlib.sha256(lines[index - 1]).hexdigest()
            if entry.get("prev_hash") != expected_hash:
                raise LogError(f"line {line_number}: prev_hash does not match line {line_number - 1}")
        kind = entry["kind"]
        proposal_id = entry.get("proposal_id")
        if kind == "proposal":
            if proposal_id in proposals:
                raise LogError(f"line {line_number}: duplicate proposal_id {proposal_id}")
            proposals.add(proposal_id)
        elif kind == "triage":
            if proposal_id not in proposals:
                raise LogError(f"line {line_number}: triage requires a prior proposal")
            if entry["verdict"] == "accepted":
                accepted.add(proposal_id)
        elif kind == "implementation":
            if proposal_id not in accepted:
                raise LogError(f"line {line_number}: implementation requires a prior accepted triage")
            implementations.add(proposal_id)
        elif kind == "benchmark":
            if proposal_id not in implementations:
                raise LogError(f"line {line_number}: benchmark requires a prior implementation")
            benchmarks.add(proposal_id)
        elif kind == "disposition" and proposal_id not in benchmarks:
            raise LogError(f"line {line_number}: disposition requires a prior benchmark")
        elif kind == "review" and proposal_id not in implementations:
            raise LogError(f"line {line_number}: review requires a prior implementation")


def verify(path: Path) -> list[dict[str, Any]]:
    """Validates a log and returns its entries."""
    entries, lines = read_log(path)
    validate(entries, lines)
    return entries


def append_entry(path: Path, values: dict[str, Any]) -> dict[str, Any]:
    """Validates and appends one entry to a log."""
    entries, lines = read_log(path)
    validate(entries, lines)
    forbidden = {"seq", "ts", "prev_hash"} & values.keys()
    if forbidden:
        raise LogError(f"generated fields cannot be supplied: {', '.join(sorted(forbidden))}")
    entry = dict(values)
    entry["seq"] = len(entries) + 1
    entry["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    entry["prev_hash"] = hashlib.sha256(lines[-1]).hexdigest() if lines else None
    new_line = canonical_line(entry).encode()
    validate([*entries, entry], [*lines, new_line])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(new_line + b"\n")
    return entry


def _display(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _escape_cell(value: object) -> str:
    return _display(value).replace("|", "\\|").replace("\n", "<br>")


def render(entries: list[dict[str, Any]]) -> str:
    """Renders entries as deterministic Markdown."""
    output = [
        "# RSI Log",
        "",
        "Generated from `log.jsonl`. Do not edit this file directly.",
        "",
        "## Chronology",
        "",
        "| Seq | Timestamp | Kind | Actor | Work item | Summary |",
        "|---:|---|---|---|---|---|",
    ]
    proposals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        output.append(
            "| {seq} | {ts} | {kind} | {actor} | {work_item} | {summary} |".format(
                **{
                    field: _escape_cell(entry[field])
                    for field in ("seq", "ts", "kind", "actor", "work_item", "summary")
                }
            )
        )
        if "proposal_id" in entry:
            proposals[entry["proposal_id"]].append(entry)
    for proposal_id in sorted(proposals, key=lambda item: proposals[item][0]["seq"]):
        output.extend(["", f"## Proposal {proposal_id}", ""])
        for entry in proposals[proposal_id]:
            output.extend([f"### {entry['seq']}. {entry['kind']}", "", entry["summary"], ""])
            details = COMMON_FIELDS | {"proposal_id"}
            for field in sorted(entry.keys() - details):
                output.append(f"- **{field}:** {_display(entry[field])}")
            if entry.keys() - details:
                output.append("")
    return "\n".join(output).rstrip() + "\n"


def render_proposal(proposal_id: str, entries: list[dict[str, Any]]) -> str:
    """Renders one proposal thread as Markdown."""
    output = [f"# Proposal {proposal_id}", ""]
    for entry in entries:
        output.extend([f"## {entry['seq']}. {entry['kind']}", "", entry["summary"], ""])
        details = COMMON_FIELDS | {"proposal_id"}
        for field in sorted(entry.keys() - details):
            output.append(f"- **{field}:** {_display(entry[field])}")
        if entry.keys() - details:
            output.append("")
    return "\n".join(output).rstrip() + "\n"


def _load_values(args: argparse.Namespace) -> dict[str, Any]:
    if args.json_file:
        content = sys.stdin.read() if args.json_file == "-" else Path(args.json_file).read_text()
        values = json.loads(content)
        if not isinstance(values, dict):
            raise LogError("append input must be a JSON object")
    else:
        values = {}
    for field in KIND_FIELDS.keys() | {"actor", "work_item", "summary"} | set().union(*KIND_FIELDS.values()):
        value = getattr(args, field, None)
        if value is not None:
            if field in JSON_FIELDS:
                try:
                    value = json.loads(value)
                except json.JSONDecodeError as error:
                    raise LogError(f"--{field.replace('_', '-')} must contain JSON") from error
            values[field] = value
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    append_parser = subparsers.add_parser("append")
    append_parser.add_argument("--log", type=Path, default=argparse.SUPPRESS)
    append_parser.add_argument(
        "--json-file",
        "--json",
        "--input",
        metavar="PATH",
        help="reads fields from a JSON file, or - for stdin",
    )
    fields = {"kind", "actor", "work_item", "summary"} | set().union(*KIND_FIELDS.values())
    for field in sorted(fields):
        append_parser.add_argument(f"--{field.replace('_', '-')}", dest=field)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--log", type=Path, default=argparse.SUPPRESS)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--log", type=Path, default=argparse.SUPPRESS)
    render_parser.add_argument("--output", type=Path, default=DEFAULT_RENDER)
    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("--log", type=Path, default=argparse.SUPPRESS)
    show_parser.add_argument("proposal_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Runs the RSI log command-line interface."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "append":
            print(json.dumps(append_entry(args.log, _load_values(args)), ensure_ascii=False, sort_keys=True))
        elif args.command == "verify":
            entries = verify(args.log)
            print(f"verified {len(entries)} entries")
        elif args.command == "render":
            args.output.write_text(render(verify(args.log)))
            print(args.output)
        elif args.command == "show":
            matches = [entry for entry in verify(args.log) if entry.get("proposal_id") == args.proposal_id]
            if not matches:
                raise LogError(f"proposal not found: {args.proposal_id}")
            print(render_proposal(args.proposal_id, matches), end="")
    except (LogError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
