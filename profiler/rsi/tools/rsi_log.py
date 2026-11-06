#!/usr/bin/env python3
"""Manages the append-only RSI audit log."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field as dataclass_field
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
    "review": {"approved", "changes_requested", "rejected"},
}
JSON_FIELDS = {"targets", "invariants", "summary_metrics", "blocking_issues"}
APPEND_FIELDS = {"kind", "actor", "work_item", "summary"} | set().union(*KIND_FIELDS.values())


class LogError(ValueError):
    """Represents an invalid RSI log."""


def canonical_line(entry: dict[str, Any]) -> str:
    """Returns an entry encoded in the canonical on-disk format."""
    return json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_log(content: bytes) -> tuple[list[dict[str, Any]], list[bytes]]:
    """Parses entries and their exact serialized lines from log bytes."""
    entries: list[dict[str, Any]] = []
    lines: list[bytes] = []
    for line_number, raw_line in enumerate(content.splitlines(), 1):
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


def read_log(path: Path) -> tuple[list[dict[str, Any]], list[bytes]]:
    """Reads entries and their exact serialized lines from a log."""
    return _parse_log(path.read_bytes()) if path.exists() else ([], [])


def _nonempty_string(entry: dict[str, Any], field: str, line_number: int) -> None:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise LogError(f"line {line_number}: {field} must be a non-empty string")


def _validate_common_fields(entry: dict[str, Any], line_number: int) -> None:
    missing_common = COMMON_FIELDS - entry.keys()
    if line_number == 1 and entry.get("seq") == 1:
        missing_common -= {"prev_hash"}
    if missing_common:
        raise LogError(f"line {line_number}: missing fields: {', '.join(sorted(missing_common))}")
    if type(entry.get("seq")) is not int or entry["seq"] < 1:
        raise LogError(f"line {line_number}: seq must be a positive integer")
    for field in ("ts", "actor", "work_item", "summary"):
        _nonempty_string(entry, field, line_number)


def _validate_timestamp(entry: dict[str, Any], line_number: int) -> None:
    try:
        parsed_ts = datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
    except ValueError as error:
        raise LogError(f"line {line_number}: ts must be an ISO 8601 timestamp") from error
    if parsed_ts.tzinfo is None or parsed_ts.utcoffset() != timezone.utc.utcoffset(parsed_ts):
        raise LogError(f"line {line_number}: ts must be UTC")


def _validate_kind(entry: dict[str, Any], line_number: int) -> str:
    kind = entry.get("kind")
    if kind not in KIND_FIELDS:
        raise LogError(f"line {line_number}: unknown kind: {kind!r}")
    return kind


def _validate_kind_fields(entry: dict[str, Any], kind: str, line_number: int) -> None:
    required = KIND_FIELDS[kind]
    missing = required - entry.keys()
    if missing:
        raise LogError(f"line {line_number}: missing fields for {kind}: {', '.join(sorted(missing))}")
    for field in required - JSON_FIELDS:
        _nonempty_string(entry, field, line_number)


def _validate_verdict(entry: dict[str, Any], kind: str, line_number: int) -> None:
    if kind in VERDICTS and entry["verdict"] not in VERDICTS[kind]:
        choices = ", ".join(sorted(VERDICTS[kind]))
        raise LogError(f"line {line_number}: {kind} verdict must be one of: {choices}")


def _validate_proposal_fields(entry: dict[str, Any], line_number: int) -> None:
    if not isinstance(entry["targets"], (list, dict)):
        raise LogError(f"line {line_number}: targets must be a JSON array or object")
    if not isinstance(entry["invariants"], list):
        raise LogError(f"line {line_number}: invariants must be a JSON array")


def _validate_benchmark_fields(entry: dict[str, Any], line_number: int) -> None:
    if not isinstance(entry["summary_metrics"], dict):
        raise LogError(f"line {line_number}: summary_metrics must be a JSON object")


def _validate_review_fields(entry: dict[str, Any], line_number: int) -> None:
    if not isinstance(entry["blocking_issues"], list):
        raise LogError(f"line {line_number}: blocking_issues must be a JSON array")


KIND_VALIDATORS = {
    "proposal": _validate_proposal_fields,
    "benchmark": _validate_benchmark_fields,
    "review": _validate_review_fields,
}


def _validate_entry(entry: dict[str, Any], line_number: int) -> None:
    _validate_common_fields(entry, line_number)
    kind = _validate_kind(entry, line_number)
    _validate_timestamp(entry, line_number)
    _validate_kind_fields(entry, kind, line_number)
    _validate_verdict(entry, kind, line_number)
    validator = KIND_VALIDATORS.get(kind)
    if validator is not None:
        validator(entry, line_number)


@dataclass
class _ReferenceState:
    proposals: set[str] = dataclass_field(default_factory=set)
    triage_verdicts: dict[str, str] = dataclass_field(default_factory=dict)
    implementations: set[str] = dataclass_field(default_factory=set)
    benchmarks: set[str] = dataclass_field(default_factory=set)


def _record_proposal(entry: dict[str, Any], line_number: int, state: _ReferenceState) -> None:
    proposal_id = entry["proposal_id"]
    if proposal_id in state.proposals:
        raise LogError(f"line {line_number}: duplicate proposal_id {proposal_id}")
    state.proposals.add(proposal_id)


def _record_triage(entry: dict[str, Any], line_number: int, state: _ReferenceState) -> None:
    proposal_id = entry["proposal_id"]
    if proposal_id not in state.proposals:
        raise LogError(f"line {line_number}: triage requires a prior proposal")
    state.triage_verdicts[proposal_id] = entry["verdict"]


def _record_implementation(entry: dict[str, Any], line_number: int, state: _ReferenceState) -> None:
    proposal_id = entry["proposal_id"]
    if state.triage_verdicts.get(proposal_id) != "accepted":
        raise LogError(f"line {line_number}: implementation requires a prior accepted triage")
    state.implementations.add(proposal_id)


def _record_benchmark(entry: dict[str, Any], line_number: int, state: _ReferenceState) -> None:
    proposal_id = entry["proposal_id"]
    if proposal_id not in state.implementations:
        raise LogError(f"line {line_number}: benchmark requires a prior implementation")
    state.benchmarks.add(proposal_id)


def _validate_disposition_reference(entry: dict[str, Any], line_number: int, state: _ReferenceState) -> None:
    if entry["proposal_id"] not in state.benchmarks:
        raise LogError(f"line {line_number}: disposition requires a prior benchmark")


def _validate_review_references(entry: dict[str, Any], line_number: int, state: _ReferenceState) -> None:
    proposal_id = entry["proposal_id"]
    if proposal_id not in state.proposals:
        raise LogError(f"line {line_number}: review requires a prior proposal")
    if proposal_id not in state.implementations:
        raise LogError(f"line {line_number}: review requires a prior implementation")


REFERENCE_VALIDATORS = {
    "proposal": _record_proposal,
    "triage": _record_triage,
    "implementation": _record_implementation,
    "benchmark": _record_benchmark,
    "disposition": _validate_disposition_reference,
    "review": _validate_review_references,
}


def _validate_chain_entry(entry: dict[str, Any], lines: list[bytes], index: int) -> None:
    line_number = index + 1
    if entry["seq"] != line_number:
        raise LogError(f"line {line_number}: expected seq {line_number}, found {entry['seq']}")
    if index == 0:
        if entry.get("prev_hash") is not None:
            raise LogError("line 1: genesis prev_hash must be null or absent")
        return
    expected_hash = hashlib.sha256(lines[index - 1]).hexdigest()
    if entry.get("prev_hash") != expected_hash:
        raise LogError(f"line {line_number}: prev_hash does not match line {line_number - 1}")


def validate(entries: list[dict[str, Any]], lines: list[bytes]) -> None:
    """Validates entry schemas, ordering, hashes, and references."""
    state = _ReferenceState()
    for index, entry in enumerate(entries):
        line_number = index + 1
        _validate_entry(entry, line_number)
        _validate_chain_entry(entry, lines, index)
        validator = REFERENCE_VALIDATORS.get(entry["kind"])
        if validator is not None:
            validator(entry, line_number, state)


def verify(path: Path) -> list[dict[str, Any]]:
    """Validates a log and returns its entries."""
    entries, lines = read_log(path)
    validate(entries, lines)
    return entries


def append_entry(path: Path, values: dict[str, Any]) -> dict[str, Any]:
    """Validates and appends one entry to a log."""
    forbidden = {"seq", "ts", "prev_hash"} & values.keys()
    if forbidden:
        raise LogError(f"generated fields cannot be supplied: {', '.join(sorted(forbidden))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        entries, lines = _parse_log(handle.read())
        validate(entries, lines)
        entry = dict(values)
        entry["seq"] = len(entries) + 1
        entry["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        entry["prev_hash"] = hashlib.sha256(lines[-1]).hexdigest() if lines else None
        new_line = canonical_line(entry).encode()
        validate([*entries, entry], [*lines, new_line])
        handle.seek(0, os.SEEK_END)
        handle.write(new_line + b"\n")
        handle.flush()
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
    for field in APPEND_FIELDS:
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
    for field in sorted(APPEND_FIELDS):
        help_text = "JSON-encoded array or object" if field in JSON_FIELDS else None
        append_parser.add_argument(f"--{field.replace('_', '-')}", dest=field, help=help_text)
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
