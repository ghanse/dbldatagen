"""Tests the RSI audit log tooling."""

import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import rsi_log


SCRIPT = Path(rsi_log.__file__)
RSI_DIR = SCRIPT.parents[1]


def run_cli(log: Path, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    """Runs the log CLI against a temporary log."""
    return subprocess.run(
        [sys.executable, SCRIPT, "--log", log, "append", *args],
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def write_genesis(path: Path) -> None:
    """Writes a valid genesis decision."""
    path.write_text(
        json.dumps(
            {
                "seq": 1,
                "ts": "2026-11-06T00:00:00Z",
                "kind": "decision",
                "actor": "polly",
                "work_item": "P1",
                "summary": "Project kickoff.",
            }
        )
        + "\n"
    )


def proposal() -> dict[str, object]:
    """Returns valid proposal fields."""
    return {
        "kind": "proposal",
        "actor": "worker:test/session",
        "work_item": "P2.1",
        "summary": "Profile quantiles.",
        "proposal_id": "R1",
        "hypothesis": "Quantiles improve marginal reconstruction.",
        "mechanism": "Store a bounded quantile sketch.",
        "targets": {"datasets": ["skew"], "metrics": ["wasserstein"]},
        "expected_effect": "Lower Wasserstein distance.",
        "invariants": ["privacy threshold unchanged"],
    }


def event(kind: str, **fields: object) -> dict[str, object]:
    """Returns common fields merged with kind-specific fields."""
    return {
        "kind": kind,
        "actor": "polly",
        "work_item": "P2.1",
        "summary": f"Recorded {kind}.",
        **fields,
    }


def append_accepted_proposal(log: Path) -> None:
    """Appends a proposal and accepted triage."""
    rsi_log.append_entry(log, proposal())
    rsi_log.append_entry(log, event("triage", proposal_id="R1", verdict="accepted", reason="Promising."))


def implementation() -> dict[str, object]:
    """Returns valid implementation fields."""
    return event(
        "implementation",
        proposal_id="R1",
        branch="worker/r1",
        pr_url="https://example.test/pr/1",
        commit="abc123",
        worktree="/tmp/r1",
    )


def test_append_verify_and_render(tmp_path: Path) -> None:
    """Appends, verifies, and deterministically renders entries."""
    log = tmp_path / "log.jsonl"
    write_genesis(log)
    added = rsi_log.append_entry(log, proposal())
    assert added["seq"] == 2
    first_line = log.read_bytes().splitlines()[0]
    assert added["prev_hash"] == hashlib.sha256(first_line).hexdigest()
    entries = rsi_log.verify(log)
    first_render = rsi_log.render(entries)
    second_render = rsi_log.render(rsi_log.verify(log))
    assert first_render == second_render
    assert "## Proposal R1" in first_render
    assert "Profile quantiles." in first_render
    shown = rsi_log.render_proposal("R1", [entries[1]])
    assert shown.startswith("# Proposal R1\n")


def test_hash_chain_detects_tampering(tmp_path: Path) -> None:
    """Rejects a changed line protected by a later hash."""
    log = tmp_path / "log.jsonl"
    write_genesis(log)
    rsi_log.append_entry(log, proposal())
    content = log.read_text().replace("Project kickoff.", "Changed kickoff.")
    log.write_text(content)
    with pytest.raises(rsi_log.LogError, match="prev_hash"):
        rsi_log.verify(log)


def test_referential_rule_violation(tmp_path: Path) -> None:
    """Rejects implementation without an accepted triage."""
    log = tmp_path / "log.jsonl"
    write_genesis(log)
    rsi_log.append_entry(log, proposal())
    with pytest.raises(rsi_log.LogError, match="accepted triage"):
        rsi_log.append_entry(
            log,
            implementation(),
        )


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (event("triage", proposal_id="missing", verdict="accepted", reason="Unknown."), "prior proposal"),
        (
            event(
                "disposition",
                proposal_id="missing",
                verdict="rejected",
                reason="No evidence.",
                evidence_path="results.json",
                retained_patch="change.patch",
            ),
            "prior benchmark",
        ),
        (
            event(
                "review",
                proposal_id="missing",
                reviewer="reviewer:test/session",
                verdict="approved",
                blocking_issues=[],
            ),
            "prior proposal",
        ),
    ],
)
def test_missing_references(tmp_path: Path, entry: dict[str, object], message: str) -> None:
    """Rejects entries whose referenced predecessors do not exist."""
    log = tmp_path / "log.jsonl"
    write_genesis(log)
    with pytest.raises(rsi_log.LogError, match=message):
        rsi_log.append_entry(log, entry)


def test_duplicate_proposal_id(tmp_path: Path) -> None:
    """Rejects a duplicate proposal ID."""
    log = tmp_path / "log.jsonl"
    write_genesis(log)
    rsi_log.append_entry(log, proposal())
    with pytest.raises(rsi_log.LogError, match="duplicate proposal_id"):
        rsi_log.append_entry(log, proposal())


def test_concurrent_append_serializes_writers(tmp_path: Path) -> None:
    """Serializes concurrent writers without losing entries."""
    log = tmp_path / "log.jsonl"
    write_genesis(log)
    entries = [event("decision", summary=f"Decision {number}.") for number in range(8)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda entry: rsi_log.append_entry(log, entry), entries))
    verified = rsi_log.verify(log)
    assert [entry["seq"] for entry in verified] == list(range(1, 10))


def test_latest_triage_revokes_acceptance(tmp_path: Path) -> None:
    """Rejects implementation after a later triage revokes acceptance."""
    log = tmp_path / "log.jsonl"
    write_genesis(log)
    append_accepted_proposal(log)
    rsi_log.append_entry(log, event("triage", proposal_id="R1", verdict="deferred", reason="Wait."))
    with pytest.raises(rsi_log.LogError, match="accepted triage"):
        rsi_log.append_entry(log, implementation())


def test_review_verdict(tmp_path: Path) -> None:
    """Rejects an unknown review verdict."""
    log = tmp_path / "log.jsonl"
    write_genesis(log)
    append_accepted_proposal(log)
    rsi_log.append_entry(log, implementation())
    with pytest.raises(rsi_log.LogError, match="review verdict"):
        rsi_log.append_entry(
            log,
            event(
                "review",
                proposal_id="R1",
                reviewer="reviewer:test/session",
                verdict="maybe",
                blocking_issues=[],
            ),
        )


def test_verify_cli_exit_codes(tmp_path: Path) -> None:
    """Returns zero for a valid log and nonzero for an invalid log."""
    log = tmp_path / "log.jsonl"
    write_genesis(log)
    valid = subprocess.run([sys.executable, SCRIPT, "verify", "--log", log], check=False)
    log.write_text("not json\n")
    invalid = subprocess.run([sys.executable, SCRIPT, "verify", "--log", log], check=False)
    assert valid.returncode == 0
    assert invalid.returncode != 0


def test_append_decision_with_cli_flags(tmp_path: Path) -> None:
    """Maps common CLI flags to a decision entry."""
    log = tmp_path / "log.jsonl"
    result = run_cli(
        log,
        "--kind",
        "decision",
        "--actor",
        "worker:test/session",
        "--work-item",
        "P1.7a",
        "--summary",
        "Fix flag mapping.",
    )
    assert result.returncode == 0, result.stderr
    assert rsi_log.verify(log)[0] | {"seq": None, "ts": None, "prev_hash": None} == {
        "seq": None,
        "ts": None,
        "prev_hash": None,
        "kind": "decision",
        "actor": "worker:test/session",
        "work_item": "P1.7a",
        "summary": "Fix flag mapping.",
    }


def test_append_proposal_with_cli_flags_and_json_values(tmp_path: Path) -> None:
    """Maps proposal flags and decodes its JSON-valued fields."""
    log = tmp_path / "log.jsonl"
    result = run_cli(
        log,
        "--kind",
        "proposal",
        "--actor",
        "worker:test/session",
        "--work-item",
        "P2.1",
        "--summary",
        "Profile quantiles.",
        "--proposal-id",
        "R1",
        "--hypothesis",
        "Quantiles improve reconstruction.",
        "--mechanism",
        "Store a bounded sketch.",
        "--targets",
        '{"datasets":["skew"]}',
        "--expected-effect",
        "Lower distance.",
        "--invariants",
        '["privacy unchanged"]',
    )
    assert result.returncode == 0, result.stderr
    entry = rsi_log.verify(log)[0]
    assert entry["targets"] == {"datasets": ["skew"]}
    assert entry["invariants"] == ["privacy unchanged"]
    assert {field: entry[field] for field in rsi_log.KIND_FIELDS["proposal"] - rsi_log.JSON_FIELDS} == {
        "proposal_id": "R1",
        "hypothesis": "Quantiles improve reconstruction.",
        "mechanism": "Store a bounded sketch.",
        "expected_effect": "Lower distance.",
    }


def test_append_review_with_cli_flags(tmp_path: Path) -> None:
    """Maps review flags, including JSON blocking issues, after valid predecessors."""
    log = tmp_path / "log.jsonl"
    append_accepted_proposal(log)
    rsi_log.append_entry(log, implementation())
    result = run_cli(
        log,
        "--kind",
        "review",
        "--actor",
        "polly",
        "--work-item",
        "P2.1",
        "--summary",
        "Requested changes.",
        "--proposal-id",
        "R1",
        "--reviewer",
        "reviewer:test/session",
        "--verdict",
        "changes_requested",
        "--blocking-issues",
        '["Add CLI tests"]',
    )
    assert result.returncode == 0, result.stderr
    entry = rsi_log.verify(log)[-1]
    assert entry["reviewer"] == "reviewer:test/session"
    assert entry["verdict"] == "changes_requested"
    assert entry["blocking_issues"] == ["Add CLI tests"]


def test_append_flags_override_json_input(tmp_path: Path) -> None:
    """Uses CLI flags in preference to fields loaded from JSON."""
    log = tmp_path / "log.jsonl"
    json_entry = event("decision", actor="json-actor", summary="JSON summary.")
    result = run_cli(
        log,
        "--json",
        "-",
        "--actor",
        "flag-actor",
        "--summary",
        "Flag summary.",
        input_text=json.dumps(json_entry),
    )
    assert result.returncode == 0, result.stderr
    entry = rsi_log.verify(log)[0]
    assert entry["actor"] == "flag-actor"
    assert entry["summary"] == "Flag summary."
    assert entry["kind"] == "decision"
    assert entry["work_item"] == "P2.1"


def test_all_append_flags_map_to_their_fields(tmp_path: Path) -> None:
    """Maps every append flag to its identically named entry field."""
    log = tmp_path / "log.jsonl"
    expected = {field: f"value-{field}" for field in rsi_log.APPEND_FIELDS}
    expected.update(
        kind="decision",
        actor="worker:test/session",
        work_item="P1.7a",
        summary="Exercise every flag.",
        targets=["target"],
        invariants=["invariant"],
        summary_metrics={"score": 1},
        blocking_issues=["issue"],
    )
    flags = []
    for field in sorted(expected):
        value = json.dumps(expected[field]) if field in rsi_log.JSON_FIELDS else str(expected[field])
        flags.extend((f"--{field.replace('_', '-')}", value))
    result = run_cli(log, *flags)
    assert result.returncode == 0, result.stderr
    entry = rsi_log.verify(log)[0]
    assert {field: entry[field] for field in rsi_log.APPEND_FIELDS} == expected


def test_truncated_last_line(tmp_path: Path) -> None:
    """Rejects a truncated final JSON line."""
    log = tmp_path / "log.jsonl"
    write_genesis(log)
    with log.open("ab") as handle:
        handle.write(b'{"seq":2')
    with pytest.raises(rsi_log.LogError, match="invalid JSON"):
        rsi_log.verify(log)


def test_committed_render_is_current() -> None:
    """Matches the committed rendering to the committed log."""
    assert (RSI_DIR / "LOG.md").read_text() == rsi_log.render(rsi_log.verify(RSI_DIR / "log.jsonl"))
