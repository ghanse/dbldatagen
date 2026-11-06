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
