"""Tests the RSI audit log tooling."""

import hashlib
import json
from pathlib import Path

import pytest
import rsi_log


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
            {
                "kind": "implementation",
                "actor": "worker:test/session",
                "work_item": "P2.1",
                "summary": "Implemented quantiles.",
                "proposal_id": "R1",
                "branch": "worker/r1",
                "pr_url": "https://example.test/pr/1",
                "commit": "abc123",
                "worktree": "/tmp/r1",
            },
        )
