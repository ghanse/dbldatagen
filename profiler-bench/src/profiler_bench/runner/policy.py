"""Loads and hashes frozen acceptance policies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_POLICY = Path(__file__).resolve().parents[3] / "policies" / "default.json"


def load_policy(path: Path | None = None) -> dict[str, Any]:
    """Returns a policy JSON object from `path`, defaulting to the packaged policy."""
    policy_path = path or DEFAULT_POLICY
    data = json.loads(policy_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"policy must be a JSON object: {policy_path}")
    return data


def policy_hash(policy: dict[str, Any]) -> str:
    """Returns the SHA-256 of the canonical policy JSON."""
    payload = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def policy_path_hash(path: Path) -> str:
    """Returns the SHA-256 of the on-disk policy file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
