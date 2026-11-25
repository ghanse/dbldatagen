"""Loads and hashes frozen acceptance policies."""

from __future__ import annotations

import copy
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


def effective_speed_samples(policy: dict[str, Any], proposal_type: str) -> dict[str, int]:
    """Returns per-tier speed sample counts for a proposal type.

    Accuracy proposals take only a smoke check: `speed_samples_accuracy` overrides
    `speed_samples` tier by tier when present. Other proposal types keep the full counts.
    """
    counts = dict(policy.get("speed_samples", {}))
    if proposal_type == "accuracy":
        counts.update(policy.get("speed_samples_accuracy", {}))
    return {str(tier): int(n) for tier, n in counts.items()}


def screen_section(policy: dict[str, Any]) -> dict[str, Any]:
    """Returns the policy's `screen` section, raising when it is missing or malformed."""
    screen = policy.get("screen")
    if not isinstance(screen, dict):
        raise ValueError("policy is missing a screen section")
    for key in ("tier", "fidelity_seeds", "margin"):
        if key not in screen:
            raise ValueError(f"policy.screen is missing {key!r}")
    return screen


def apply_screen_overrides(policy: dict[str, Any]) -> dict[str, Any]:
    """Returns a copy of `policy` configured for the screening stage.

    Screening uses the screen tier and seed count, accuracy smoke speed samples on that tier
    only, and the looser accuracy thresholds from `screened_accuracy_thresholds`. The returned
    policy still carries the original `screen` section for documentation and re-evaluation.
    """
    screen = screen_section(policy)
    tier = str(screen["tier"])
    seeds = int(screen["fidelity_seeds"])
    speed = {str(name): int(count) for name, count in dict(screen.get("speed_samples", {tier: 2})).items()}
    out = copy.deepcopy(policy)
    out["fidelity_tier"] = tier
    out["fidelity_seeds"] = seeds
    out["speed_tiers"] = [tier]
    out["speed_samples"] = {**dict(out.get("speed_samples", {})), **speed}
    out["speed_samples_accuracy"] = {**dict(out.get("speed_samples_accuracy", {})), **speed}
    out["accuracy"] = screened_accuracy_thresholds(out)
    return out


def screened_accuracy_thresholds(policy: dict[str, Any]) -> dict[str, Any]:
    """Returns accuracy thresholds loosened by `screen.margin`.

    Improvement floors (`min_mean_improvement`, `min_target_improvement`) are multiplied by
    `(1 - margin)`. Regression ceilings (`max_dataset_regression`, `max_structure_regression`)
    are multiplied by `(1 + margin)`. `ci_lower_must_exceed` is unchanged. Margin 0.5 therefore
    halves the required gain and raises allowed regressions by 50%, so borderline proposals that
    would be inconclusive or barely rejected under the official gate become `screen_pass`.
    """
    screen = screen_section(policy)
    margin = float(screen["margin"])
    if not 0.0 <= margin < 1.0:
        raise ValueError(f"screen.margin must be in [0, 1), got {margin}")
    accuracy = dict(policy.get("accuracy", {}))
    for key in ("min_mean_improvement", "min_target_improvement"):
        if key in accuracy:
            accuracy[key] = float(accuracy[key]) * (1.0 - margin)
    for key in ("max_dataset_regression", "max_structure_regression"):
        if key in accuracy:
            accuracy[key] = float(accuracy[key]) * (1.0 + margin)
    return accuracy


def policy_hash(policy: dict[str, Any]) -> str:
    """Returns the SHA-256 of the canonical policy JSON."""
    payload = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def policy_path_hash(path: Path) -> str:
    """Returns the SHA-256 of the on-disk policy file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
