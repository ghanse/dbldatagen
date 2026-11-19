"""Configurable handling policy for detected sensitive data."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dbldatagen_profiler.model import PiiClass, Semantic, Sensitivity


class PiiAction(str, Enum):
    """Action applied to a detected sensitive column or nested element."""

    SUBSTITUTE = "substitute"
    GENERALIZE = "generalize"
    SUPPRESS = "suppress"
    DROP = "drop"
    KEEP = "keep"


class PiiRule(BaseModel):
    """Action and optional action-specific parameters."""

    model_config = ConfigDict(extra="forbid")

    action: PiiAction
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_parameters(self) -> PiiRule:
        if self.action == PiiAction.GENERALIZE and not self.parameters:
            raise ValueError("generalize requires parameters")
        if self.action != PiiAction.GENERALIZE and self.parameters:
            raise ValueError(f"{self.action.value} does not accept parameters")
        return self


class PiiPolicy(BaseModel):
    """Rules for handling detected sensitive data, from broadest to most specific."""

    model_config = ConfigDict(extra="forbid")

    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    sensitivity: dict[Sensitivity, PiiRule] = Field(
        default_factory=lambda: {kind: PiiRule(action=PiiAction.SUBSTITUTE) for kind in Sensitivity}
    )
    pii_class: dict[PiiClass, PiiRule] = Field(default_factory=dict)
    columns: dict[str, PiiRule] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_all_sensitivities(self) -> PiiPolicy:
        missing = sorted(kind.value for kind in Sensitivity if kind not in self.sensitivity)
        if missing:
            raise ValueError(f"sensitivity rules are missing {missing}")
        return self

    def resolve(self, reference: str, semantic: Semantic) -> tuple[PiiRule, str]:
        """Returns the most specific matching rule and its source."""
        if reference in self.columns:
            return self.columns[reference], "column"
        if semantic.pii_class in self.pii_class:
            return self.pii_class[semantic.pii_class], "pii_class"
        if semantic.sensitivity in self.sensitivity:
            return self.sensitivity[semantic.sensitivity], "sensitivity"
        return PiiRule(action=PiiAction.SUBSTITUTE), "default"

    def has_override(self, reference: str, semantic: Semantic) -> bool:
        """Returns whether a class or column override explicitly names the detection."""
        return reference in self.columns or semantic.pii_class in self.pii_class

    def to_yaml(self) -> str:
        """Serializes the policy to stable YAML."""
        return _dump_yaml_mapping(self.model_dump(mode="json"))


def load_policy(source: str | bytes | Path | dict[str, Any]) -> PiiPolicy:
    """Loads a policy from YAML, JSON, a path, or an already parsed mapping."""
    if isinstance(source, Path):
        payload = _load_yaml(source.read_text(encoding="utf-8"))
    elif isinstance(source, bytes):
        payload = _load_yaml(source.decode())
    elif isinstance(source, str):
        candidate = Path(source)
        if "\n" not in source and candidate.is_file():
            payload = _load_yaml(candidate.read_text(encoding="utf-8"))
        else:
            payload = _load_yaml(source)
    else:
        payload = source
    if not isinstance(payload, dict):
        raise ValueError("PII policy must be a YAML or JSON mapping")
    return PiiPolicy.model_validate(payload)


def dump_policy_json(policy: PiiPolicy) -> str:
    """Serializes a policy to stable JSON."""
    return json.dumps(policy.model_dump(mode="json"), indent=2) + "\n"


def _load_yaml(text: str) -> object:
    """Loads the mapping-only YAML subset used by policy files."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.lstrip(" ")
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(stripped)
        key, separator, value = stripped.partition(":")
        if not separator or not key.strip():
            raise ValueError(f"invalid policy YAML on line {line_number}")
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        key = key.strip()
        value = value.strip()
        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _yaml_scalar(value)
    return root


def _yaml_scalar(value: str) -> object:
    if value.startswith("{") and value.endswith("}"):
        entries = [item.strip() for item in value[1:-1].split(",") if item.strip()]
        return {key.strip(): _yaml_scalar(item.strip()) for entry in entries for key, _, item in [entry.partition(":")]}
    if value in {"true", "false"}:
        return value == "true"
    if value in {"null", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip("'\"")


def _dump_yaml_mapping(mapping: dict[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in mapping.items():
        if isinstance(value, dict):
            if value:
                lines.append(f"{prefix}{key}:")
                lines.append(_dump_yaml_mapping(value, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}{key}: {{}}")
        else:
            lines.append(f"{prefix}{key}: {_dump_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def _dump_yaml_scalar(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return str(value)
