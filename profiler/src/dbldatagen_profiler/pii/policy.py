"""Configurable handling policy for detected sensitive data."""

from __future__ import annotations

import json
from enum import Enum
from importlib import import_module
from pathlib import Path
from types import ModuleType
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
    pii_class: PiiClass | None = None
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_parameters(self) -> PiiRule:
        if self.pii_class is not None and self.action != PiiAction.SUBSTITUTE:
            raise ValueError("pii_class is only valid for substitute rules")
        if self.action == PiiAction.GENERALIZE and not self.parameters:
            raise ValueError("generalize requires parameters")
        if self.action != PiiAction.GENERALIZE and self.parameters:
            raise ValueError(f"{self.action.value} does not accept parameters")
        return self


class PiiPolicy(BaseModel):
    """Rules for handling detected sensitive data, from broadest to most specific."""

    model_config = ConfigDict(extra="forbid")

    min_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
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
        if any(rule.pii_class is not None for rule in [*self.sensitivity.values(), *self.pii_class.values()]):
            raise ValueError("pii_class may only appear on a column rule")
        return self

    def validate_columns(self, references: set[str]) -> None:
        """Rejects fully-qualified column rules that match no input column."""
        unknown = sorted(set(self.columns) - references)
        if unknown:
            raise ValueError(f"PII policy column rules match no input column: {unknown}")

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
        yaml = _yaml_module()
        return str(yaml.safe_dump(self.model_dump(mode="json", exclude_none=True), sort_keys=False))


def load_policy(source: str | bytes | Path | dict[str, Any]) -> PiiPolicy:
    """Loads a policy from YAML, JSON, a path, or an already parsed mapping."""
    if isinstance(source, Path):
        payload = _load_text(source.read_text(encoding="utf-8"))
    elif isinstance(source, bytes):
        payload = _load_text(source.decode())
    elif isinstance(source, str):
        payload = _load_text(source)
    else:
        payload = source
    if not isinstance(payload, dict):
        raise ValueError("PII policy must be a YAML or JSON mapping")
    return PiiPolicy.model_validate(payload)


def dump_policy_json(policy: PiiPolicy) -> str:
    """Serializes a policy to stable JSON."""
    return json.dumps(policy.model_dump(mode="json", exclude_none=True), indent=2) + "\n"


def _load_text(text: str) -> object:
    """Loads JSON, or YAML when the optional YAML extra is installed."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _yaml_module().safe_load(text)


def _yaml_module() -> ModuleType:
    """Returns PyYAML or raises an actionable optional-extra error."""
    try:
        return import_module("yaml")
    except ModuleNotFoundError as error:
        raise ValueError("YAML policy support requires dbldatagen-profiler[yaml]; JSON policies always work") from error
