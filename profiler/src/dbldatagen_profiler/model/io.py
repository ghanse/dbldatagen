"""Reads, writes, and describes profiles in JSON.

`load_profile` applies the spec's version rules: profiles with a different major
version are rejected, and profiles with a newer minor version are read with
unknown fields dropped. `profile_json_schema` builds the JSON Schema that is
published as `profiler/spec/profile.schema.json`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from dbldatagen_profiler.model.profile import LENIENT_CONTEXT_KEY, SPEC_VERSION, Profile, parse_version


JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def load_profile(source: str | bytes | Mapping[str, Any]) -> Profile:
    """Validates a profile from JSON text or a parsed JSON object.

    Args:
        source: JSON text, or an already parsed JSON object.

    Returns:
        The validated `Profile`.

    Raises:
        ValueError: If `source` is not valid JSON.
        pydantic.ValidationError: If the profile does not follow the spec.
    """
    data = json.loads(source) if isinstance(source, str | bytes) else dict(source)
    version = data.get("spec_version") if isinstance(data, dict) else None
    lenient = isinstance(version, str) and _is_newer_minor(version)
    return Profile.model_validate(data, context={LENIENT_CONTEXT_KEY: lenient})


def dump_profile(profile: Profile) -> str:
    """Serializes a profile to indented JSON, leaving out unset optional fields.

    Args:
        profile: Profile to serialize.

    Returns:
        JSON text.
    """
    return profile.model_dump_json(indent=2, exclude_none=True)


def profile_json_schema() -> dict[str, Any]:
    """Builds the JSON Schema for the current spec version.

    Returns:
        The JSON Schema as a dictionary.
    """
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "$comment": f"dbldatagen profile spec {SPEC_VERSION}. Generated from dbldatagen_profiler.model; do not edit.",
        **Profile.model_json_schema(),
    }


def render_json_schema() -> str:
    """Renders the JSON Schema as the text stored in `profiler/spec/profile.schema.json`.

    Returns:
        Indented JSON text ending in a newline.
    """
    return json.dumps(profile_json_schema(), indent=2) + "\n"


def _is_newer_minor(version: str) -> bool:
    """Returns whether `version` has the supported major version and a newer minor version."""
    try:
        major, minor, _ = parse_version(version)
    except ValueError:
        return False
    supported_major, supported_minor, _ = parse_version(SPEC_VERSION)
    return major == supported_major and minor > supported_minor
