"""Reads, writes, and describes profiles in JSON.

`load_profile` ignores unknown fields by default, so a reader accepts profiles
written against a newer minor spec version; `strict=True` rejects them, which
producers and tests use to check their output. Profiles with a different major
version are always rejected. `profile_json_schema` builds the JSON Schema that is
published as `profiler/spec/profile.schema.json`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from dbldatagen_profiler.model.profile import SPEC_VERSION, STRICT_CONTEXT_KEY, Profile


JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def load_profile(source: str | bytes | Mapping[str, Any], *, strict: bool = False) -> Profile:
    """Validates a profile from JSON text or a parsed JSON object.

    Args:
        source: JSON text, or an already parsed JSON object.
        strict: Optional flag that rejects unknown fields instead of ignoring them (default false).

    Returns:
        The validated `Profile`.

    Raises:
        ValueError: If `source` is not valid JSON or the profile does not follow the spec. Validation failures are
            raised as `pydantic.ValidationError`, a subclass of `ValueError`.
    """
    data = json.loads(source) if isinstance(source, str | bytes) else dict(source)
    return Profile.model_validate(data, context={STRICT_CONTEXT_KEY: strict})


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

    The schema accepts unknown fields so that readers in any language stay
    compatible with newer minor versions.

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
