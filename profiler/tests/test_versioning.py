from typing import Any

import pytest
from pydantic import ValidationError

from dbldatagen_profiler.model import SPEC_VERSION, Profile, load_profile, parse_version


def with_extras(profile: dict[str, Any], version: str) -> dict[str, Any]:
    """Returns `profile` at `version` with unknown fields at the profile, table, and column level."""
    profile["spec_version"] = version
    profile["future_block"] = {"a": 1}
    profile["tables"][0]["future_table_field"] = True
    profile["tables"][0]["columns"][0]["future_column_field"] = [1, 2]
    return profile


def test_spec_version_is_semver() -> None:
    assert parse_version(SPEC_VERSION) == (0, 1, 0)


@pytest.mark.parametrize("version", ["0.1", "v0.1.0", "0.1.0-rc1", ""])
def test_malformed_version_is_rejected(minimal_profile: dict[str, Any], version: str) -> None:
    minimal_profile["spec_version"] = version
    with pytest.raises(ValidationError):
        load_profile(minimal_profile)


def test_other_major_version_is_rejected(minimal_profile: dict[str, Any]) -> None:
    minimal_profile["spec_version"] = "1.0.0"
    with pytest.raises(ValidationError, match="major version"):
        load_profile(minimal_profile)


@pytest.mark.parametrize("version", ["0.1.0", "0.1.7", "0.0.9"])
def test_unknown_fields_are_rejected_up_to_supported_minor(minimal_profile: dict[str, Any], version: str) -> None:
    with pytest.raises(ValidationError):
        load_profile(with_extras(minimal_profile, version))


def test_newer_minor_version_drops_unknown_fields(minimal_profile: dict[str, Any]) -> None:
    profile = load_profile(with_extras(minimal_profile, "0.2.3"))
    assert profile.spec_version == "0.2.3"
    assert profile.model_extra is None
    assert "future_column_field" not in profile.tables[0].columns[0].model_dump()


def test_direct_model_validation_is_strict(minimal_profile: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Profile.model_validate(with_extras(minimal_profile, "0.2.0"))


def test_newer_minor_version_still_rejects_unknown_enum_values(minimal_profile: dict[str, Any]) -> None:
    minimal_profile["spec_version"] = "0.2.0"
    minimal_profile["tables"][0]["columns"][0]["logical_type"] = "float128"
    with pytest.raises(ValidationError):
        load_profile(minimal_profile)
