from typing import Any

import pytest
from pydantic import ValidationError

from dbldatagen_profiler.model import SPEC_VERSION, load_profile


def with_extras(profile: dict[str, Any], version: str) -> dict[str, Any]:
    """Returns `profile` at `version` with unknown fields at the profile, table, and column level."""
    profile["spec_version"] = version
    profile["future_block"] = {"a": 1}
    profile["tables"][0]["future_table_field"] = True
    profile["tables"][0]["columns"][0]["future_column_field"] = [1, 2]
    return profile


def test_spec_version_is_0_1_0() -> None:
    assert SPEC_VERSION == "0.1.0"


@pytest.mark.parametrize("version", ["0.1", "v0.1.0", "0.1.0-rc1", ""])
def test_malformed_version_is_rejected(minimal_profile: dict[str, Any], version: str) -> None:
    minimal_profile["spec_version"] = version
    with pytest.raises(ValidationError):
        load_profile(minimal_profile)


@pytest.mark.parametrize("version", ["1.0.0", "2.1.0", "10.0.0"])
def test_other_major_version_is_rejected(minimal_profile: dict[str, Any], version: str) -> None:
    minimal_profile["spec_version"] = version
    with pytest.raises(ValidationError, match="spec_version"):
        load_profile(minimal_profile)


@pytest.mark.parametrize("version", ["0.1.0", "0.2.3", "0.17.0"])
def test_unknown_fields_are_ignored_by_default(minimal_profile: dict[str, Any], version: str) -> None:
    profile = load_profile(with_extras(minimal_profile, version))
    assert profile.spec_version == version
    assert "future_column_field" not in profile.tables[0].columns[0].model_dump()


@pytest.mark.parametrize("version", ["0.1.0", "0.2.3"])
def test_strict_mode_rejects_unknown_fields(minimal_profile: dict[str, Any], version: str) -> None:
    with pytest.raises(ValidationError, match="unknown fields"):
        load_profile(with_extras(minimal_profile, version), strict=True)


def test_strict_mode_reports_nested_unknown_fields(minimal_profile: dict[str, Any]) -> None:
    minimal_profile["tables"][0]["columns"][0]["typo_field"] = 1
    with pytest.raises(ValidationError, match="typo_field"):
        load_profile(minimal_profile, strict=True)


def test_unknown_enum_values_are_still_rejected(minimal_profile: dict[str, Any]) -> None:
    minimal_profile["spec_version"] = "0.2.0"
    minimal_profile["tables"][0]["columns"][0]["logical_type"] = "float128"
    with pytest.raises(ValidationError):
        load_profile(minimal_profile)
