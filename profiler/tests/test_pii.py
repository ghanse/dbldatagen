from pathlib import Path
from typing import Any

import jsonschema
import pytest
from pydantic import ValidationError

from dbldatagen_profiler.cli import _read_key
from dbldatagen_profiler.model import Semantic, load_profile, profile_json_schema


def test_semantic_evidence_contains_only_method_and_rate() -> None:
    semantic = Semantic.model_validate(
        {
            "pii_class": "email",
            "sensitivity": "direct_identifier",
            "confidence": 0.9,
            "evidence": [{"method": "email", "match_rate": 0.95}],
        },
        context={"strict": True},
    )
    assert semantic.model_dump(mode="json")["evidence"] == [{"method": "email", "match_rate": 0.95}]


@pytest.mark.parametrize(
    ("sensitivity", "stats", "message"),
    [
        ("direct_identifier", {"numeric": {"min": "1"}}, "literal statistics"),
        ("secret", {"numeric": {"max": "9"}}, "literal statistics"),
        (
            "direct_identifier",
            {
                "logical_type": "string",
                "source_type": "string",
                "string": {"top_values": [{"value": "raw", "frequency": 1.0}]},
            },
            "literal statistics",
        ),
        (
            "secret",
            {
                "logical_type": "string",
                "source_type": "string",
                "string": {"patterns": [{"regex": "secret-[0-9]+", "coverage": 1.0}]},
            },
            "character classes",
        ),
    ],
    ids=["direct-min", "secret-max", "direct-top-values", "secret-literal-pattern"],
)
def test_sensitive_columns_reject_literal_statistics(
    minimal_profile: dict[str, Any], sensitivity: str, stats: dict[str, Any], message: str
) -> None:
    column = minimal_profile["tables"][0]["columns"][0]
    column["semantic"] = {
        "pii_class": "account_id",
        "sensitivity": sensitivity,
        "confidence": 0.9,
    }
    column.update(stats)
    with pytest.raises(ValidationError, match=message):
        load_profile(minimal_profile)


def test_generalized_sensitive_pattern_is_accepted(minimal_profile: dict[str, Any]) -> None:
    minimal_profile["privacy"] = {"min_value_count": 1}
    column = minimal_profile["tables"][0]["columns"][0]
    column.update(
        logical_type="string",
        source_type="string",
        semantic={"pii_class": "secret_token", "sensitivity": "secret", "confidence": 1.0},
        string={"patterns": [{"regex": "[A-Za-z]+-[0-9]+", "coverage": 1.0}]},
    )
    load_profile(minimal_profile)


def test_quasi_identifier_sets_reference_columns(minimal_profile: dict[str, Any]) -> None:
    minimal_profile["tables"][0]["quasi_identifier_sets"] = [
        {"columns": ["id", "missing"], "unique_fraction": 0.9, "k_anonymity": 1}
    ]
    with pytest.raises(ValidationError, match="unknown columns"):
        load_profile(minimal_profile)


def test_0_1_profile_with_no_semantic_fields_still_loads(minimal_profile: dict[str, Any]) -> None:
    minimal_profile["spec_version"] = "0.1.0"
    assert load_profile(minimal_profile, strict=True).spec_version == "0.1.0"


def test_0_2_semantic_profile_validates_against_json_schema(minimal_profile: dict[str, Any]) -> None:
    minimal_profile["spec_version"] = "0.2.0"
    minimal_profile["tables"][0]["columns"][0]["semantic"] = {
        "pii_class": "account_id",
        "sensitivity": "direct_identifier",
        "confidence": 0.9,
        "evidence": [{"method": "sample_uniqueness", "match_rate": 1.0}],
    }
    parsed = load_profile(minimal_profile, strict=True)
    jsonschema.validate(parsed.model_dump(mode="json", exclude_none=True), profile_json_schema())


def test_key_file_requires_32_bytes(tmp_path: Path) -> None:
    path = tmp_path / "key"
    path.write_bytes(b"short")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="at least 32 bytes"):
        _read_key(path, allow_insecure=False)


def test_key_file_rejects_group_or_other_permissions_without_override(tmp_path: Path) -> None:
    path = tmp_path / "key"
    key = b"k" * 32
    path.write_bytes(key)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="readable by group or other"):
        _read_key(path, allow_insecure=False)
    assert _read_key(path, allow_insecure=True) == key
