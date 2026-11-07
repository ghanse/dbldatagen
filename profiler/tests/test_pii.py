from typing import Any

import pytest
from pydantic import ValidationError

from dbldatagen_profiler.model import Semantic, load_profile


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


def test_direct_identifier_rejects_literal_statistics(minimal_profile: dict[str, Any]) -> None:
    column = minimal_profile["tables"][0]["columns"][0]
    column["semantic"] = {
        "pii_class": "account_id",
        "sensitivity": "direct_identifier",
        "confidence": 0.9,
    }
    column["numeric"] = {"min": "1"}
    with pytest.raises(ValidationError, match="literal statistics"):
        load_profile(minimal_profile)


def test_quasi_identifier_sets_reference_columns(minimal_profile: dict[str, Any]) -> None:
    minimal_profile["tables"][0]["quasi_identifier_sets"] = [
        {"columns": ["id", "missing"], "unique_fraction": 0.9, "k_anonymity": 1}
    ]
    with pytest.raises(ValidationError, match="unknown columns"):
        load_profile(minimal_profile)
