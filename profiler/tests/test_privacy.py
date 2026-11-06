from typing import Any

import pytest
from pydantic import ValidationError

from dbldatagen_profiler.model import load_profile


def profile_with(
    minimal_profile: dict[str, Any], privacy: dict[str, Any] | None = None, **column_overrides: object
) -> dict[str, Any]:
    """Returns `minimal_profile` with 1000 rows, a nullable string column `s`, and an optional privacy block."""
    table = minimal_profile["tables"][0]
    table["row_count"] = 1000
    s = {"name": "s", "logical_type": "string", "source_type": "string", "nullable": True, "null_fraction": 0.5}
    table["columns"].append({**s, **column_overrides})
    if privacy is not None:
        minimal_profile["privacy"] = privacy
    return minimal_profile


def top_values(*frequencies: float) -> dict[str, Any]:
    """Returns string stats with one top value per frequency."""
    return {"top_values": [{"value": f"v{i}", "frequency": f} for i, f in enumerate(frequencies)]}


def enumeration(min_value_count: int) -> dict[str, Any]:
    """Returns an enumeration constraint on column `s`."""
    return {
        "kind": "enumeration",
        "column": "s",
        "values": ["a", "b"],
        "min_value_count": min_value_count,
        "satisfied_fraction": 1.0,
    }


def test_default_policy_is_conservative(minimal_profile: dict[str, Any]) -> None:
    privacy = load_profile(minimal_profile).privacy
    assert (privacy.min_value_count, privacy.extremes.value, privacy.enumerations.value) == (10, "exact", "exact")
    assert privacy.allow_expression_constraints is False


def test_top_values_meeting_k_are_accepted(minimal_profile: dict[str, Any]) -> None:
    load_profile(profile_with(minimal_profile, string=top_values(0.5, 0.02)))


def test_top_values_below_default_k_are_rejected(minimal_profile: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="fewer than 10 rows"):
        load_profile(profile_with(minimal_profile, string=top_values(0.5, 0.018)))


@pytest.mark.parametrize(("k", "frequency", "ok"), [(1, 0.002, True), (100, 0.1, False), (100, 0.2, True)])
def test_k_is_taken_from_the_policy(minimal_profile: dict[str, Any], k: int, frequency: float, ok: bool) -> None:
    profile = profile_with(minimal_profile, {"min_value_count": k}, string=top_values(frequency))
    if ok:
        load_profile(profile)
    else:
        with pytest.raises(ValidationError):
            load_profile(profile)


def test_enumeration_below_k_is_rejected(minimal_profile: dict[str, Any]) -> None:
    profile = profile_with(minimal_profile)
    profile["tables"][0]["check_constraints"] = [enumeration(9)]
    with pytest.raises(ValidationError, match="fewer than 10 rows"):
        load_profile(profile)
    profile["tables"][0]["check_constraints"] = [enumeration(10)]
    load_profile(profile)


def test_nested_top_values_are_rejected(minimal_profile: dict[str, Any]) -> None:
    element = {"name": "element", "logical_type": "string", "source_type": "string", "nullable": False}
    element |= {"null_fraction": 0.0, "string": top_values(0.5)}
    profile = profile_with(minimal_profile, logical_type="array", source_type="array<string>", children=[element])
    with pytest.raises(ValidationError, match="top-level"):
        load_profile(profile)


def test_suppressed_enumerations_forbid_top_values(minimal_profile: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="suppressed"):
        load_profile(profile_with(minimal_profile, {"enumerations": "suppressed"}, string=top_values(0.5)))


def test_suppressed_enumerations_forbid_enumeration_constraints(minimal_profile: dict[str, Any]) -> None:
    profile = profile_with(minimal_profile, {"enumerations": "suppressed"})
    profile["tables"][0]["check_constraints"] = [enumeration(100)]
    with pytest.raises(ValidationError, match="suppressed"):
        load_profile(profile)


def test_suppressed_extremes_forbid_min_max(minimal_profile: dict[str, Any]) -> None:
    minimal_profile["privacy"] = {"extremes": "suppressed"}
    minimal_profile["tables"][0]["columns"][0]["numeric"] = {"mean": 1.0}
    load_profile(minimal_profile)
    minimal_profile["tables"][0]["columns"][0]["numeric"] = {"max": "5"}
    with pytest.raises(ValidationError, match="suppressed"):
        load_profile(minimal_profile)


def test_strict_privacy_profile_may_omit_all_literals(minimal_profile: dict[str, Any]) -> None:
    privacy = {"min_value_count": 50, "extremes": "suppressed", "enumerations": "suppressed"}
    load_profile(profile_with(minimal_profile, privacy, string={"length": {"mean": 4.0}}), strict=True)


def numeric_column(**numeric: object) -> dict[str, Any]:
    """Returns a float64 column with the given numeric stats."""
    column = {"name": "x", "logical_type": "float64", "source_type": "double", "nullable": False}
    return {**column, "null_fraction": 0.0, "numeric": numeric}


@pytest.mark.parametrize(
    ("numeric", "ok"),
    [
        ({"quantiles": [{"q": 0.01, "value": 1.0}, {"q": 0.99, "value": 9.0}]}, True),
        ({"quantiles": [{"q": 0.0, "value": 0.0}]}, False),
        ({"quantiles": [{"q": 1.0, "value": 10.0}]}, False),
        ({"histogram": {"edges": [0, 5, 10], "counts": [1, 1], "generalized_outer_edges": True}}, True),
        ({"histogram": {"edges": [0.3, 5, 9.7], "counts": [1, 1]}}, False),
    ],
)
def test_suppressed_extremes_cover_quantiles_and_histogram_edges(
    minimal_profile: dict[str, Any], numeric: dict[str, Any], ok: bool
) -> None:
    minimal_profile["privacy"] = {"extremes": "suppressed"}
    minimal_profile["tables"][0]["columns"].append(numeric_column(**numeric))
    if ok:
        load_profile(minimal_profile)
    else:
        with pytest.raises(ValidationError, match="suppressed"):
            load_profile(minimal_profile)


def test_extreme_quantiles_and_exact_edges_are_allowed_by_default(minimal_profile: dict[str, Any]) -> None:
    histogram = {"edges": [0.3, 9.7], "counts": [2]}
    quantiles = [{"q": 0.0, "value": 0.3}, {"q": 1.0, "value": 9.7}]
    minimal_profile["tables"][0]["columns"].append(numeric_column(quantiles=quantiles, histogram=histogram))
    load_profile(minimal_profile)


@pytest.mark.parametrize(("coverage", "ok"), [(0.02, True), (0.018, False)])
def test_patterns_must_cover_k_rows(minimal_profile: dict[str, Any], coverage: float, ok: bool) -> None:
    patterns = {"patterns": [{"regex": "[a-z]+", "coverage": 0.9}, {"regex": "ACME-[0-9]{4}", "coverage": coverage}]}
    profile = profile_with(minimal_profile, string=patterns)
    if ok:
        load_profile(profile)
    else:
        with pytest.raises(ValidationError, match="ACME"):
            load_profile(profile)


@pytest.mark.parametrize(("privacy", "ok"), [(None, False), ({"allow_expression_constraints": True}, True)])
def test_expression_constraints_need_explicit_permission(
    minimal_profile: dict[str, Any], privacy: dict[str, Any] | None, ok: bool
) -> None:
    profile = profile_with(minimal_profile, privacy)
    expression = {"kind": "expression", "expression": "s <> 'secret'", "satisfied_fraction": 1.0}
    profile["tables"][0]["check_constraints"] = [expression]
    if ok:
        load_profile(profile)
    else:
        with pytest.raises(ValidationError, match="expression"):
            load_profile(profile)
