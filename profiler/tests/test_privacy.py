import copy
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


def array_profile(
    minimal_profile: dict[str, Any], string: dict[str, Any], value_count: int | None, row_count: int = 1000
) -> dict[str, Any]:
    """Returns `minimal_profile` with an array<string> column whose elements carry `string` stats."""
    element = {"name": "element", "logical_type": "string", "source_type": "string", "nullable": False}
    element |= {"null_fraction": 0.0, "string": string}
    if value_count is not None:
        element["value_count"] = value_count
    profile = profile_with(minimal_profile, logical_type="array", source_type="array<string>", children=[element])
    profile["tables"][0]["row_count"] = row_count
    return profile


@pytest.mark.parametrize(
    ("value_count", "frequency", "ok"),
    [
        (5000, 0.002, True),
        (100, 0.05, False),
        (100, 0.1, True),
        (None, 0.5, False),
    ],
    ids=["more-elements-than-rows", "fewer-elements-than-rows-rare", "fewer-elements-than-rows-ok", "missing"],
)
def test_nested_top_values_use_value_count(
    minimal_profile: dict[str, Any], value_count: int | None, frequency: float, ok: bool
) -> None:
    profile = array_profile(minimal_profile, top_values(frequency), value_count)
    if ok:
        load_profile(profile)
    else:
        with pytest.raises(ValidationError, match=r"value_count|fewer than 10"):
            load_profile(profile)


@pytest.mark.parametrize(
    ("value_count", "coverage", "ok"),
    [(20000, 0.001, True), (20, 0.4, False), (20, 0.5, True), (None, 1.0, False)],
    ids=["more-elements-than-rows", "fewer-elements-than-rows-rare", "fewer-elements-than-rows-ok", "missing"],
)
def test_nested_patterns_use_value_count(
    minimal_profile: dict[str, Any], value_count: int | None, coverage: float, ok: bool
) -> None:
    patterns = {"patterns": [{"regex": "[a-z]+", "coverage": coverage}]}
    profile = array_profile(minimal_profile, patterns, value_count)
    if ok:
        load_profile(profile)
    else:
        with pytest.raises(ValidationError, match=r"value_count|fewer than 10"):
            load_profile(profile)


def test_nested_value_count_wins_over_table_rows(minimal_profile: dict[str, Any]) -> None:
    profile = array_profile(minimal_profile, top_values(0.01), value_count=500, row_count=1_000_000)
    with pytest.raises(ValidationError, match="fewer than 10"):
        load_profile(profile)


def test_struct_fields_inherit_value_count(minimal_profile: dict[str, Any]) -> None:
    field: dict[str, Any] = {"name": "zip", "logical_type": "string", "source_type": "string", "nullable": True}
    field |= {"null_fraction": 0.0, "string": {"patterns": [{"regex": "[0-9]{5}", "coverage": 0.02}]}}
    struct: dict[str, Any] = {"logical_type": "struct", "source_type": "struct<zip:string>", "children": [field]}
    load_profile(profile_with(copy.deepcopy(minimal_profile), **struct))
    field["string"]["patterns"][0]["coverage"] = 0.005
    with pytest.raises(ValidationError, match="fewer than 10"):
        load_profile(profile_with(minimal_profile, **struct))


def test_struct_inside_array_needs_element_value_count(minimal_profile: dict[str, Any]) -> None:
    field = {"name": "f", "logical_type": "string", "source_type": "string", "nullable": True, "null_fraction": 0.0}
    field["string"] = {"patterns": [{"regex": "x", "coverage": 1.0}]}
    element = {"name": "element", "logical_type": "struct", "source_type": "struct<f:string>", "nullable": True}
    element |= {"null_fraction": 0.0, "children": [field]}
    profile = profile_with(
        minimal_profile, logical_type="array", source_type="array<struct<f:string>>", children=[element]
    )
    with pytest.raises(ValidationError, match="value_count"):
        load_profile(profile)
    element["value_count"] = 50
    load_profile(profile)


def test_value_count_is_rejected_on_top_level_columns(minimal_profile: dict[str, Any]) -> None:
    minimal_profile["tables"][0]["columns"][0]["value_count"] = 2
    with pytest.raises(ValidationError, match="nested"):
        load_profile(minimal_profile)


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
