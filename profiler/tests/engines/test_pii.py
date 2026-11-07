import ipaddress
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path

import jsonschema
import pytest
from pyspark.sql import DataFrame, Row, SparkSession

from dbldatagen_profiler.cli import main
from dbldatagen_profiler.engines.spark import profile_dataframe
from dbldatagen_profiler.model import (
    EngineInfo,
    KeyCandidate,
    PiiClass,
    PrivacyPolicy,
    Profile,
    Relationship,
    TableProfile,
    dump_profile,
    profile_json_schema,
)
from dbldatagen_profiler.pii import SubstitutionOptions, detect, substitute
from dbldatagen_profiler.pii.detect import analyze_table
from dbldatagen_profiler.pii.substitute import _mapping_surrogate


pytestmark = [pytest.mark.spark, pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")]


def test_detects_validators_and_negatives_deterministically(spark: SparkSession) -> None:
    rows = [(f"user{i}@example.com", "4532015112830366", f"SKU-{i:016d}", "1234567890123456") for i in range(20)]
    frame = spark.createDataFrame(rows, "email string, card_number string, sku string, random_digits string")
    first = detect({"t": frame.repartition(1)}, sample_rows=20)["t"]
    second = detect({"t": frame.repartition(4)}, sample_rows=20)["t"]
    assert first == second
    assert first["email"].pii_class.value == "email"
    assert first["card_number"].pii_class.value == "credit_card"
    assert "sku" not in first
    assert "random_digits" not in first
    assert all(
        set(item.model_dump()) == {"method", "match_rate"} for value in first.values() for item in value.evidence
    )


@pytest.mark.parametrize(
    ("name", "spark_type", "values", "pii_class", "sensitivity"),
    [
        ("email", "string", ["alice@example.com", "bob@example.org"], "email", "direct_identifier"),
        ("phone", "string", ["+14155552671", "+12125550199"], "phone", "direct_identifier"),
        ("full_name", "string", ["Alice Example", "Bob Example"], "person_name", "direct_identifier"),
        ("street_address", "string", ["1 Main St", "2 Oak Ave"], "street_address", "direct_identifier"),
        ("city", "string", ["Boston", "New York"], "city", "quasi_identifier"),
        ("postal_code", "string", ["02110", "10001"], "postal_code", "quasi_identifier"),
        ("ssn", "string", ["123-45-6789", "234-56-7890"], "national_id", "direct_identifier"),
        ("card_number", "string", ["4532015112830366", "6011111111111117"], "credit_card", "direct_identifier"),
        ("iban", "string", ["GB82WEST12345698765432", "DE89370400440532013000"], "iban", "direct_identifier"),
        ("ip_address", "string", ["192.0.2.1", "2001:db8::1"], "ip_address", "direct_identifier"),
        ("date_of_birth", "date", [date(1980, 1, 2), date(1990, 3, 4)], "date_of_birth", "quasi_identifier"),
        ("account_id", "string", ["acct-001", "acct-002"], "account_id", "direct_identifier"),
        (
            "secret_token",
            "string",
            ["yZ2mQ8vL5pR9tW3nK7xC", "aB4dE6gH8jK2mN5qS7uV"],
            "secret_token",
            "secret",
        ),
        (
            "notes",
            "string",
            ["contact alice@example.com today", "call +14155552671 tomorrow"],
            "free_text_pii",
            "free_text",
        ),
        ("other_pii", "string", ["custom-a", "custom-b"], "other", "direct_identifier"),
    ],
)
def test_detects_each_pii_class_with_expected_sensitivity(
    spark: SparkSession,
    name: str,
    spark_type: str,
    values: list[object],
    pii_class: str,
    sensitivity: str,
) -> None:
    frame = spark.createDataFrame([(value,) for value in values], f"{name} {spark_type}")
    semantic = detect({"t": frame})["t"][name]
    assert semantic.pii_class.value == pii_class
    assert semantic.sensitivity.value == sensitivity


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("sku", ["4532015112830366", "6011111111111117"]),
        ("random_digits", ["1234567890123456", "1111222233334444"]),
    ],
    ids=["sku-is-not-card", "non-luhn-is-not-card"],
)
def test_card_detection_negatives(spark: SparkSession, name: str, values: list[str]) -> None:
    semantic = detect({"t": spark.createDataFrame([(value,) for value in values], f"{name} string")})["t"].get(name)
    assert semantic is None or semantic.pii_class.value != "credit_card"


def test_uuid_primary_key_is_not_person_data(spark: SparkSession) -> None:
    values = ["550e8400-e29b-41d4-a716-446655440000", "123e4567-e89b-42d3-a456-426614174000"]
    semantic = detect({"t": spark.createDataFrame([(value,) for value in values], "primary_key string")})["t"]
    assert semantic["primary_key"].pii_class.value == "account_id"
    assert semantic["primary_key"].pii_class.value != "person_name"


def test_uuid_primary_and_foreign_key_preserve_join_integrity(spark: SparkSession) -> None:
    values = ["550e8400-e29b-41d4-a716-446655440000", "123e4567-e89b-42d3-a456-426614174000"]
    parents = spark.createDataFrame([(value,) for value in values], "entity_id string")
    children = spark.createDataFrame([(values[0],), (values[0],), (values[1],)], "entity_id string")
    semantic = detect({"parents": parents})["parents"]["entity_id"]
    assert semantic.pii_class.value == "account_id" and semantic.sensitivity.value == "direct_identifier"
    profile = _profile(
        [
            ("parents", 2, [_column("entity_id", semantic)]),
            ("children", 3, [_column("entity_id", None)]),
        ],
        relationships=[
            Relationship(
                child_table="children",
                child_columns=["entity_id"],
                parent_table="parents",
                parent_columns=["entity_id"],
                coverage=1.0,
            )
        ],
    )
    profile.tables[0].primary_key = KeyCandidate(columns=["entity_id"], confidence=1.0)
    output = substitute({"parents": parents, "children": children}, profile, key=b"uuid-domain-key").tables
    assert _relationship_stats(
        {"parents": parents, "children": children}, "children", ["entity_id"], "parents", ["entity_id"]
    ) == _relationship_stats(output, "children", ["entity_id"], "parents", ["entity_id"])
    report = json.dumps({"columns": [column.model_dump(mode="json") for column in profile.tables[0].columns]})
    assert "secret_token" not in report


def test_name_hint_without_matching_values_stays_low_confidence(spark: SparkSession) -> None:
    frame = spark.createDataFrame([("not-an-email",), ("still-not-an-email",)], "email string")
    semantic = detect({"t": frame})["t"]["email"]
    assert semantic.pii_class.value == "email"
    assert semantic.confidence <= 0.30


@pytest.mark.parametrize(("matching", "detected"), [(1, False), (2, True)], ids=["below-20-percent", "at-20-percent"])
def test_free_text_span_threshold(spark: SparkSession, matching: int, detected: bool) -> None:
    values = [f"ordinary note {index}" for index in range(10 - matching)] + [
        f"reach user{index}@example.com now" for index in range(matching)
    ]
    semantic = detect({"t": spark.createDataFrame([(value,) for value in values], "payload string")})["t"].get(
        "payload"
    )
    assert (semantic is not None and semantic.pii_class.value == "free_text_pii") is detected


def test_mid_threshold_whole_values_do_not_fall_through_to_free_text(spark: SparkSession) -> None:
    values = [f"user{index}@example.com" for index in range(5)] + [f"plain-{index}" for index in range(5)]
    semantic = detect({"t": spark.createDataFrame([(value,) for value in values], "value string")})["t"].get("value")
    assert semantic is None or semantic.pii_class.value != "free_text_pii"


def test_quasi_identifier_triple_is_found(spark: SparkSession) -> None:
    rows = [(f"{10000 + index % 5}", date(1980 + index, 1, 1), "F" if index % 2 else "M") for index in range(10)]
    frame = spark.createDataFrame(rows, "zip string, dob date, gender string")
    semantics, sets = analyze_table(frame)
    assert semantics["zip"].pii_class.value == "postal_code"
    assert (["zip", "dob", "gender"], 1.0, 1) in sets


def test_independent_low_cardinality_columns_are_not_quasi_identifiers(spark: SparkSession) -> None:
    rows = [(left, middle, right) for left in ("a", "b") for middle in ("x", "y") for right in ("m", "n")] * 2
    frame = spark.createDataFrame(rows, "left string, middle string, right string")
    _, sets = analyze_table(frame)
    assert sets == []


def test_evidence_serialization_contains_no_real_values(spark: SparkSession) -> None:
    real_values = ["alice@example.com", "bob@example.org"]
    semantics = detect({"customers": spark.createDataFrame([(value,) for value in real_values], "email string")})
    serialized = json.dumps(
        {
            table: {column: value.model_dump(mode="json") for column, value in columns.items()}
            for table, columns in semantics.items()
        }
    )
    assert all(value not in serialized for value in real_values)
    assert set(semantics["customers"]["email"].evidence[0].model_dump()) == {"method", "match_rate"}


def test_detected_profile_validates_against_schema(spark: SparkSession) -> None:
    frame = spark.createDataFrame([("alice@example.com",), ("bob@example.org",)], "email string")
    profile = profile_dataframe("customers", frame)
    jsonschema.validate(profile.model_dump(mode="json", exclude_none=True), profile_json_schema())


def test_nested_array_struct_email_is_detected_substituted_and_domain_consistent(spark: SparkSession) -> None:
    frame = spark.createDataFrame(
        [([Row(email="alice@example.com")], "alice@example.com")],
        "contacts array<struct<email:string>>, backup_email string",
    )
    profile = profile_dataframe("t", frame)
    contacts = profile.tables[0].columns[0]
    assert contacts.children is not None and contacts.children[0].children is not None
    assert contacts.children[0].children[0].semantic is not None
    assert contacts.children[0].children[0].semantic.pii_class.value == "email"
    options = SubstitutionOptions(domains={"email-domain": ["t.contacts.element.email", "t.backup_email"]})
    row = substitute({"t": frame}, profile, key=b"nested-email-key", options=options).tables["t"].first()
    assert row is not None
    assert row.contacts[0].email == row.backup_email
    assert row.contacts[0].email != "alice@example.com"


def test_nested_map_phone_value_is_detected_and_substituted(spark: SparkSession) -> None:
    frame = spark.createDataFrame([({"mobile": "+14155552671"},)], "phones map<string,string>")
    profile = profile_dataframe("t", frame)
    phones = profile.tables[0].columns[0]
    assert phones.children is not None and phones.children[1].semantic is not None
    assert phones.children[1].semantic.pii_class.value == "phone"
    row = substitute({"t": frame}, profile, key=b"nested-phone-key").tables["t"].first()
    assert row is not None
    assert row.phones["mobile"] != "+14155552671"
    assert row.phones["mobile"].startswith("+1415")


def test_nested_and_top_level_values_use_same_resolved_mapping_after_rehash(spark: SparkSession) -> None:
    options = SubstitutionOptions(
        domains={"shared-email": ["t.contacts.element.email", "t.primary_email"]}, surrogate_bits=2
    )
    candidates = [f"a{index}@example.com" for index in range(10)]
    by_surrogate: dict[str, list[str]] = {}
    for value in candidates:
        surrogate = _mapping_surrogate(PiiClass.EMAIL, value, "shared-email", b"nested-rehash-key", options, 0)
        by_surrogate.setdefault(surrogate, []).append(value)
    first, second = next(values[:2] for values in by_surrogate.values() if len(values) >= 2)
    frame = spark.createDataFrame(
        [([Row(email=first), Row(email=second), Row(email=first)], first)],
        "contacts array<struct<email:string>>, primary_email string",
    )
    profile = profile_dataframe("t", frame)
    result = substitute({"t": frame}, profile, key=b"nested-rehash-key", options=options)
    row = result.tables["t"].first()
    assert row is not None
    nested = [item.email for item in row.contacts]
    assert nested[0] == row.primary_email == nested[2]
    assert len(set(nested)) == 2
    assert set(nested).isdisjoint({first, second})
    assert all(item["collisions_resolved"] >= 2 for item in result.report["columns"])


def test_nested_map_keys_and_null_containers_preserve_shape_and_order(spark: SparkSession) -> None:
    first, second = "a@example.com", "b@example.com"
    frame = spark.createDataFrame(
        [
            ([Row(email=first), None, Row(email=second)], {first: "first", second: "second"}),
            (None, None),
            ([], {}),
        ],
        "contacts array<struct<email:string>>, lookup map<string,string>",
    )
    profile = profile_dataframe("t", frame)
    options = SubstitutionOptions(
        domains={"nested-email": ["t.contacts.element.email", "t.lookup.key"]}, surrogate_bits=2
    )
    rows = substitute({"t": frame}, profile, key=b"nested-shape-key", options=options).tables["t"].collect()
    populated = next(row for row in rows if row.contacts)
    assert populated.contacts[1] is None
    assert [item is None for item in populated.contacts] == [False, True, False]
    ordered_surrogates = [populated.contacts[0].email, populated.contacts[2].email]
    assert len(populated.lookup) == 2
    assert set(populated.lookup) == set(ordered_surrogates)
    assert set(populated.lookup.values()) == {"first", "second"}
    assert any(row.contacts is None and row.lookup is None for row in rows)
    assert any(row.contacts == [] and row.lookup == {} for row in rows)


def test_substitution_preserves_relationships_and_card_validity(spark: SparkSession) -> None:
    parents = spark.createDataFrame(
        [("a@example.com", "4532015112830366"), ("b@example.com", "4532015112830366")],
        "email string, card string",
    )
    children = spark.createDataFrame([("a@example.com",), ("a@example.com",), ("b@example.com",)], "email string")
    semantics = detect({"parents": parents, "children": children})
    parent_profile = TableProfile(
        name="parents",
        row_count=2,
        columns=[
            _column("email", semantics["parents"]["email"]),
            _column("card", semantics["parents"]["card"]),
        ],
    )
    child_profile = TableProfile(
        name="children", row_count=3, columns=[_column("email", semantics["children"]["email"])]
    )
    profile = Profile(
        spec_version="0.2.0",
        created_at="2026-11-07T12:00:00Z",
        engine=EngineInfo(name="test", version="1"),
        privacy=PrivacyPolicy(),
        tables=[parent_profile, child_profile],
        relationships=[
            Relationship(
                child_table="children",
                child_columns=["email"],
                parent_table="parents",
                parent_columns=["email"],
                coverage=1.0,
            )
        ],
    )
    result = substitute({"parents": parents, "children": children}, profile, key=b"test-key")
    parent_values = {row.email for row in result.tables["parents"].select("email").collect()}
    child_values = [row.email for row in result.tables["children"].select("email").collect()]
    assert set(child_values) == parent_values
    assert len(set(child_values)) == 2
    assert all(value not in {"a@example.com", "b@example.com"} for value in child_values)
    row = result.tables["parents"].first()
    assert row is not None
    card = row.card
    assert _luhn(card)
    assert result.report["columns"]


def test_generalization_and_collision_hook(spark: SparkSession) -> None:
    frame = spark.createDataFrame([("12345",), ("12346",)], "postal_code string")
    semantic = detect({"t": frame})["t"]["postal_code"]
    profile = Profile(
        spec_version="0.2.0",
        created_at="2026-11-07T12:00:00Z",
        engine=EngineInfo(name="test", version="1"),
        tables=[TableProfile(name="t", row_count=2, columns=[_column("postal_code", semantic)])],
    )
    result = substitute(
        {"t": frame}, profile, key=b"key", options=SubstitutionOptions(generalize={"t.postal_code": "postal_prefix"})
    )
    assert {row.postal_code for row in result.tables["t"].collect()} == {"123**"}


def test_unchanged_quasi_identifier_is_reported_as_skipped(spark: SparkSession) -> None:
    frame = spark.createDataFrame([("12345",), ("12346",)], "postal_code string")
    semantic = _semantic("postal_code", "quasi_identifier")
    profile = _profile([("t", 2, [_column("postal_code", semantic)])])
    result = substitute({"t": frame}, profile, key=b"quasi-key")
    assert {row.postal_code for row in result.tables["t"].collect()} == {"12345", "12346"}
    assert result.report["columns"][0]["skipped_reason"] == "quasi_identifier_unchanged"


def test_date_generalization_truncates_to_month_and_is_reported(spark: SparkSession) -> None:
    frame = spark.createDataFrame([(date(1980, 5, 17),)], "dob date")
    profile = _profile([("t", 1, [_typed_column("dob", "date", _semantic("date_of_birth", "quasi_identifier"))])])
    result = substitute(
        {"t": frame}, profile, key=b"generalize-key", options=SubstitutionOptions(generalize={"t.dob": "month"})
    )
    row = result.tables["t"].first()
    assert row is not None and row.dob == date(1980, 5, 1)
    assert result.report["columns"][0]["generalizations"] == ["month"]


def test_forced_collisions_are_resolved(spark: SparkSession) -> None:
    frame = spark.createDataFrame([(str(value),) for value in range(100, 108)], "account_id string")
    semantic = {
        "pii_class": "account_id",
        "sensitivity": "direct_identifier",
        "confidence": 1.0,
        "evidence": [{"method": "column_name", "match_rate": 1.0}],
    }
    profile = Profile(
        spec_version="0.2.0",
        created_at="2026-11-07T12:00:00Z",
        engine=EngineInfo(name="test", version="1"),
        tables=[TableProfile(name="t", row_count=8, columns=[_column("account_id", semantic)])],
    )
    result = substitute({"t": frame}, profile, key=b"collision-key", options=SubstitutionOptions(surrogate_bits=3))
    output = [row.account_id for row in result.tables["t"].collect()]
    assert len(set(output)) == 8
    assert not set(output) & {str(value) for value in range(100, 108)}
    assert result.report["columns"][0]["collisions_resolved"] > 0


def test_surrogate_that_matches_an_original_is_rehashed(spark: SparkSession) -> None:
    options = SubstitutionOptions(surrogate_bits=4)
    first = "42"
    survivor = _mapping_surrogate(PiiClass.ACCOUNT_ID, first, "t.account_id", b"survivor-key", options, 0)
    assert survivor != first
    values = [first, survivor]
    frame = spark.createDataFrame([(value,) for value in values], "account_id string")
    profile = _profile([("t", 2, [_column("account_id", _semantic("account_id", "direct_identifier"))])])
    result = substitute({"t": frame}, profile, key=b"survivor-key", options=options)
    output = {row.account_id for row in result.tables["t"].collect()}
    assert len(output) == 2
    assert output.isdisjoint(values)
    assert result.report["columns"][0]["collisions_resolved"] > 0


def test_too_small_surrogate_space_names_the_column(spark: SparkSession) -> None:
    values = [str(value) for value in range(10, 100)]
    frame = spark.createDataFrame([(value,) for value in values], "account_id string")
    profile = _profile([("t", len(values), [_column("account_id", _semantic("account_id", "direct_identifier"))])])
    with pytest.raises(ValueError, match=r"space is too small.*t\.account_id"):
        substitute({"t": frame}, profile, key=b"capacity-key")


@pytest.mark.parametrize(
    ("pii_class", "sensitivity", "original", "validator"),
    [
        ("email", "direct_identifier", "alice@example.com", lambda value: re.fullmatch(r"[^@]+@[^@]+\.[^@]+", value)),
        ("credit_card", "direct_identifier", "4532015112830366", lambda value: _luhn(value)),
        ("iban", "direct_identifier", "GB82WEST12345698765432", lambda value: _iban_valid(value)),
        ("national_id", "direct_identifier", "123-45-6789", lambda value: _ssn_valid(value)),
        ("ip_address", "direct_identifier", "2001:db8::1", lambda value: ipaddress.ip_address(value)),
        ("phone", "direct_identifier", "+14155552671", lambda value: re.fullmatch(r"\+1\d{10}", value)),
    ],
    ids=["email", "luhn-card", "mod97-iban", "ssn", "ip", "phone"],
)
def test_substitution_outputs_valid_values_per_class(
    spark: SparkSession,
    pii_class: str,
    sensitivity: str,
    original: str,
    validator: Callable[[str], object],
) -> None:
    frame = spark.createDataFrame([(original,)], "value string")
    profile = _profile([("t", 1, [_column("value", _semantic(pii_class, sensitivity))])])
    output = substitute({"t": frame}, profile, key=b"format-key").tables["t"].first()
    assert output is not None
    assert output.value != original
    assert validator(output.value)


def test_phone_country_and_area_code_distribution_is_preserved(spark: SparkSession) -> None:
    values = ["+14155552671", "+14155550123", "+442079460958", "+442079460123"]
    frame = spark.createDataFrame([(value,) for value in values], "phone string")
    profile = _profile([("t", len(values), [_column("phone", _semantic("phone", "direct_identifier"))])])
    output = [row.phone for row in substitute({"t": frame}, profile, key=b"phone-key").tables["t"].collect()]
    assert Counter(value[:5] for value in output) == Counter(value[:5] for value in values)


def test_substitution_preserves_null_distinct_and_frequency_statistics(spark: SparkSession) -> None:
    values = ["a@example.com", "a@example.com", "b@example.com", "c@example.com", None, ""]
    frame = spark.createDataFrame([(value,) for value in values], "email string")
    profile = _profile([("t", len(values), [_column("email", _semantic("email", "direct_identifier"))])])
    output = [row.email for row in substitute({"t": frame}, profile, key=b"stats-key").tables["t"].collect()]
    assert sum(value is None for value in output) == sum(value is None for value in values)
    assert len({value for value in output if value is not None}) == len(
        {value for value in values if value is not None}
    )
    assert sorted(Counter(output).values()) == sorted(Counter(values).values())
    assert {value for value in output if value}.isdisjoint(value for value in values if value)
    assert None in output and "" in output


def test_substitution_is_keyed_and_partition_deterministic(spark: SparkSession) -> None:
    values = [(f"account-{index:02d}",) for index in range(12)]
    frame = spark.createDataFrame(values, "account_id string")
    profile = _profile([("t", len(values), [_column("account_id", _semantic("account_id", "direct_identifier"))])])

    def output(key: bytes, partitions: int) -> list[str]:
        result = substitute({"t": frame.repartition(partitions)}, profile, key=key).tables["t"]
        return sorted(row.account_id for row in result.collect())

    assert output(b"same-key", 1) == output(b"same-key", 4)
    assert output(b"same-key", 2) != output(b"different-key", 2)


def test_three_table_composite_fk_integrity_and_fanout_are_preserved(spark: SparkSession) -> None:
    accounts = spark.createDataFrame(
        [("us", "acct-a"), ("us", "acct-b"), ("eu", "acct-a")], "region_id string, account_id string"
    )
    orders = spark.createDataFrame(
        [
            ("us", "acct-a", "order-1"),
            ("us", "acct-a", "order-2"),
            ("us", "acct-b", "order-3"),
            ("eu", "acct-a", "order-4"),
        ],
        "region_id string, account_id string, order_id string",
    )
    items = spark.createDataFrame(
        [
            ("us", "acct-a", "order-1", "item-a"),
            ("us", "acct-a", "order-1", "item-b"),
            ("us", "acct-a", "order-2", "item-c"),
            ("eu", "acct-a", "order-4", "item-d"),
        ],
        "region_id string, account_id string, order_id string, item_id string",
    )
    direct = _semantic("account_id", "direct_identifier")
    profile = _profile(
        [
            ("accounts", 3, [_column("region_id", direct), _column("account_id", direct)]),
            (
                "orders",
                4,
                [_column("region_id", None), _column("account_id", None), _column("order_id", direct)],
            ),
            (
                "items",
                4,
                [
                    _column("region_id", None),
                    _column("account_id", None),
                    _column("order_id", None),
                    _column("item_id", direct),
                ],
            ),
        ],
        relationships=[
            Relationship(
                child_table="orders",
                child_columns=["region_id", "account_id"],
                parent_table="accounts",
                parent_columns=["region_id", "account_id"],
                coverage=1.0,
            ),
            Relationship(
                child_table="items",
                child_columns=["region_id", "account_id", "order_id"],
                parent_table="orders",
                parent_columns=["region_id", "account_id", "order_id"],
                coverage=1.0,
            ),
        ],
    )
    source = {"accounts": accounts, "orders": orders, "items": items}
    output = substitute(source, profile, key=b"relational-key").tables
    relationships = [
        ("orders", ["region_id", "account_id"], "accounts", ["region_id", "account_id"]),
        ("items", ["region_id", "account_id", "order_id"], "orders", ["region_id", "account_id", "order_id"]),
    ]
    for child_table, child_columns, parent_table, parent_columns in relationships:
        assert _relationship_stats(
            source, child_table, child_columns, parent_table, parent_columns
        ) == _relationship_stats(output, child_table, child_columns, parent_table, parent_columns)
    originals = {value for frame in source.values() for row in frame.collect() for value in row if value}
    substituted = {value for frame in output.values() for row in frame.collect() for value in row if value}
    assert originals.isdisjoint(substituted)


def test_date_of_birth_shift_is_bounded_and_domain_consistent(spark: SparkSession) -> None:
    people = spark.createDataFrame(
        [("person-a", date(1980, 1, 2)), ("person-b", date(1980, 1, 2))], "entity_id string, dob date"
    )
    history = spark.createDataFrame(
        [("person-a", date(1980, 1, 2)), ("person-b", date(1980, 1, 2))], "entity_id string, dob date"
    )
    semantic = _semantic("date_of_birth", "quasi_identifier")
    profile = _profile(
        [
            ("people", 2, [_column("entity_id", None), _typed_column("dob", "date", semantic)]),
            ("history", 2, [_column("entity_id", None), _typed_column("dob", "date", semantic)]),
        ]
    )
    for table in profile.tables:
        table.primary_key = KeyCandidate(columns=["entity_id"], confidence=1.0)
    profile.relationships = [
        Relationship(
            child_table="history",
            child_columns=["entity_id"],
            parent_table="people",
            parent_columns=["entity_id"],
            coverage=1.0,
        )
    ]
    options = SubstitutionOptions(domains={"birth-domain": ["people.dob", "history.dob"]}, date_shift_days=7)
    substituted = substitute({"people": people, "history": history}, profile, key=b"date-key", options=options)
    people_output = {row.entity_id: row.dob for row in substituted.tables["people"].collect()}
    history_output = {row.entity_id: row.dob for row in substituted.tables["history"].collect()}
    assert people_output == history_output
    assert people_output["person-a"] != people_output["person-b"]
    assert all(abs((shifted - date(1980, 1, 2)).days) <= 7 for shifted in people_output.values())
    assert all(item.get("date_shift_key") == "entity" for item in substituted.report["columns"])


def test_free_text_replaces_only_detected_span(spark: SparkSession) -> None:
    original = "prefix bytes | alice@example.com | suffix bytes"
    frame = spark.createDataFrame([(original,)], "notes string")
    profile = _profile([("t", 1, [_column("notes", _semantic("free_text_pii", "free_text"))])])
    row = substitute({"t": frame}, profile, key=b"text-key").tables["t"].first()
    assert row is not None
    assert row.notes.startswith("prefix bytes | ") and row.notes.endswith(" | suffix bytes")
    assert "alice@example.com" not in row.notes


def test_keep_email_domain_option(spark: SparkSession) -> None:
    frame = spark.createDataFrame([("alice@rare.example",)], "email string")
    profile = _profile([("t", 1, [_column("email", _semantic("email", "direct_identifier"))])])
    default = substitute({"t": frame}, profile, key=b"mail-key").tables["t"].first()
    kept = (
        substitute({"t": frame}, profile, key=b"mail-key", options=SubstitutionOptions(keep_email_domain=True))
        .tables["t"]
        .first()
    )
    assert default is not None and kept is not None
    assert len(default.email) == len("alice@rare.example") == len(kept.email)
    assert default.email.endswith("@masked.invalid")
    assert kept.email.endswith("@rare.example")


def test_report_and_logs_never_contain_values_or_key_material(
    spark: SparkSession, caplog: pytest.LogCaptureFixture
) -> None:
    original, key = "alice@example.com", b"super-secret-test-key"
    frame = spark.createDataFrame([(original,)], "email string")
    profile = _profile([("t", 1, [_column("email", _semantic("email", "direct_identifier"))])])
    result = substitute({"t": frame}, profile, key=key)
    serialized = json.dumps(result.report)
    assert original not in serialized and key.decode() not in serialized
    assert original not in caplog.text and key.decode() not in caplog.text


def test_non_pii_profile_stats_are_identical_after_substitution(spark: SparkSession) -> None:
    frame = spark.createDataFrame(
        [("a@example.com", 10), ("b@example.com", 20), ("a@example.com", 20)], "email string, amount int"
    )
    profile = _profile(
        [("t", 3, [_column("email", _semantic("email", "direct_identifier")), _typed_column("amount", "int32", None)])]
    )
    output = substitute({"t": frame}, profile, key=b"profile-key").tables["t"]
    before = profile_dataframe("t", frame, detect_pii=False).tables[0].columns[1]
    after = profile_dataframe("t", output, detect_pii=False).tables[0].columns[1]
    assert before.null_fraction == after.null_fraction
    assert before.distinct_count == after.distinct_count
    assert before.numeric is not None and after.numeric is not None
    assert before.numeric.min == after.numeric.min and before.numeric.max == after.numeric.max
    assert before.numeric.mean == pytest.approx(after.numeric.mean)
    assert before.numeric.stddev == pytest.approx(after.numeric.stddev)


def test_cli_detect_and_substitute_end_to_end(spark: SparkSession, tmp_path: Path) -> None:
    source = tmp_path / "source"
    semantic_path = tmp_path / "semantic.json"
    profile_path = tmp_path / "profile.json"
    key_path = tmp_path / "key.bin"
    output = tmp_path / "output"
    frame = spark.createDataFrame([("a@example.com",), ("b@example.com",)], "email string")
    frame.write.parquet(str(source))
    assert main(["pii", "detect", "--table", f"customers={source}", "-o", str(semantic_path)]) == 0
    assert json.loads(semantic_path.read_text())["customers"]["email"]["pii_class"] == "email"
    profile_path.write_text(dump_profile(profile_dataframe("customers", frame)), encoding="utf-8")
    key_path.write_bytes(b"cli-test-key".ljust(32, b"!"))
    key_path.chmod(0o600)
    assert (
        main(
            [
                "pii",
                "substitute",
                "--table",
                f"customers={source}",
                "--profile",
                str(profile_path),
                "--key-file",
                str(key_path),
                "-o",
                str(output),
            ]
        )
        == 0
    )
    values = {row.email for row in spark.read.parquet(str(output / "customers")).collect()}
    assert values.isdisjoint({"a@example.com", "b@example.com"})
    assert json.loads((output / "report.json").read_text())["columns"][0]["values_substituted"] == 2


def _column(name: str, semantic: object) -> dict[str, object]:
    return {
        "name": name,
        "logical_type": "string",
        "source_type": "string",
        "nullable": True,
        "null_fraction": 0.0,
        "semantic": semantic,
    }


def _typed_column(name: str, logical_type: str, semantic: object) -> dict[str, object]:
    return {
        "name": name,
        "logical_type": logical_type,
        "source_type": "date" if logical_type == "date" else "int",
        "nullable": True,
        "null_fraction": 0.0,
        "semantic": semantic,
    }


def _semantic(pii_class: str, sensitivity: str) -> dict[str, object]:
    return {
        "pii_class": pii_class,
        "sensitivity": sensitivity,
        "confidence": 1.0,
        "evidence": [{"method": "test_fixture", "match_rate": 1.0}],
    }


def _profile(
    tables: list[tuple[str, int, list[dict[str, object]]]], relationships: list[Relationship] | None = None
) -> Profile:
    return Profile(
        spec_version="0.2.0",
        created_at="2026-11-07T12:00:00Z",
        engine=EngineInfo(name="test", version="1"),
        tables=[TableProfile(name=name, row_count=row_count, columns=columns) for name, row_count, columns in tables],
        relationships=relationships,
    )


def _relationship_stats(
    tables: Mapping[str, DataFrame],
    child_table: str,
    child_columns: list[str],
    parent_table: str,
    parent_columns: list[str],
) -> tuple[int, int, list[int]]:
    parents = {tuple(row[column] for column in parent_columns) for row in tables[parent_table].collect()}
    children = Counter(tuple(row[column] for column in child_columns) for row in tables[child_table].collect())
    matched = sum(count for key, count in children.items() if key in parents)
    return matched, sum(children.values()), sorted(children.get(key, 0) for key in parents)


def _iban_valid(value: str) -> bool:
    compact = re.sub(r"\s", "", value).upper()
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(ord(char) - 55) if char.isalpha() else char for char in rearranged)
    return int(numeric) % 97 == 1


def _ssn_valid(value: str) -> bool:
    return re.fullmatch(r"(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}", value) is not None


def _luhn(value: str) -> bool:
    digits = list(map(int, re.sub(r"\D", "", value)))
    total = 0
    for index, digit in enumerate(reversed(digits)):
        number = digit * 2 if index % 2 else digit
        total += number - 9 if number > 9 else number
    return total % 10 == 0
