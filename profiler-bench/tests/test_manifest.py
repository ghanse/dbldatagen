from typing import Any

import pytest
from pydantic import ValidationError

from profiler_bench.catalog import DATASETS, Manifest, generate
from profiler_bench.catalog.manifest import (
    Column,
    Distribution,
    Family,
    ForeignKey,
    Format,
    QuasiIdentifierSet,
    RangeCheck,
    Table,
)


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_manifest_round_trips_through_json(name: str) -> None:
    _, truth = generate(DATASETS[name], "xs", 0)
    assert Manifest.model_validate_json(truth.model_dump_json()) == truth


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_manifest_types_every_column(name: str) -> None:
    _, truth = generate(DATASETS[name], "xs", 0)

    def untyped(columns: tuple[Column, ...]) -> list[str]:
        return [c.name for c in columns if not c.type] + [n for c in columns for n in untyped(c.children())]

    assert all(t.row_count >= 0 for t in truth.tables)
    assert [n for t in truth.tables for n in untyped(t.columns)] == []


@pytest.mark.parametrize(
    ("family", "params"),
    [
        (Family.NORMAL, {"mean": 0.0}),
        (Family.NORMAL, {"mean": 0.0, "std": 1.0, "skew": 1.0}),
        (Family.CATEGORICAL, {"values": ["a", "b"], "probabilities": [0.5, 0.6]}),
        (Family.CATEGORICAL, {"values": ["a"], "probabilities": [0.5, 0.5]}),
        (Family.MIXTURE, {"components": [{"weight": 0.5, "mean": 0.0, "std": 1.0}]}),
        (Family.SEASONAL, {"start": "a", "end": "b", "hour_weights": [1] * 23, "weekday_weights": [1] * 7}),
    ],
)
def test_distribution_rejects_bad_params(family: Family, params: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Distribution(family=family, params=params)


def test_column_rejects_format_shares_above_one() -> None:
    with pytest.raises(ValidationError, match="format shares"):
        Column(name="c", formats=(Format(regex="a", share=0.6), Format(regex="b", share=0.6)))


def test_table_rejects_unknown_column_references() -> None:
    with pytest.raises(ValidationError, match="unknown columns"):
        Table(name="t", columns=(Column(name="a"),), primary_key=("b",))


def test_manifest_rejects_foreign_key_to_non_key() -> None:
    parent = Table(name="p", columns=(Column(name="id"), Column(name="x")), primary_key=("id",))
    child = Table(name="c", columns=(Column(name="pid"),))
    with pytest.raises(ValidationError, match="is not a key"):
        Manifest(
            dataset="d",
            tier="xs",
            structure_seed=0,
            description="",
            tables=(parent, child),
            foreign_keys=(
                ForeignKey(child_table="c", child_columns=("pid",), parent_table="p", parent_columns=("x",)),
            ),
        )


def test_manifest_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Column.model_validate({"name": "c", "nul_rate": 0.1})


@pytest.mark.parametrize(
    ("family", "params"),
    [
        (Family.NORMAL, {"mean": 0.0, "std": 0.0}),
        (Family.NORMAL, {"mean": float("nan"), "std": 1.0}),
        (Family.LOGNORMAL, {"mu": 0.0, "sigma": -1.0}),
        (Family.EXPONENTIAL, {"scale": 0}),
        (Family.PARETO, {"alpha": 1.0, "scale": -1.0}),
        (Family.POISSON, {"lam": 0.0}),
        (Family.POISSON, {"lam": 1.0, "shift": 0.5}),
        (Family.BINOMIAL, {"n": 0, "p": 0.5}),
        (Family.BINOMIAL, {"n": 3, "p": 1.0}),
        (Family.NEGATIVE_BINOMIAL, {"n": 2, "p": 0.0}),
        (Family.ZIPF, {"s": 0.0, "n_values": 10}),
        (Family.ZIPF, {"s": 1.0, "n_values": 0}),
        (Family.UNIFORM, {"low": 1.0, "high": 1.0}),
        (Family.UNIFORM, {"low": "2024-02-01", "high": "2024-01-01"}),
        (Family.DISCRETE_UNIFORM, {"low": 3, "high": 2}),
        (Family.DISCRETE_UNIFORM, {"low": 0.5, "high": 2}),
        (Family.CATEGORICAL, {"values": ["a", "a"], "probabilities": [0.5, 0.5]}),
        (Family.CATEGORICAL, {"values": ["a", "b"], "probabilities": [1.5, -0.5]}),
        (Family.MIXTURE, {"components": []}),
        (Family.MIXTURE, {"components": [{"weight": 1.0, "mean": 0.0, "std": 0.0}]}),
        (Family.MIXTURE, {"components": [{"weight": 1.0, "mean": 0.0}]}),
        (Family.SEQUENCE, {"start": 1, "step": 0}),
        (
            Family.SEASONAL,
            {"start": "2024-01-02", "end": "2024-01-01", "hour_weights": [1] * 24, "weekday_weights": [1] * 7},
        ),
        (
            Family.SEASONAL,
            {"start": "2024-01-01", "end": "2024-01-02", "hour_weights": [0] * 24, "weekday_weights": [1] * 7},
        ),
    ],
)
def test_distribution_rejects_out_of_domain_params(family: Family, params: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Distribution(family=family, params=params)


@pytest.mark.parametrize(("low", "high"), [(5, 1), ("2024-02-01", "2024-01-01"), ("1.5", "0.5")])
def test_range_check_rejects_unordered_bounds(low: object, high: object) -> None:
    with pytest.raises(ValidationError, match="not ordered"):
        RangeCheck(column="c", min=low, max=high)


def test_privacy_classes_replace_the_sensitive_flag() -> None:
    assert Column(name="c", privacy="secret").sensitive
    assert not Column(name="c").sensitive
    with pytest.raises(ValidationError):
        Column(name="c", privacy="pii")
    with pytest.raises(ValidationError):
        Column.model_validate({"name": "c", "sensitive": True})


def test_strings_dataset_plants_privacy_classes_and_a_quasi_identifier_set() -> None:
    _, truth = generate(DATASETS["strings"], "xs", 0)
    contacts = truth.table("contacts")
    privacy = {c.name: c.privacy for c in contacts.columns if c.privacy}
    assert privacy["phone"] == privacy["uuid"] == privacy["email"] == "direct_identifier"
    assert privacy["session_token"] == "secret"
    assert privacy["comment"] == "free_text"
    (qi,) = contacts.quasi_identifiers
    assert qi.columns == ("home_zip", "birth_date", "gender")
    assert all(privacy[c] == "quasi_identifier" for c in qi.columns)
    assert 0.9 < qi.unique_fraction < 1


def test_table_rejects_quasi_identifiers_on_unknown_columns() -> None:
    with pytest.raises(ValidationError, match="unknown columns"):
        Table(
            name="t",
            columns=(Column(name="a"),),
            quasi_identifiers=(QuasiIdentifierSet(columns=("a", "b"), unique_fraction=0.5, mechanism="m"),),
        )


def test_manifest_labels_declared_values_as_generating_parameters() -> None:
    _, truth = generate(DATASETS["numeric"], "xs", 0)
    assert truth.declared_values == "generating_parameters"
    assert truth.model_dump()["declared_values"] == "generating_parameters"


@pytest.mark.parametrize(
    ("name", "table"), [("retail", "customers"), ("retail", "order_items"), ("strings", "contacts")]
)
def test_realistic_tables_are_shuffled_with_hashed_keys(name: str, table: str) -> None:
    tables, truth = generate(DATASETS[name], "xs", 0)
    assert truth.table(table).row_order == "shuffled"
    key = truth.table(table).primary_key[0]
    values = tables[table].column(key).to_numpy()
    assert not (values[1:] >= values[:-1]).all()


def test_manifest_accepts_the_earlier_seed_field() -> None:
    _, truth = generate(DATASETS["numeric"], "xs", 4)
    data = truth.model_dump()
    data["seed"] = data.pop("structure_seed")
    restored = Manifest.model_validate(data)
    assert restored == truth
    assert restored.seed == restored.structure_seed == 4
