from typing import Any

import pytest
from pydantic import ValidationError

from profiler_bench.catalog import DATASETS, Manifest, generate
from profiler_bench.catalog.manifest import Column, Distribution, Family, ForeignKey, Format, Table


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
            seed=0,
            description="",
            tables=(parent, child),
            foreign_keys=(
                ForeignKey(child_table="c", child_columns=("pid",), parent_table="p", parent_columns=("x",)),
            ),
        )


def test_manifest_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Column.model_validate({"name": "c", "nul_rate": 0.1})
