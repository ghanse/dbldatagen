"""Discriminator feature hygiene, entity split, and the foreign-key join discriminator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from profiler_bench.catalog.manifest import Column, ForeignKey, Manifest, Table
from profiler_bench.fidelity import reconstruction, score_reconstruction
from profiler_bench.fidelity.scores import DiscriminatorScore, ReconstructionScore


PRODUCTS = Table(
    name="products",
    primary_key=("product_id",),
    columns=(
        Column(name="product_id", type="bigint", unique=True),
        Column(name="category", type="string"),
        Column(name="unit_price", type="double"),
    ),
)
ORDERS = Table(
    name="orders",
    primary_key=("order_id",),
    columns=(
        Column(name="order_id", type="bigint", unique=True),
        Column(name="ticket", type="bigint", note="Undeclared identifier."),
        Column(name="order_day", type="bigint"),
    ),
)
ITEMS = Table(
    name="order_items",
    primary_key=("order_id", "line_no"),
    columns=(
        Column(name="order_id", type="bigint"),
        Column(name="line_no", type="bigint"),
        Column(name="product_id", type="bigint"),
        Column(name="quantity", type="bigint"),
        Column(name="unit_price", type="double"),
    ),
)
RETAIL = Manifest(
    dataset="retail_like",
    tier="xs",
    structure_seed=0,
    description="",
    tables=(PRODUCTS, ORDERS, ITEMS),
    foreign_keys=(
        ForeignKey(
            child_table="order_items", child_columns=("order_id",), parent_table="orders", parent_columns=("order_id",)
        ),
        ForeignKey(
            child_table="order_items",
            child_columns=("product_id",),
            parent_table="products",
            parent_columns=("product_id",),
        ),
    ),
)


def _retail(spark: SparkSession, seed: int) -> dict[str, DataFrame]:
    """Returns one sample of a retail-like dataset: each sample draws its own product prices and ids."""
    rng = np.random.default_rng(seed)
    n_products, n_orders = 60, 1500
    products = pd.DataFrame(
        {
            "product_id": np.arange(n_products) + 1000 * seed,
            "category": rng.choice(["a", "b", "c"], n_products),
            "unit_price": rng.lognormal(3.0, 0.6, n_products),
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": np.arange(n_orders) + 100_000 * seed,
            "ticket": rng.permutation(10 * n_orders)[:n_orders],
            "order_day": rng.integers(0, 365, n_orders),
        }
    )
    lines = rng.integers(1, 5, n_orders)
    order_index = np.repeat(np.arange(n_orders), lines)
    product_index = rng.integers(0, n_products, len(order_index))
    items = pd.DataFrame(
        {
            "order_id": orders["order_id"].to_numpy()[order_index],
            "line_no": np.concatenate([np.arange(1, k + 1) for k in lines]),
            "product_id": products["product_id"].to_numpy()[product_index],
            "quantity": rng.integers(1, 6, len(order_index)),
            "unit_price": products["unit_price"].to_numpy()[product_index],
        }
    )
    return {
        name: spark.createDataFrame(frame)
        for name, frame in (("products", products), ("orders", orders), ("order_items", items))
    }


def _units(score: ReconstructionScore) -> dict[str, DiscriminatorScore]:
    found = {unit: d for unit, d in score.discriminators().items() if d is not None}
    assert len(found) == len(score.discriminators())
    return found


@pytest.fixture(scope="module")
def retail_ceiling(spark: SparkSession) -> ReconstructionScore:
    return score_reconstruction(_retail(spark, 0), _retail(spark, 1), RETAIL, seed=0)


def test_key_and_identifier_columns_are_not_features(retail_ceiling: ReconstructionScore) -> None:
    tables = {t.name: t.discriminator for t in retail_ceiling.tables}
    assert tables["orders"] is not None and tables["orders"].excluded == ["order_id", "ticket"]
    assert tables["order_items"] is not None
    assert tables["order_items"].excluded == ["line_no", "order_id", "product_id"]
    assert tables["order_items"].split_by == ["product_id", "order_id"]
    assert tables["products"] is not None and tables["products"].excluded == ["product_id"]


def test_retail_like_real_vs_real_is_near_chance_after_the_fix(retail_ceiling: ReconstructionScore) -> None:
    for unit, score in _units(retail_ceiling).items():
        assert abs(score.auc - 0.5) < 0.1, unit
    gap = retail_ceiling.discriminator_gap(retail_ceiling)
    assert gap.gap == 0.0


def test_retail_like_real_vs_real_is_separable_without_the_fix(
    spark: SparkSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reconstruction, "declared_identifiers", lambda truth, foreign_keys: set())
    monkeypatch.setattr(reconstruction, "_near_unique_integers", lambda schema, frame, share: set())
    monkeypatch.setattr(reconstruction, "_split_columns", lambda truth, foreign_keys, frame: [])
    score = score_reconstruction(_retail(spark, 0), _retail(spark, 1), RETAIL, seed=0)
    items = next(t.discriminator for t in score.tables if t.name == "order_items")
    assert items is not None and items.auc > 0.9


def _shuffled_fk(spark: SparkSession, shuffle: bool) -> tuple[dict[str, DataFrame], Manifest]:
    rng = np.random.default_rng(0)
    customers = pd.DataFrame({"customer_id": np.arange(400), "tier": rng.choice(["gold", "basic"], 400)})
    owner = rng.integers(0, 400, 4000)
    gold = customers["tier"].to_numpy()[owner] == "gold"
    orders = pd.DataFrame(
        {
            "order_id": np.arange(4000),
            "customer_id": rng.permutation(owner) if shuffle else owner,
            "amount": np.where(gold, 100.0, 10.0) + rng.normal(0, 1, 4000),
        }
    )
    manifest = Manifest(
        dataset="shop",
        tier="xs",
        structure_seed=0,
        description="",
        tables=(
            Table(
                name="customers",
                primary_key=("customer_id",),
                columns=(Column(name="customer_id", type="bigint"), Column(name="tier", type="string")),
            ),
            Table(
                name="orders",
                primary_key=("order_id",),
                columns=(
                    Column(name="order_id", type="bigint"),
                    Column(name="customer_id", type="bigint"),
                    Column(name="amount", type="double"),
                ),
            ),
        ),
        foreign_keys=(
            ForeignKey(
                child_table="orders",
                child_columns=("customer_id",),
                parent_table="customers",
                parent_columns=("customer_id",),
            ),
        ),
    )
    return {"customers": spark.createDataFrame(customers), "orders": spark.createDataFrame(orders)}, manifest


def test_join_discriminator_detects_a_shuffled_foreign_key(spark: SparkSession) -> None:
    """Half the shuffled orders keep a same-tier owner, so the best possible AUC is about 0.75."""
    real, manifest = _shuffled_fk(spark, shuffle=False)
    synth, _ = _shuffled_fk(spark, shuffle=True)
    score = score_reconstruction(real, synth, manifest, seed=0)
    units = _units(score)
    assert abs(units["orders"].auc - 0.5) < 0.1
    assert units["orders['customer_id'] -> customers['customer_id']"].auc > 0.65
    ceiling = score_reconstruction(real, real, manifest, seed=0)
    gap = score.discriminator_gap(ceiling)
    assert gap.units["orders"] < 0.05
    assert gap.units["orders['customer_id'] -> customers['customer_id']"] > 0.2
    assert gap.gap == pytest.approx(sum(gap.units.values()) / 3)


def test_a_constant_synthetic_key_falls_back_to_a_row_split(spark: SparkSession) -> None:
    real, manifest = _shuffled_fk(spark, shuffle=False)
    broken = {
        "customers": real["customers"].withColumn("customer_id", F.lit(0).cast("bigint")),
        "orders": real["orders"].withColumn("customer_id", F.lit(0).cast("bigint")),
    }
    units = _units(score_reconstruction(real, broken, manifest, seed=0))
    assert units["customers"].split_fallback == ["synth"]
    assert units["orders"].split_fallback == ["synth"]
    assert units["orders['customer_id'] -> customers['customer_id']"].rows <= 4000


def test_a_missing_synth_table_counts_as_fully_separable(spark: SparkSession) -> None:
    real, manifest = _shuffled_fk(spark, shuffle=False)
    ceiling = score_reconstruction(real, real, manifest, seed=0)
    partial = score_reconstruction(real, {"customers": real["customers"]}, manifest, seed=0)
    gap = partial.discriminator_gap(ceiling)
    assert gap.units["orders"] == 1.0
    assert gap.units["orders['customer_id'] -> customers['customer_id']"] == 1.0


def test_discriminator_artifacts_hold_no_values_or_probabilities(retail_ceiling: ReconstructionScore) -> None:
    for score in _units(retail_ceiling).values():
        assert set(DiscriminatorScore.model_fields) == {
            "auc",
            "log_loss",
            "rows",
            "n_test",
            "histogram",
            "features",
            "excluded",
            "split_by",
            "split_fallback",
            "score",
        }
        assert all(sum(counts) == score.n_test for counts in score.histogram.values())
        assert all(isinstance(v, int) for counts in score.histogram.values() for v in counts)
