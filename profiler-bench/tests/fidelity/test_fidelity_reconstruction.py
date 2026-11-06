import json
from collections.abc import Callable, Iterator

import pytest
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from profiler_bench.catalog import load, materialize
from profiler_bench.fidelity import ReconstructionScore, ceiling, load_config, overall, score_reconstruction
from profiler_bench.fidelity.scores import TableReconstruction


def _score(
    name: str, spark: SparkSession, edit: Callable[[dict[str, DataFrame]], None] | None = None
) -> ReconstructionScore:
    real = load(name, "xs", 0, spark)
    synth = dict(real)
    if edit is not None:
        edit(synth)
    return score_reconstruction(real, synth, materialize(name, "xs", 0).manifest, seed=0)


def _table(score: ReconstructionScore, name: str) -> TableReconstruction:
    return next(t for t in score.tables if t.name == name)


def _scores(node: object) -> Iterator[float]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("score", "schema_match") or key in ("metrics", "groups"):
                yield from (value.values() if isinstance(value, dict) else [value])
            else:
                yield from _scores(value)
    elif isinstance(node, list):
        for value in node:
            yield from _scores(value)


@pytest.fixture(scope="module")
def numeric_identical(spark: SparkSession) -> ReconstructionScore:
    return _score("numeric", spark)


@pytest.mark.parametrize("name", ["edge", "nested", "retail", "strings", "temporal"])
def test_identical_inputs_score_one(name: str, spark: SparkSession) -> None:
    score = _score(name, spark)
    assert score.score == pytest.approx(1.0, abs=1e-9)
    assert all(t.score == pytest.approx(1.0, abs=1e-9) for t in score.tables)


def test_identical_numeric_scores_one_and_covers_every_group(numeric_identical: ReconstructionScore) -> None:
    table = _table(numeric_identical, "measurements")
    assert numeric_identical.score == pytest.approx(1.0, abs=1e-9)
    assert set(table.groups) >= {"marginal", "dependence_declared", "dependence_random", "constraints", "keys"}
    assert table.discriminator is not None and table.discriminator.auc == 0.5


def test_scores_are_bounded_and_round_trip_through_json(numeric_identical: ReconstructionScore) -> None:
    text = numeric_identical.model_dump_json()
    assert ReconstructionScore.model_validate_json(text) == numeric_identical
    values = list(_scores(json.loads(text)))
    assert values and all(0.0 <= v <= 1.0 for v in values)
    assert numeric_identical.config_hash == load_config().hash()


def test_scores_are_deterministic(spark: SparkSession, numeric_identical: ReconstructionScore) -> None:
    def shift(frames: dict[str, DataFrame]) -> None:
        frames["measurements"] = frames["measurements"].withColumn("normal", frames["measurements"]["normal"] + 1)

    first, second = _score("numeric", spark, shift), _score("numeric", spark, shift)
    assert first.model_dump_json() == second.model_dump_json()


def _column(score: ReconstructionScore, table: str, path: str) -> dict[str, float]:
    return next(c.metrics for c in _table(score, table).columns if c.path == path)


def test_shifted_mean_lowers_the_marginal(spark: SparkSession, numeric_identical: ReconstructionScore) -> None:
    def shift(frames: dict[str, DataFrame]) -> None:
        frames["measurements"] = frames["measurements"].withColumn("normal", frames["measurements"]["normal"] + 5)

    metrics = _column(_score("numeric", spark, shift), "measurements", "normal")
    assert metrics["ks"] < 0.9
    assert metrics["wasserstein"] < 0.9
    assert _column(numeric_identical, "measurements", "normal")["ks"] == 1.0


def test_extra_nulls_lower_the_null_fraction(spark: SparkSession) -> None:
    def blank(frames: dict[str, DataFrame]) -> None:
        frames["measurements"] = frames["measurements"].withColumn(
            "lognormal", F.expr("IF(id % 5 = 0, NULL, lognormal)")
        )

    score = _score("numeric", spark, blank)
    assert _column(score, "measurements", "lognormal")["null_fraction"] == pytest.approx(0.8, abs=0.01)
    assert _table(score, "measurements").groups["null_fraction"] < 1.0


def test_shuffled_column_breaks_the_declared_dependency(spark: SparkSession) -> None:
    def shuffle(frames: dict[str, DataFrame]) -> None:
        df = frames["measurements"]
        ranked = df.withColumn("__r", F.row_number().over(Window.orderBy("id")))
        shuffled = df.select(F.row_number().over(Window.orderBy(F.xxhash64("id"))).alias("__r"), "copula_y")
        frames["measurements"] = ranked.drop("copula_y").join(shuffled, "__r").drop("__r").select(*df.columns)

    score = _score("numeric", spark, shuffle)
    table = _table(score, "measurements")
    pair = next(p for p in table.pairs if p.columns == ("copula_x", "copula_y"))
    assert pair.score < 0.4
    assert table.groups["dependence_declared"] < 0.95
    assert _column(score, "measurements", "copula_y")["ks"] == 1.0


def test_violated_constraint_lowers_satisfaction(spark: SparkSession) -> None:
    def swap(frames: dict[str, DataFrame]) -> None:
        frames["measurements"] = frames["measurements"].withColumn(
            "range_low", F.expr("IF(id % 4 = 0, range_high + 1, range_low)")
        )

    table = _table(_score("numeric", spark, swap), "measurements")
    check = next(c for c in table.checks if c.label == "range_low <= range_high")
    assert check.score == pytest.approx(0.75, abs=0.01)


def test_broken_foreign_key_and_duplicate_key_are_scored(spark: SparkSession) -> None:
    def corrupt(frames: dict[str, DataFrame]) -> None:
        orders = frames["orders"]
        orders = orders.withColumn("customer_id", F.expr("IF(order_id % 5 = 0, customer_id + 1000000, customer_id)"))
        frames["orders"] = orders.withColumn("order_id", F.expr("IF(order_id % 10 = 0, order_id - 1, order_id)"))

    table = _table(_score("retail", spark, corrupt), "orders")
    fk = table.foreign_keys[0]
    assert fk.real_coverage == 1.0
    assert fk.synth_coverage == pytest.approx(0.8, abs=0.01)
    assert table.groups["foreign_keys"] < 0.95
    assert table.keys[0].score == pytest.approx(0.9, abs=0.01)


def test_violated_functional_dependency_is_scored(spark: SparkSession) -> None:
    def corrupt(frames: dict[str, DataFrame]) -> None:
        frames["customers"] = frames["customers"].withColumn(
            "region", F.expr("IF(customer_id % 2 = 0, 'nowhere', region)")
        )

    table = _table(_score("retail", spark, corrupt), "customers")
    fd = next(c for c in table.checks if c.kind == "functional_dependency")
    assert fd.score < 0.8


def test_missing_column_and_table_score_zero(spark: SparkSession) -> None:
    def drop(frames: dict[str, DataFrame]) -> None:
        frames["orders"] = frames["orders"].drop("status")
        del frames["products"]

    score = _score("retail", spark, drop)
    orders = _table(score, "orders")
    assert orders.schema_match == pytest.approx(4 / 5)
    assert next(c for c in orders.columns if c.path == "status").kind == "missing"
    assert _table(score, "products").score == 0.0
    assert next(c for c in orders.checks if c.label == "enumeration status").score == 0.0


def test_nested_perturbation_moves_array_length_and_element_marginals(spark: SparkSession) -> None:
    def corrupt(frames: dict[str, DataFrame]) -> None:
        frames["profiles"] = frames["profiles"].withColumn("scores", F.expr("transform(scores, x -> x + 10)"))
        frames["profiles"] = frames["profiles"].withColumn("tags", F.expr("IF(tags IS NULL, NULL, slice(tags, 1, 1))"))

    score = _score("nested", spark, corrupt)
    assert _column(score, "profiles", "scores[]")["ks"] < 0.5
    assert _column(score, "profiles", "tags#length")["ks"] < 0.9


def test_ceiling_is_high_but_below_identical(spark: SparkSession, numeric_identical: ReconstructionScore) -> None:
    top = ceiling("numeric", "xs", spark=spark)
    assert 0.9 < top.score < 1.0
    summary = overall([numeric_identical], {"numeric": top})
    assert summary.relative["numeric"] == 1.0
    assert summary.relative_score == 1.0
    assert top.relative_to(top) == 1.0
