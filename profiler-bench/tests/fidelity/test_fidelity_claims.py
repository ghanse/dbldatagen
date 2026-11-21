"""Tests for claim verification, including the project-then-sample sampling fix."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from dbldatagen_profiler.model import DependenceMethod, Dependency
from profiler_bench.fidelity.claims import Verifier
from profiler_bench.fidelity.config import load_config
from profiler_bench.fidelity.frames import ROW_FRAME, collect
from profiler_bench.fidelity.reconstruction import pair_measure


def _low_cardinality_table(spark: SparkSession, rows: int = 100_000, seed: int = 0) -> tuple[DataFrame, float]:
    """Table with duplicate low-cardinality pairs plus a unique id.

    Many rows share the same (x, y) values. Hashing only those two columns therefore keeps or drops whole
    duplicate groups together, which biases Pearson and makes the sample size drift. A unique `id` column
    makes full-row hashes distinct so `bounded()` samples rows, not distinct pairs.
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 5, size=rows)
    # Mostly y == x, sometimes a different discrete value — few distinct pairs, many duplicate rows.
    y = np.where(rng.random(rows) < 0.85, x, rng.integers(0, 5, size=rows))
    x_f, y_f = x.astype(np.float64), y.astype(np.float64)
    pdf = pd.DataFrame({"x": x_f, "y": y_f, "id": np.arange(rows, dtype=np.int64)})
    return spark.createDataFrame(pdf), float(np.corrcoef(x_f, y_f)[0, 1])


def test_dependency_pearson_matches_full_table_on_low_cardinality(spark: SparkSession) -> None:
    """Full-row sampling keeps Pearson and n near the full-table truth; project-then-sample does not."""
    df, full_r = _low_cardinality_table(spark)
    config = load_config()
    rows = df.count()
    assert rows > config.sampling.max_rows
    verifier = Verifier({"t": df}, seed=0, config=config)
    sampled = verifier.dependency_sample("t", ("x", "y"))
    assert sampled is not None
    frame, kinds = sampled
    measured = pair_measure(frame, ("x", "y"), "pearson", kinds, config)
    target = config.sampling.max_rows
    assert abs(measured - full_r) <= 0.02
    assert abs(len(frame) - target) / target <= 0.05
    claim = Dependency(columns=("x", "y"), method=DependenceMethod.PEARSON, value=full_r)
    assert verifier.dependency("t", claim)

    # Project-then-sample (the old bug): content-hash over (x, y) only collapses identical pairs.
    old_leaves, old_frames = collect(df.select("x", "y"), None, rows, target, 0)
    old_frame = old_frames[ROW_FRAME]
    old_kinds = {leaf.path: "numeric" for leaf in old_leaves}
    old_r = pair_measure(old_frame, ("x", "y"), "pearson", old_kinds, config)
    assert abs(old_r - full_r) > 0.02 or abs(len(old_frame) - target) / target > 0.05


def test_dependency_sample_is_deterministic(spark: SparkSession) -> None:
    df, _ = _low_cardinality_table(spark, rows=50_000)
    config = load_config()
    first = Verifier({"t": df}, seed=7, config=config).dependency_sample("t", ("x", "y"))
    second = Verifier({"t": df}, seed=7, config=config).dependency_sample("t", ("x", "y"))
    assert first is not None and second is not None
    assert first[0]["__h"].tolist() == second[0]["__h"].tolist()
    assert first[0][["x", "y"]].to_dict(orient="list") == second[0][["x", "y"]].to_dict(orient="list")
