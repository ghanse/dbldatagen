"""Reconstruction fidelity: how closely synthetic tables match the real tables they imitate.

Marginals, null fractions, and pairwise dependence are computed on bounded seeded samples collected to the driver.
Row counts, null fractions of top-level columns, distinct counts, constraints, keys, and foreign keys are Spark
aggregates over the full tables.
"""

from __future__ import annotations

import itertools
import re
import zlib
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from pyspark.errors import AnalysisException
from pyspark.sql import Column as SparkColumn
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from profiler_bench.catalog import load, materialize
from profiler_bench.catalog.manifest import (
    Check,
    ComparisonCheck,
    EnumerationCheck,
    ForeignKey,
    Manifest,
    RangeCheck,
    Table,
    _check_columns,
)
from profiler_bench.fidelity import stats
from profiler_bench.fidelity.config import MetricsConfig, load_config
from profiler_bench.fidelity.frames import ROW_FRAME, Leaf, collect, identifier, leaf_kind, quoted, require_utc
from profiler_bench.fidelity.scores import (
    CheckScore,
    ColumnScore,
    DiscriminatorScore,
    ForeignKeyScore,
    KeyScore,
    PairScore,
    ReconstructionScore,
    TableReconstruction,
    clamp,
    mean,
    weighted,
)


MARGINAL_METRICS = frozenset(
    {"ks", "wasserstein", "tvd", "format", "length_ks", "length_wasserstein", "special_values", "missing"}
)
MIN_DISCRIMINATOR_ROWS = 20


def score_reconstruction(
    real: Mapping[str, DataFrame],
    synth: Mapping[str, DataFrame],
    manifest: Manifest,
    *,
    seed: int,
    config: MetricsConfig | None = None,
) -> ReconstructionScore:
    """Scores synthetic tables against the real tables of a catalog dataset.

    Args:
        real: Real DataFrames by table name.
        synth: Synthetic DataFrames by table name; a missing table scores 0.
        manifest: Ground-truth manifest of the real dataset.
        seed: Seed of every sample and random pair draw.
        config: Metrics configuration; defaults to the packaged one.

    Raises:
        ValueError: If the Spark session time zone is not UTC.
    """
    config = config or load_config()
    require_utc(list(real.values()))
    tables = [
        _score_table(table, real, synth, manifest.foreign_keys, seed, config)
        for table in manifest.tables
        if table.name in real
    ]
    return ReconstructionScore(
        dataset=manifest.dataset,
        tier=manifest.tier,
        seed=seed,
        config_version=config.version,
        config_hash=config.hash(),
        tables=tables,
        score=mean([t.score for t in tables]) or 0.0,
    )


def ceiling(
    name: str,
    tier: str,
    *,
    spark: SparkSession,
    seeds: tuple[int, int] = (0, 1),
    seed: int = 0,
    config: MetricsConfig | None = None,
) -> ReconstructionScore:
    """Scores one catalog seed of a dataset against another: the best reconstruction score achievable.

    Args:
        name: Dataset name.
        tier: Scale tier.
        spark: Spark session.
        seeds: Catalog seeds of the real and the stand-in synthetic data.
        seed: Seed of the metric's sampling.
        config: Metrics configuration; defaults to the packaged one.
    """
    truth = materialize(name, tier, seeds[0]).manifest
    return score_reconstruction(
        load(name, tier, seeds[0], spark), load(name, tier, seeds[1], spark), truth, seed=seed, config=config
    )


def _score_table(
    truth: Table,
    real: Mapping[str, DataFrame],
    synth: Mapping[str, DataFrame],
    foreign_keys: tuple[ForeignKey, ...],
    seed: int,
    config: MetricsConfig,
) -> TableReconstruction:
    real_df = real[truth.name]
    real_rows, real_columns = _column_stats(real_df)
    if truth.name not in synth:
        groups = dict.fromkeys(config.reconstruction_weights, 0.0)
        return TableReconstruction(
            name=truth.name, real_rows=real_rows, synth_rows=0, schema_match=0.0, groups=groups, score=0.0
        )
    synth_df = synth[truth.name]
    real_types = {f.name: f.dataType.simpleString() for f in real_df.schema.fields}
    synth_types = {f.name: f.dataType.simpleString() for f in synth_df.schema.fields}
    good = [name for name, dtype in real_types.items() if synth_types.get(name) == dtype]
    names = set(real_types) | set(synth_types)
    schema_match = len(good) / len(names) if names else 1.0
    synth_df = synth_df.select(*(quoted(n) for n in good))
    synth_rows, synth_columns = _column_stats(synth_df)
    cap = config.sampling.max_rows
    real_leaves, real_frames = collect(real_df, truth, real_rows, cap, seed)
    _, synth_frames = collect(synth_df, truth, synth_rows, cap, seed)
    kinds = {
        leaf.path: leaf_kind(leaf, real_frames[leaf.frame][leaf.path], config.marginal.categorical_max_distinct)
        for leaf in real_leaves
    }
    columns = [
        _score_leaf(
            leaf,
            kinds[leaf.path],
            real_frames,
            synth_frames,
            real_columns,
            synth_columns,
            real_rows,
            synth_rows,
            config.marginal.top_values,
        )
        for leaf in real_leaves
        if leaf.root in good
    ]
    columns += [
        ColumnScore(path=n, kind="missing", metrics={"missing": 0.0, "null_fraction": 0.0, "distinct": 0.0})
        for n in real_types
        if n not in good
    ]
    row_frame_kinds = {leaf.path: kinds[leaf.path] for leaf in real_leaves if leaf.frame == ROW_FRAME}
    pairs = _pairs(truth, row_frame_kinds, real_frames[ROW_FRAME], synth_frames[ROW_FRAME], seed, config)
    checks = _checks(truth, synth_df, set(good))
    keys = _keys(truth, synth_df, set(good))
    fks = [_foreign_key(fk, real, synth) for fk in foreign_keys if fk.child_table == truth.name]
    usable = {p: k for p, k in row_frame_kinds.items() if p in synth_frames[ROW_FRAME].columns}
    discriminator = _discriminator(real_frames[ROW_FRAME], synth_frames[ROW_FRAME], usable, seed, config)
    marginal = [mean([v for k, v in c.metrics.items() if k in MARGINAL_METRICS]) for c in columns]
    groups, score = weighted(
        {
            "schema": (schema_match + _ratio(real_rows, synth_rows)) / 2,
            "marginal": mean([m for m in marginal if m is not None]),
            "null_fraction": mean([c.metrics["null_fraction"] for c in columns if "null_fraction" in c.metrics]),
            "distinct": mean([c.metrics["distinct"] for c in columns if "distinct" in c.metrics]),
            "dependence_declared": mean([p.score for p in pairs if p.declared]),
            "dependence_random": mean([p.score for p in pairs if not p.declared]),
            "constraints": mean([c.score for c in checks]),
            "keys": mean([k.score for k in keys]),
            "foreign_keys": mean([f.score for f in fks]),
            "discriminator": discriminator.score if discriminator else None,
        },
        config.reconstruction_weights,
    )
    return TableReconstruction(
        name=truth.name,
        real_rows=real_rows,
        synth_rows=synth_rows,
        schema_match=schema_match,
        columns=columns,
        pairs=pairs,
        checks=checks,
        keys=keys,
        foreign_keys=fks,
        discriminator=discriminator,
        groups=groups,
        score=score,
    )


def _ratio(a: float, b: float) -> float:
    return (min(a, b) + 1) / (max(a, b) + 1)


def _column_stats(df: DataFrame) -> tuple[int, dict[str, tuple[int, int | None]]]:
    rows = df.count()
    fields = df.schema.fields
    if not fields:
        return rows, {}
    values = ", ".join(f"named_struct('i', {i}, 'v', {_text(f)})" for i, f in enumerate(fields))
    found = {
        r["i"]: (r["present"], r["distinct"])
        for r in df.selectExpr(f"inline(array({values}))")
        .groupBy("i")
        .agg(F.count("v").alias("present"), F.approx_count_distinct("v", 0.01).alias("distinct"))
        .collect()
    }
    return rows, {
        f.name: (found.get(i, (0, 0))[0], found.get(i, (0, 0))[1] if _atomic(f) else None) for i, f in enumerate(fields)
    }


def _text(field: T.StructField) -> str:
    name = identifier(field.name)
    return f"CAST({name} AS STRING)" if _atomic(field) else f"to_json({name})"


def _atomic(field: T.StructField) -> bool:
    return isinstance(field.dataType, T.AtomicType)


def _score_leaf(
    leaf: Leaf,
    kind: str,
    real_frames: Mapping[str, pd.DataFrame],
    synth_frames: Mapping[str, pd.DataFrame],
    real_columns: Mapping[str, tuple[int, int | None]],
    synth_columns: Mapping[str, tuple[int, int | None]],
    real_rows: int,
    synth_rows: int,
    top: int,
) -> ColumnScore:
    real = real_frames[leaf.frame][leaf.path]
    synth_frame = synth_frames.get(leaf.frame)
    synth = synth_frame[leaf.path] if synth_frame is not None else pd.Series([], dtype=object)
    metrics: dict[str, float] = {}
    raw: dict[str, float] = {}
    if kind in ("numeric", "length"):
        _numeric(real, synth, metrics, raw, "")
    else:
        _categorical(real.dropna(), synth.dropna(), metrics, raw, kind, leaf, top)
    if leaf.dtype == "number":
        _specials(real, synth, metrics, raw)
    if leaf.top_level:
        real_null = 1 - real_columns[leaf.root][0] / real_rows if real_rows else 0.0
        synth_null = 1 - synth_columns[leaf.root][0] / synth_rows if synth_rows else 0.0
        _null(real_null, synth_null, metrics, raw)
        real_distinct, synth_distinct = real_columns[leaf.root][1], synth_columns[leaf.root][1]
        if real_distinct is not None and synth_distinct is not None:
            metrics["distinct"] = _ratio(real_distinct, synth_distinct)
            raw.update(real_distinct=real_distinct, synth_distinct=synth_distinct)
    elif kind != "length":
        _null(_null_share(real), _null_share(synth), metrics, raw)
    return ColumnScore(path=leaf.path, kind=kind, metrics=metrics, raw=raw)


def _null_share(values: pd.Series) -> float:
    return float(values.isna().mean()) if len(values) else 0.0


def _null(real: float, synth: float, metrics: dict[str, float], raw: dict[str, float]) -> None:
    metrics["null_fraction"] = clamp(1 - abs(real - synth))
    raw.update(real_null_fraction=real, synth_null_fraction=synth)


def _numeric(real: pd.Series, synth: pd.Series, metrics: dict[str, float], raw: dict[str, float], prefix: str) -> None:
    a, b = stats.finite(real), stats.finite(synth)
    if len(a) == 0:
        return
    ks, w1 = stats.ks_wasserstein(a, b) if len(b) else (1.0, 1.0)
    metrics[f"{prefix}ks"], metrics[f"{prefix}wasserstein"] = 1 - ks, 1 - w1
    raw[f"{prefix}ks"], raw[f"{prefix}wasserstein"] = ks, w1


def _categorical(
    real: pd.Series, synth: pd.Series, metrics: dict[str, float], raw: dict[str, float], kind: str, leaf: Leaf, top: int
) -> None:
    if len(real) == 0:
        return
    if len(synth) == 0:
        metrics["tvd"], raw["tvd"] = 0.0, 1.0
        return
    distance = stats.tvd(real, synth, stats.top_values(real, top))
    metrics["tvd"], raw["tvd"] = 1 - distance, distance
    if leaf.dtype != "string":
        return
    if kind == "string":
        _numeric(real.map(len), synth.map(len), metrics, raw, "length_")
    formats = leaf.truth.formats if leaf.truth is not None else ()
    if formats:
        patterns = [re.compile(f.regex) for f in formats]
        distance = 0.5 * float(np.abs(_format_shares(real, patterns) - _format_shares(synth, patterns)).sum())
        metrics["format"], raw["format"] = 1 - distance, distance


def _format_shares(values: pd.Series, patterns: list[re.Pattern[str]]) -> np.ndarray:
    counts = np.zeros(len(patterns) + 1)
    for value in values.astype(str):
        index = next((i for i, p in enumerate(patterns) if p.fullmatch(value)), len(patterns))
        counts[index] += 1
    return counts / max(len(values), 1)


def _specials(real: pd.Series, synth: pd.Series, metrics: dict[str, float], raw: dict[str, float]) -> None:
    a, b = _special_shares(real), _special_shares(synth)
    if a[1:].sum() == 0 and b[1:].sum() == 0:
        return
    distance = 0.5 * float(np.abs(a - b).sum())
    metrics["special_values"], raw["special_values"] = 1 - distance, distance


def _special_shares(values: pd.Series) -> np.ndarray:
    numbers = pd.to_numeric(values.dropna(), errors="coerce").to_numpy(dtype=np.float64)
    if len(numbers) == 0:
        return np.zeros(4)
    nan, pos, neg = np.isnan(numbers).sum(), (numbers == np.inf).sum(), (numbers == -np.inf).sum()
    return np.array([len(numbers) - nan - pos - neg, nan, pos, neg], dtype=np.float64) / len(numbers)


def _pairs(
    truth: Table, kinds: Mapping[str, str], real: pd.DataFrame, synth: pd.DataFrame, seed: int, config: MetricsConfig
) -> list[PairScore]:
    declared = [(dep.columns, dep.measure) for dep in truth.dependencies if set(dep.columns) <= set(kinds)]
    taken = {frozenset(columns) for columns, _ in declared}
    candidates = [
        pair
        for pair in itertools.combinations(sorted(p for p, k in kinds.items() if k in ("numeric", "categorical")), 2)
        if frozenset(pair) not in taken and "." not in pair[0] + pair[1]
    ]
    rng = np.random.default_rng([seed, zlib.crc32(truth.name.encode())])
    count = min(config.dependence.random_pairs, len(candidates))
    drawn = [candidates[i] for i in sorted(rng.choice(len(candidates), size=count, replace=False))] if count else []
    random_pairs = [(pair, _measure_for(kinds[pair[0]], kinds[pair[1]])) for pair in drawn]
    random_pairs = [
        ((b, a), m) if m == "correlation_ratio" and kinds[a] == "numeric" else ((a, b), m) for (a, b), m in random_pairs
    ]
    result = []
    for (columns, measure), is_declared in [*((d, True) for d in declared), *((r, False) for r in random_pairs)]:
        real_value = _measure(real, columns, measure, kinds, config)
        present = set(columns) <= set(synth.columns)
        synth_value = _measure(synth, columns, measure, kinds, config) if present else None
        score = clamp(1 - abs(real_value - synth_value)) if synth_value is not None else 0.0
        result.append(
            PairScore(
                columns=columns, measure=measure, declared=is_declared, real=real_value, synth=synth_value, score=score
            )
        )
    return result


def _measure_for(a: str, b: str) -> str:
    if a == b == "numeric":
        return "spearman"
    if a == b == "categorical":
        return "cramers_v"
    return "correlation_ratio"


def _measure(
    frame: pd.DataFrame, columns: tuple[str, str], measure: str, kinds: Mapping[str, str], config: MetricsConfig
) -> float:
    x_name, y_name = columns
    pdf = frame[[x_name, y_name]].dropna()
    numeric_x = measure in ("pearson", "spearman") or (measure == "correlation_ratio" and kinds[x_name] == "numeric")
    numeric_y = measure != "cramers_v"
    x = _numbers(pdf[x_name]) if numeric_x else pdf[x_name].astype(str)
    y = _numbers(pdf[y_name]) if numeric_y else pdf[y_name].astype(str)
    keep = x.notna() & y.notna()
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return 0.0
    if measure == "pearson":
        return stats.pearson(x, y)
    if measure == "spearman":
        return stats.spearman(x, y)
    if measure == "cramers_v":
        return stats.cramers_v(x, y)
    return stats.correlation_ratio(x, y, config.dependence.correlation_bins, numeric_x)


def _numbers(values: pd.Series) -> pd.Series:
    numbers = pd.to_numeric(values, errors="coerce").astype(np.float64)
    return numbers.where(np.isfinite(numbers))


def _violated(check: Check, df: DataFrame) -> SparkColumn:
    if isinstance(check, RangeCheck):
        column, dtype = quoted(check.column), df.schema[check.column].dataType
        bad = F.lit(False)
        if check.min is not None:
            bad = bad | (column < F.lit(check.min).cast(dtype))
        if check.max is not None:
            bad = bad | (column > F.lit(check.max).cast(dtype))
        condition = bad
    elif isinstance(check, EnumerationCheck):
        condition = ~quoted(check.column).isin(list(check.values))
    elif isinstance(check, ComparisonCheck):
        condition = ~F.expr(f"`{check.left}` {check.op} `{check.right}`")
    else:
        condition = ~F.expr(check.sql)
    return F.coalesce(condition, F.lit(False))


def _check_label(check: Check) -> str:
    if isinstance(check, RangeCheck):
        return f"range {check.column} in [{check.min}, {check.max}]"
    if isinstance(check, EnumerationCheck):
        return f"enumeration {check.column}"
    if isinstance(check, ComparisonCheck):
        return f"{check.left} {check.op} {check.right}"
    return check.sql


def _checks(truth: Table, df: DataFrame, present: set[str]) -> list[CheckScore]:
    conditions: dict[str, SparkColumn | None] = {}
    for check in truth.checks:
        condition = None
        if set(_check_columns(check)) <= present:
            try:
                condition = _violated(check, df)
                df.select(condition)
            except AnalysisException:
                condition = None
        conditions[_check_label(check)] = condition
    valid = {label: c for label, c in conditions.items() if c is not None}
    counts: dict[str, Any] = {}
    if valid:
        aggregates = [F.count(F.lit(1)).alias("__rows")]
        aggregates += [F.sum(c.cast("long")).alias(f"v{i}") for i, c in enumerate(valid.values())]
        row = df.agg(*aggregates).collect()[0]
        counts = {label: (row[f"v{i}"] or 0, row["__rows"]) for i, label in enumerate(valid)}
    result = [
        CheckScore(
            label=label,
            kind="check",
            score=(1 - counts[label][0] / counts[label][1] if counts[label][1] else 1.0) if label in counts else 0.0,
        )
        for label in conditions
    ]
    for fd in truth.functional_dependencies:
        label = f"{', '.join(fd.determinant)} -> {', '.join(fd.dependent)}"
        result.append(
            CheckScore(
                label=label, kind="functional_dependency", score=_fd_score(fd.determinant, fd.dependent, df, present)
            )
        )
    return result


def _fd_score(determinant: tuple[str, ...], dependent: tuple[str, ...], df: DataFrame, present: set[str]) -> float:
    columns = [*determinant, *dependent]
    if not set(columns) <= present:
        return 0.0
    kept, total = (
        df.dropna(subset=columns)
        .groupBy(*(quoted(c) for c in columns))
        .count()
        .groupBy(*(quoted(c) for c in determinant))
        .agg(F.max("count").alias("kept"), F.sum("count").alias("total"))
        .agg(F.sum("kept"), F.sum("total"))
        .collect()[0]
    )
    return float(kept) / float(total) if total else 1.0


def _keys(truth: Table, df: DataFrame, present: set[str]) -> list[KeyScore]:
    declared: list[tuple[Literal["primary", "unique"], tuple[str, ...]]] = (
        [("primary", truth.primary_key)] if truth.primary_key else []
    )
    declared += [("unique", key) for key in truth.unique_keys]
    usable = [(kind, key) for kind, key in declared if set(key) <= present]
    values: dict[str, Any] = {}
    if usable:
        aggregates = [F.count(F.lit(1)).alias("__rows")]
        for i, (_, key) in enumerate(usable):
            complete = F.lit(True)
            for column in key:
                complete = complete & quoted(column).isNotNull()
            aggregates.append(F.count_distinct(*(quoted(c) for c in key)).alias(f"d{i}"))
            aggregates.append(F.sum(complete.cast("long")).alias(f"p{i}"))
        values = df.agg(*aggregates).collect()[0].asDict()
    result = []
    for kind, key in declared:
        if (kind, key) not in usable:
            result.append(KeyScore(columns=key, kind=kind, score=0.0))
            continue
        i = usable.index((kind, key))
        base = values["__rows"] if kind == "primary" else values[f"p{i}"] or 0
        result.append(KeyScore(columns=key, kind=kind, score=values[f"d{i}"] / base if base else 1.0))
    return result


def _foreign_key(fk: ForeignKey, real: Mapping[str, DataFrame], synth: Mapping[str, DataFrame]) -> ForeignKeyScore:
    label = f"{fk.child_table}{list(fk.child_columns)} -> {fk.parent_table}{list(fk.parent_columns)}"
    real_coverage, real_fanout = _fanout(fk, real)
    try:
        synth_coverage, synth_fanout = _fanout(fk, synth)
    except (KeyError, AnalysisException):
        return ForeignKeyScore(
            label=label,
            real_coverage=real_coverage,
            synth_coverage=0.0,
            fanout_ks=1.0,
            fanout_wasserstein=1.0,
            score=0.0,
        )
    if real_fanout.empty or synth_fanout.empty:
        ks, w1 = (0.0, 0.0) if real_fanout.empty and synth_fanout.empty else (1.0, 1.0)
    else:
        ks, w1 = stats.ks_wasserstein(
            real_fanout["fanout"].to_numpy(dtype=np.float64),
            synth_fanout["fanout"].to_numpy(dtype=np.float64),
            real_fanout["parents"].to_numpy(dtype=np.float64),
            synth_fanout["parents"].to_numpy(dtype=np.float64),
        )
    integrity = clamp(1 - abs(real_coverage - synth_coverage))
    return ForeignKeyScore(
        label=label,
        real_coverage=real_coverage,
        synth_coverage=synth_coverage,
        fanout_ks=ks,
        fanout_wasserstein=w1,
        score=(integrity + (1 - ks) + (1 - w1)) / 3,
    )


def _fanout(fk: ForeignKey, frames: Mapping[str, DataFrame]) -> tuple[float, pd.DataFrame]:
    keys = list(fk.child_columns)
    child = frames[fk.child_table].select(*(quoted(c) for c in keys)).dropna()
    parent = frames[fk.parent_table].select(
        *(quoted(p).alias(c) for p, c in zip(fk.parent_columns, fk.child_columns, strict=True))
    )
    total = child.count()
    missing = child.join(parent, keys, "left_anti").count()
    fanout: pd.DataFrame = (
        parent.join(child.groupBy(*keys).agg(F.count(F.lit(1)).alias("fanout")), keys, "left")
        .fillna(0, subset=["fanout"])
        .groupBy("fanout")
        .agg(F.count(F.lit(1)).alias("parents"))
        .orderBy("fanout")
        .toPandas()
    )
    return (1.0 - missing / total if total else 1.0), fanout


def _discriminator(
    real: pd.DataFrame, synth: pd.DataFrame, kinds: Mapping[str, str], seed: int, config: MetricsConfig
) -> DiscriminatorScore | None:
    rows = min(config.sampling.discriminator_rows, len(real), len(synth))
    if rows < MIN_DISCRIMINATOR_ROWS or not kinds:
        return None
    encoder_kinds = {p: "numeric" if k in ("numeric", "length") else k for p, k in kinds.items()}
    real, synth = real.iloc[:rows], synth.iloc[:rows]
    encoder = stats.Encoder(real, encoder_kinds, config.marginal.top_values)
    x_real, x_synth = encoder.features(real), encoder.features(synth)
    order = np.random.default_rng(seed).permutation(rows)
    train, test = order[: rows // 2], order[rows // 2 :]
    value = stats.knn_auc(x_real[train], x_synth[train], x_real[test], x_synth[test], config.discriminator.k)
    return DiscriminatorScore(auc=value, rows=rows, score=clamp(1 - abs(value - 0.5) * 2))
