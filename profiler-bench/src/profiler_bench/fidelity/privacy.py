"""Privacy guard: whether a profile or the synthetic data generated from it gives away real rows or sensitive values.

Data checks compare the synth with a real holdout, a fresh draw from the same catalog dataset under another seed. The
holdout never reached the profiler, so the rate at which it matches the real data is the rate expected by chance.
Every check is pass/fail against a threshold of the metrics configuration; values themselves are never reported.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import ValidationError
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from dbldatagen_profiler.model import ColumnProfile, EnumerationConstraint, LogicalType, Profile
from profiler_bench.catalog import load
from profiler_bench.catalog.manifest import Manifest, Table
from profiler_bench.fidelity import regex, stats
from profiler_bench.fidelity.config import MetricsConfig, load_config
from profiler_bench.fidelity.frames import ROW_FRAME, collect, identifier, leaf_kind, require_utc
from profiler_bench.fidelity.scores import GuardCheck, PrivacyScore
from profiler_bench.fidelity.truth import VALUE_LEAK_CATEGORIES, quasi_identifier_sets, sensitive_columns


def score_privacy(
    real: Mapping[str, DataFrame],
    synth: Mapping[str, DataFrame],
    profile: Profile,
    manifest: Manifest,
    *,
    holdout: Mapping[str, DataFrame] | None = None,
    seed: int = 0,
    config: MetricsConfig | None = None,
) -> PrivacyScore:
    """Runs the privacy guard on a profile and the synthetic data generated from it.

    Args:
        real: Real DataFrames by table name.
        synth: Synthetic DataFrames by table name.
        profile: Profile of the real data.
        manifest: Ground-truth manifest of the real data.
        holdout: Real holdout DataFrames; defaults to the catalog dataset at seed `manifest.seed` plus the configured
            offset.
        seed: Seed of the distance-to-closest-record samples.
        config: Metrics configuration; defaults to the packaged one.

    Raises:
        ValueError: If the Spark session time zone is not UTC.
    """
    config = config or load_config()
    require_utc(list(real.values()))
    session = next(iter(real.values())).sparkSession
    if holdout is None:
        offset = config.sampling.holdout_seed_offset
        holdout = load(manifest.dataset, manifest.tier, manifest.seed + offset, session)
    limits = config.privacy
    checks: list[GuardCheck] = []
    for table in manifest.tables:
        if table.name in synth and table.name in real and table.name in holdout:
            checks += _table_checks(table, real[table.name], synth[table.name], holdout[table.name], seed, config)
    leaks = _profile_leaks(profile, manifest, real, limits.min_substring_length)
    checks.append(_at_most("profile.sensitive_value_leaks", sum(leaks.values()), limits.max_profile_leaks, leaks))
    violations = _k_violations(profile, real)
    checks.append(_at_most("profile.k_threshold", sum(violations.values()), limits.max_k_violations, violations))
    return PrivacyScore(
        dataset=manifest.dataset,
        config_version=config.version,
        config_hash=config.hash(),
        checks=checks,
        passed=all(c.passed for c in checks),
    )


def _at_most(name: str, value: float, threshold: float, detail: Mapping[str, float] | None = None) -> GuardCheck:
    return GuardCheck(name=name, value=value, threshold=threshold, passed=value <= threshold, detail=dict(detail or {}))


def _excess(name: str, synth: tuple[float, int], holdout: tuple[float, int], threshold: float, z: float) -> GuardCheck:
    """Returns a check that the synth rate exceeds the holdout rate by at most `threshold` plus `z` standard errors."""
    (synth_rate, synth_n), (holdout_rate, holdout_n) = synth, holdout
    pooled = (synth_rate * synth_n + holdout_rate * holdout_n) / max(synth_n + holdout_n, 1)
    noise = z * math.sqrt(pooled * (1 - pooled) * (1 / max(synth_n, 1) + 1 / max(holdout_n, 1)))
    value = synth_rate - holdout_rate
    return GuardCheck(
        name=name,
        value=value,
        threshold=threshold,
        passed=value <= threshold + noise,
        detail={"synth": synth_rate, "holdout": holdout_rate, "noise_allowance": noise},
    )


def _table_checks(
    truth: Table, real: DataFrame, synth: DataFrame, holdout: DataFrame, seed: int, config: MetricsConfig
) -> list[GuardCheck]:
    limits = config.privacy
    real_types = {f.name: f.dataType.simpleString() for f in real.schema.fields}
    synth_types = {f.name: f.dataType.simpleString() for f in synth.schema.fields}
    shared = [n for n, t in real_types.items() if synth_types.get(n) == t]
    checks = []
    if shared:
        checks.append(
            _excess(
                f"{truth.name}.copy_rate_excess",
                _match_rate(synth, real, shared),
                _match_rate(holdout, real, shared),
                limits.max_copy_rate_excess,
                limits.noise_z,
            )
        )
        checks.append(_dcr(truth, real, synth.select(*(identifier(n) for n in shared)), holdout, seed, config))
    for path, category in sensitive_columns(truth).items():
        if category in VALUE_LEAK_CATEGORIES and path.split(".")[0] in shared:
            checks.append(
                _excess(
                    f"{truth.name}.{path}.value_leak_rate_excess",
                    _value_leak_rate(synth, real, path),
                    _value_leak_rate(holdout, real, path),
                    limits.max_synth_leak_rate_excess,
                    limits.noise_z,
                )
            )
    for columns in quasi_identifier_sets(truth):
        if set(columns) <= set(shared):
            checks.append(
                _excess(
                    f"{truth.name}.quasi_identifiers[{','.join(columns)}].match_rate_excess",
                    _match_rate(synth, real, columns, unique_only=True),
                    _match_rate(holdout, real, columns, unique_only=True),
                    limits.max_qi_match_rate_excess,
                    limits.noise_z,
                )
            )
    return checks


def _row_key(df: DataFrame, columns: Sequence[str]) -> DataFrame:
    return df.selectExpr(f"to_json(struct({', '.join(identifier(c) for c in columns)})) AS k")


def _match_rate(
    rows: DataFrame, reference: DataFrame, columns: Sequence[str], unique_only: bool = False
) -> tuple[float, int]:
    """Returns the share of `rows` whose `columns` equal those of a `reference` row (one unique in the reference).

    Returns:
        The share and the number of rows it is over.
    """
    keys = _row_key(rows, columns)
    ref = _row_key(reference, columns)
    ref = ref.groupBy("k").count().where("count = 1").select("k") if unique_only else ref.distinct()
    total = keys.count()
    return (keys.join(ref, "k", "left_semi").count() / total if total else 0.0), total


def _value_leak_rate(rows: DataFrame, reference: DataFrame, path: str) -> tuple[float, int]:
    """Returns the share of distinct non-null `reference` values at `path` that also appear in `rows`, and their count."""
    expr = ".".join(identifier(p) for p in path.split("."))
    values = reference.selectExpr(f"{expr} AS v").dropna().distinct()
    total = values.count()
    found = rows.selectExpr(f"{expr} AS v").dropna().distinct().join(values, "v", "left_semi").count()
    return (found / total if total else 0.0), total


def _dcr(
    truth: Table, real: DataFrame, synth: DataFrame, holdout: DataFrame, seed: int, config: MetricsConfig
) -> GuardCheck:
    cap = config.sampling.privacy_rows
    real_leaves, real_frames = collect(real, truth, real.count(), cap, seed)
    synth_frame = collect(synth, truth, synth.count(), cap, seed)[1][ROW_FRAME]
    holdout_frame = collect(holdout, truth, holdout.count(), cap, seed)[1][ROW_FRAME]
    reference = real_frames[ROW_FRAME]
    kinds = {
        leaf.path: _encoder_kind(leaf_kind(leaf, reference[leaf.path], config.marginal.categorical_max_distinct))
        for leaf in real_leaves
        if leaf.frame == ROW_FRAME and leaf.path in synth_frame.columns
    }
    name = f"{truth.name}.dcr_ratio"
    if not kinds or min(len(reference), len(synth_frame), len(holdout_frame)) == 0:
        return GuardCheck(name=name, value=1.0, threshold=config.privacy.min_dcr_ratio, passed=True)
    encoder = stats.Encoder(reference, kinds, config.marginal.top_values)
    synth_distances = _closest(encoder, reference, synth_frame)
    holdout_distances = _closest(encoder, reference, holdout_frame)
    synth_median, holdout_median = float(np.median(synth_distances)), float(np.median(holdout_distances))
    ratio = synth_median / holdout_median if holdout_median > 0 else (1.0 if synth_median == 0 else float("inf"))
    return GuardCheck(
        name=name,
        value=min(ratio, 1e9),
        threshold=config.privacy.min_dcr_ratio,
        passed=ratio >= config.privacy.min_dcr_ratio,
        detail={
            "synth_median": synth_median,
            "holdout_median": holdout_median,
            "synth_zero_share": float((synth_distances == 0).mean()),
            "holdout_zero_share": float((holdout_distances == 0).mean()),
        },
    )


def _encoder_kind(kind: str) -> str:
    return "numeric" if kind in ("numeric", "length") else kind


def _closest(encoder: stats.Encoder, reference: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    columns = list(encoder.kinds)
    ref, frame = reference[columns], rows[columns]
    distances = stats.squared_distances(encoder.features(frame), encoder.features(ref))
    codes = encoder.codes(frame, ref)
    if codes:
        distances = distances + stats.mismatches([c[0] for c in codes], [c[1] for c in codes])
    closest: np.ndarray = np.sqrt(distances.min(axis=1))
    return closest


def _profile_strings(profile: Profile) -> set[str]:
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, str):
            found.add(node)
        elif isinstance(node, dict):
            for key, value in node.items():
                found.add(str(key))
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(profile.model_dump(mode="json"))
    for table in profile.tables:
        for column in _profile_columns(table.columns):
            for pattern in (column.string.patterns or []) if column.string else []:
                found.update(regex.literals(pattern.regex))
    return found


def _profile_columns(columns: Sequence[ColumnProfile]) -> Iterator[ColumnProfile]:
    for column in columns:
        yield column
        yield from _profile_columns(column.children or [])


def _profile_leaks(
    profile: Profile, manifest: Manifest, real: Mapping[str, DataFrame], min_length: int
) -> dict[str, float]:
    strings = sorted(_profile_strings(profile))
    if not strings or not real:
        return {}
    session = next(iter(real.values())).sparkSession
    candidates = F.broadcast(session.createDataFrame([(s,) for s in strings], "c string"))
    leaks: dict[str, float] = {}
    for table in manifest.tables:
        if table.name not in real:
            continue
        for path, category in sensitive_columns(table).items():
            if category not in VALUE_LEAK_CATEGORIES:
                continue
            expr = ".".join(identifier(p) for p in path.split("."))
            values = real[table.name].selectExpr(f"CAST({expr} AS STRING) AS v").dropna().distinct()
            condition = (F.col("c") == F.col("v")) | (
                (F.length("v") >= min_length) & (F.instr(F.col("c"), F.col("v")) > 0)
            )
            leaks[f"{table.name}.{path}"] = float(values.join(candidates, condition).select("v").distinct().count())
    return leaks


def _k_violations(profile: Profile, real: Mapping[str, DataFrame]) -> dict[str, float]:
    k = profile.privacy.min_value_count
    violations: dict[str, float] = {}
    try:
        Profile.model_validate(profile.model_dump(mode="json"))
    except ValidationError as error:
        violations["profile validation"] = float(len(error.errors()))
    for table in profile.tables:
        if table.name not in real:
            continue
        df = real[table.name]
        listed: list[tuple[str, DataFrame, list[str]]] = []
        for path, values, column in _literal_columns(df, table.columns):
            if column.string and column.string.top_values:
                listed.append((f"{path} top_values", values, [v.value for v in column.string.top_values]))
        top_level = {c.name for c in table.columns}
        for constraint in table.check_constraints or []:
            if isinstance(constraint, EnumerationConstraint) and constraint.column in top_level:
                values = df.selectExpr(f"{identifier(constraint.column)} AS v")
                literals = [str(v).lower() if isinstance(v, bool) else str(v) for v in constraint.values]
                listed.append((f"{constraint.column} enumeration", values, literals))
        for label, values, literals in listed:
            counts = {
                r["v"]: r["count"]
                for r in values.selectExpr("CAST(v AS STRING) AS v")
                .where(F.col("v").isin(literals))
                .groupBy("v")
                .count()
                .collect()
            }
            rare = sum(1 for literal in set(literals) if counts.get(literal, 0) < k)
            if rare:
                violations[f"{table.name}.{label}"] = float(rare)
    return violations


def _literal_columns(df: DataFrame, columns: Sequence[ColumnProfile]) -> Iterator[tuple[str, DataFrame, ColumnProfile]]:
    for column in columns:
        yield from _walk(df, column, identifier(column.name), column.name)


def _walk(
    frame: DataFrame, column: ColumnProfile, expr: str, path: str
) -> Iterator[tuple[str, DataFrame, ColumnProfile]]:
    yield path, frame.selectExpr(f"{expr} AS v"), column
    children = column.children or []
    if column.logical_type == LogicalType.STRUCT:
        for child in children:
            yield from _walk(frame, child, f"{expr}.{identifier(child.name)}", f"{path}.{child.name}")
    elif column.logical_type == LogicalType.ARRAY:
        exploded = frame.selectExpr(f"explode({expr}) AS element")
        for child in children:
            yield from _walk(exploded, child, "element", f"{path}.element")
    elif column.logical_type == LogicalType.MAP:
        exploded = frame.selectExpr(f"explode({expr}) AS (key, value)")
        for child in children:
            yield from _walk(exploded, child, child.name, f"{path}.{child.name}")
