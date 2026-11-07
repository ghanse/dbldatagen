"""Privacy guard: whether a profile or the synthetic data generated from it gives away real rows or sensitive values.

Data checks compare the synth with a real holdout: another row sample of the same dataset structure. The holdout
never reached the profiler, so the rate at which it matches the real data is the rate expected by chance.
Every check is pass/fail against a threshold of the metrics configuration; values themselves are never reported.
"""

from __future__ import annotations

import itertools
import math
import re
import zlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import ValidationError
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from dbldatagen_profiler.model import ColumnProfile, EnumerationConstraint, LogicalType, Profile
from profiler_bench.catalog.manifest import Manifest, Table
from profiler_bench.fidelity import regex, stats
from profiler_bench.fidelity.config import MetricsConfig, load_config
from profiler_bench.fidelity.frames import ROW_FRAME, bounded, collect, identifier, leaf_kind, require_utc
from profiler_bench.fidelity.scores import GuardCheck, PrivacyScore
from profiler_bench.fidelity.truth import (
    VALUE_LEAK_CATEGORIES,
    load_instance,
    quasi_identifier_sets,
    sensitive_columns,
)


def score_privacy(
    real: Mapping[str, DataFrame],
    synth: Mapping[str, DataFrame],
    profile: Profile,
    manifest: Manifest,
    *,
    holdout: Mapping[str, DataFrame] | None = None,
    sample_seed: int | None = None,
    seed: int = 0,
    config: MetricsConfig | None = None,
) -> PrivacyScore:
    """Runs the privacy guard on a profile and the synthetic data generated from it.

    Args:
        real: Real DataFrames by table name.
        synth: Synthetic DataFrames by table name.
        profile: Profile of the real data.
        manifest: Ground-truth manifest of the real data.
        holdout: Real holdout DataFrames; defaults to the catalog dataset at the manifest's structure seed and sample
            seed `sample_seed + sampling.holdout_sample_offset`.
        sample_seed: Catalog sample seed of `real`, used to pick the holdout; defaults to the structure seed, which is
            the catalog's default sample seed.
        seed: Seed of the distance-to-closest-record samples.
        config: Metrics configuration; defaults to the packaged one.

    Raises:
        ValueError: If the Spark session time zone is not UTC.
    """
    config = config or load_config()
    require_utc(list(real.values()))
    session = next(iter(real.values())).sparkSession
    if holdout is None:
        holdout = load_instance(
            manifest.dataset,
            manifest.tier,
            structure_seed=manifest.structure_seed,
            sample_seed=(manifest.structure_seed if sample_seed is None else sample_seed)
            + config.sampling.holdout_sample_offset,
            spark=session,
        )
    limits = config.privacy
    checks: list[GuardCheck] = []
    for table in manifest.tables:
        if table.name not in real:
            continue
        if table.name not in synth:
            checks.append(_unmeasured(f"{table.name}.synth_table_present", 1.0))
            continue
        keys = set(table.primary_key) | {
            c for fk in manifest.foreign_keys if fk.child_table == table.name for c in fk.child_columns
        }
        checks += _table_checks(table, real[table.name], synth[table.name], holdout[table.name], keys, seed, config)
    risks = _pattern_risks(profile, config)
    leaks = _profile_leaks(profile, manifest, real, risks, seed, config.sampling.privacy_rows)
    violations = _k_violations(profile, real, risks, seed, config.sampling.privacy_rows)
    budget_hits = float(sum(1 for r in risks.values() if r.budget_exceeded))
    if budget_hits:
        leaks = {**leaks, "regex_budget_exceeded": budget_hits}
        violations = {**violations, "regex_budget_exceeded": budget_hits}
    checks.append(_at_most("profile.sensitive_value_leaks", sum(leaks.values()), limits.max_profile_leaks, leaks))
    checks.append(_at_most("profile.k_threshold", sum(violations.values()), limits.max_k_violations, violations))
    return PrivacyScore(
        dataset=manifest.dataset,
        config_version=config.version,
        config_hash=config.hash(),
        checks=checks,
        passed=all(c.passed for c in checks),
    )


def _unmeasured(name: str, threshold: float) -> GuardCheck:
    """Returns a failed check for something the synth lacks, so a missing table or column can never pass."""
    return GuardCheck(name=name, value=0.0, threshold=threshold, passed=False, detail={"unmeasured": 1.0})


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
    truth: Table,
    real: DataFrame,
    synth: DataFrame,
    holdout: DataFrame,
    keys: set[str],
    seed: int,
    config: MetricsConfig,
) -> list[GuardCheck]:
    limits = config.privacy
    real_types = {f.name: f.dataType.simpleString() for f in real.schema.fields}
    synth_types = {f.name: f.dataType.simpleString() for f in synth.schema.fields}
    shared = [n for n, t in real_types.items() if synth_types.get(n) == t]
    coverage = len(shared) / len(real_types) if real_types else 1.0
    real_rows, synth_rows = real.count(), synth.count()
    present = 1.0 if synth_rows or not real_rows else 0.0
    checks = [
        GuardCheck(name=f"{truth.name}.synth_columns_present", value=coverage, threshold=1.0, passed=coverage >= 1.0),
        GuardCheck(name=f"{truth.name}.synth_rows_present", value=present, threshold=1.0, passed=present >= 1.0),
    ]
    if not shared:
        names = ("copy_rate_excess", "dcr_ratio", "partial_copy_rate_excess")
        thresholds = (limits.max_copy_rate_excess, limits.min_dcr_ratio, limits.max_copy_rate_excess)
        checks += [_unmeasured(f"{truth.name}.{n}", t) for n, t in zip(names, thresholds, strict=True)]
    else:
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
        values = [c for c in shared if c not in keys]
        checks += _partial_copy(truth.name, real, synth, holdout, values, seed, config)
    for path, category in sensitive_columns(truth).items():
        if category not in VALUE_LEAK_CATEGORIES:
            continue
        name = f"{truth.name}.{path}.value_leak_rate_excess"
        if path.split(".")[0] not in shared:
            checks.append(_unmeasured(name, limits.max_synth_leak_rate_excess))
            continue
        checks.append(
            _excess(
                name,
                _value_leak_rate(synth, real, path),
                _value_leak_rate(holdout, real, path),
                limits.max_synth_leak_rate_excess,
                limits.noise_z,
            )
        )
    for columns in quasi_identifier_sets(truth):
        name = f"{truth.name}.quasi_identifiers[{','.join(columns)}].match_rate_excess"
        if not set(columns) <= set(shared):
            checks.append(_unmeasured(name, limits.max_qi_match_rate_excess))
            continue
        checks.append(
            _excess(
                name,
                _match_rate(synth, real, columns, unique_only=True),
                _match_rate(holdout, real, columns, unique_only=True),
                limits.max_qi_match_rate_excess,
                limits.noise_z,
            )
        )
    return checks


def _partial_copy(
    table: str,
    real: DataFrame,
    synth: DataFrame,
    holdout: DataFrame,
    shared: Sequence[str],
    seed: int,
    config: MetricsConfig,
) -> list[GuardCheck]:
    """Returns a check that no column pair reproduces real-unique value pairs more often than the holdout does.

    A synth row that copies some columns of a real row, row-aligned, reproduces the real row's combination of values
    in every pair of copied columns; combinations unique in the real table do not recur by chance. Primary key and
    foreign key columns are left out by the caller: their values follow from the key structure, which any sampler
    reproduces, so pairs with them measure that structure rather than copying.
    """
    pairs = list(itertools.combinations(shared, 2))
    if not pairs:
        return []
    limit = config.privacy.partial_copy_pairs
    if len(pairs) > limit:
        rng = np.random.default_rng([seed, zlib.crc32(table.encode())])
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), size=limit, replace=False))]
    synth_rates, holdout_rates = _pair_match_rates(synth, real, pairs), _pair_match_rates(holdout, real, pairs)
    checks = [
        _excess("", synth_rate, holdout_rate, config.privacy.max_copy_rate_excess, config.privacy.noise_z)
        for synth_rate, holdout_rate in zip(synth_rates, holdout_rates, strict=True)
    ]
    worst = max(range(len(pairs)), key=lambda i: checks[i].value)
    return [
        GuardCheck(
            name=f"{table}.partial_copy_rate_excess",
            value=checks[worst].value,
            threshold=config.privacy.max_copy_rate_excess,
            passed=all(c.passed for c in checks),
            detail={
                **checks[worst].detail,
                "pairs": float(len(pairs)),
                "failed_pairs": float(sum(not c.passed for c in checks)),
            },
        )
    ]


def _pair_keys(df: DataFrame, pairs: Sequence[tuple[str, str]]) -> DataFrame:
    keys = ", ".join(
        f"named_struct('p', {i}, 'k', to_json(struct({identifier(a)} AS a, {identifier(b)} AS b)))"
        for i, (a, b) in enumerate(pairs)
    )
    return df.selectExpr(f"inline(array({keys}))")


def _pair_match_rates(
    rows: DataFrame, reference: DataFrame, pairs: Sequence[tuple[str, str]]
) -> list[tuple[float, int]]:
    """Returns, for each column pair, the share of `rows` whose pair of values is unique in `reference`."""
    total = rows.count()
    unique = _pair_keys(reference, pairs).groupBy("p", "k").count().where("count = 1").select("p", "k")
    matched = {
        r["p"]: r["n"]
        for r in _pair_keys(rows, pairs)
        .join(unique, ["p", "k"], "left_semi")
        .groupBy("p")
        .agg(F.count(F.lit(1)).alias("n"))
        .collect()
    }
    return [((matched.get(i, 0) / total if total else 0.0), total) for i in range(len(pairs))]


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


@dataclass(frozen=True)
class _PatternRisk:
    """What a profile regex could give away.

    Attributes:
        compiled: The regex.
        small: Whether it matches fewer than max(k, `privacy.min_regex_language`) strings within its column's length
            range, so its matches are as good as literals.
        literals: Literal factors at least `privacy.min_literal_len` long that every match contains.
        budget_exceeded: Whether language counting exhausted the DFA state/work budget; treated as an unconditional
            privacy failure with reason `regex_budget_exceeded`.
    """

    compiled: re.Pattern[str]
    small: bool
    literals: tuple[str, ...]
    budget_exceeded: bool = False


def _pattern_risks(profile: Profile, config: MetricsConfig) -> dict[tuple[str, str, int], _PatternRisk]:
    """Returns the risk of every profile regex, keyed by table, column path, and pattern index."""
    limits = config.privacy
    limit = max(float(profile.privacy.min_value_count), limits.min_regex_language)
    risks = {}
    for table in profile.tables:
        for path, column in _profile_paths(table.columns, ""):
            stats_ = column.string
            for i, pattern in enumerate((stats_.patterns or []) if stats_ else []):
                try:
                    compiled = re.compile(pattern.regex)
                except re.error:
                    continue
                length = stats_.length if stats_ else None
                low = length.min if length and length.min is not None else 0
                high = length.max if length and length.max is not None else limits.max_regex_length
                budget_exceeded = False
                try:
                    counted = regex.count_matches(pattern.regex, low, high, max_dfa_states=limits.max_dfa_states)
                except regex.UnsupportedForCount:
                    small = True
                else:
                    if isinstance(counted, int):
                        small = counted < limit
                    else:
                        small, budget_exceeded = False, True
                literals = tuple(f for f in regex.required_literals(pattern.regex) if len(f) >= limits.min_literal_len)
                risks[(table.name, path, i)] = _PatternRisk(compiled, small, literals, budget_exceeded)
    return risks


def _profile_paths(columns: Sequence[ColumnProfile], prefix: str) -> Iterator[tuple[str, ColumnProfile]]:
    for column in columns:
        yield f"{prefix}{column.name}", column
        yield from _profile_paths(column.children or [], f"{prefix}{column.name}.")


def _rare_literals(rows: DataFrame, literals: set[str], k: int) -> set[str]:
    """Returns the literals contained in at least one but fewer than `k` of the rows' values (column `v`)."""
    if not literals:
        return set()
    text = rows.selectExpr("CAST(v AS STRING) AS v").dropna()
    counts = text.agg(
        *(F.sum((F.instr("v", F.lit(f)) > 0).cast("long")).alias(f"l{i}") for i, f in enumerate(literals))
    )
    row = counts.collect()[0]
    return {f for i, f in enumerate(literals) if 0 < (row[f"l{i}"] or 0) < k}


def _sample_values(values: DataFrame, cap: int, seed: int) -> list[str]:
    """Returns up to about `cap` distinct non-null values of column `v`, chosen by a seeded hash, as strings."""
    distinct = values.selectExpr("CAST(v AS STRING) AS v").dropna().distinct()
    return [r["v"] for r in bounded(distinct, distinct.count(), cap, seed).select("v").collect()]


def _profile_leaks(
    profile: Profile,
    manifest: Manifest,
    real: Mapping[str, DataFrame],
    risks: Mapping[tuple[str, str, int], _PatternRisk],
    seed: int,
    cap: int,
) -> dict[str, float]:
    """Counts sensitive real values the profile gives away, per column.

    A value leaks when it equals a string of the profile JSON or a literal run of a profile regex (checked on every
    value), when a small-language profile regex fully matches it, or when it contains a literal that some profile
    regex requires of every match and that fewer than k real rows of the column contain (both checked on a seeded
    sample of the column's values). A literal shared by k or more rows, such as an email domain, does not single a
    value out.
    """
    k = profile.privacy.min_value_count
    strings = sorted(_profile_strings(profile))
    if not real:
        return {}
    session = next(iter(real.values())).sparkSession
    candidates = F.broadcast(session.createDataFrame([(s,) for s in strings], "c string"))
    small = [r.compiled for r in risks.values() if r.small]
    literals = {f for r in risks.values() for f in r.literals}
    leaks: dict[str, float] = {}
    for table in manifest.tables:
        if table.name not in real:
            continue
        for path, category in sensitive_columns(table).items():
            if category not in VALUE_LEAK_CATEGORIES:
                continue
            expr = ".".join(identifier(p) for p in path.split("."))
            rows = real[table.name].selectExpr(f"CAST({expr} AS STRING) AS v").dropna()
            values = rows.distinct()
            exact = {r["v"] for r in values.join(candidates, F.col("c") == F.col("v"), "left_semi").collect()}
            sample = _sample_values(values, cap, seed)
            matched = {v for v in sample for p in small if p.fullmatch(v)}
            present = {f for f in literals if any(f in v for v in sample)}
            rare = _rare_literals(rows, present, k)
            contained = {v for v in sample if any(f in v for f in rare)}
            leaks[f"{table.name}.{path}"] = float(len(exact | matched | contained))
    return leaks


def _k_violations(
    profile: Profile,
    real: Mapping[str, DataFrame],
    risks: Mapping[tuple[str, str, int], _PatternRisk],
    seed: int,
    cap: int,
) -> dict[str, float]:
    """Counts literal values of the profile held by fewer than `privacy.min_value_count` real rows, per source.

    Sources are `top_values`, enumeration constraints, the matches of small-language regexes among a seeded sample
    of the column's values, and the literals a regex requires of every match (see `_PatternRisk`).
    """
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
            for i, _ in enumerate((column.string.patterns or []) if column.string else []):
                risk = risks.get((table.name, path, i))
                if risk is not None:
                    listed += _pattern_literals(f"{path} pattern {i}", values, risk, seed, cap)
                    rare_literals = _rare_literals(values, set(risk.literals), k)
                    if rare_literals:
                        violations[f"{table.name}.{path} pattern {i} literals"] = float(len(rare_literals))
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


def _pattern_literals(
    label: str, values: DataFrame, risk: _PatternRisk, seed: int, cap: int
) -> list[tuple[str, DataFrame, list[str]]]:
    """Returns the sampled values a small-language regex matches, to be held to the k-threshold like literals."""
    if not risk.small:
        return []
    hits = [v for v in _sample_values(values, cap, seed) if risk.compiled.fullmatch(v)]
    return [(label, values, hits)] if hits else []


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
