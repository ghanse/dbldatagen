"""Structure recovery: how much of a dataset's planted structure a profile states.

Each planted item (a key, foreign key, check, functional dependency, dependency, string format, null rate, or
cardinality) earns a credit in [0, 1]. Recall is the mean credit over planted items; precision is the mean credit over
the profile's claims about the same targets. Key, foreign key, check, and dependency claims about structure the
manifest does not plant are checked against the real data (see `claims`): true ones are left out of precision and
false ones count as 0. A structure the profile spec cannot express, or the profile omits, is reported as not
recovered. Planted values are the manifest's generating parameters.
"""

from __future__ import annotations

import math
import re
import zlib
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np
from pyspark.sql import DataFrame, SparkSession

from dbldatagen_profiler.model import (
    Cardinality,
    CheckConstraint,
    ColumnProfile,
    ComparisonConstraint,
    EnumerationConstraint,
    ExpressionConstraint,
    LogicalType,
    Profile,
    RangeConstraint,
    TableProfile,
    parse_temporal,
)
from profiler_bench.catalog.manifest import (
    Bound,
    Check,
    Column,
    ComparisonCheck,
    EnumerationCheck,
    ExpressionCheck,
    Manifest,
    RangeCheck,
    Table,
)
from profiler_bench.catalog.verification import _rate
from profiler_bench.fidelity import regex
from profiler_bench.fidelity.claims import Verifier
from profiler_bench.fidelity.config import MetricsConfig, load_config
from profiler_bench.fidelity.scores import (
    StructureCategory,
    StructureItem,
    StructureScore,
    TableStructure,
    mean,
    weighted,
)
from profiler_bench.fidelity.truth import load_instance


FLIPPED = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "=", "!=": "!="}
IMPLIES = {"<": {"<", "<=", "!="}, ">": {">", ">=", "!="}, "=": {"=", "<=", ">="}}


def score_structure(
    profile: Profile,
    manifest: Manifest,
    *,
    real: Mapping[str, DataFrame] | None = None,
    sample_seed: int | None = None,
    seed: int = 0,
    config: MetricsConfig | None = None,
) -> StructureScore:
    """Scores how much of the structure planted in a catalog dataset the profile recovered.

    Args:
        profile: Profile of the dataset.
        manifest: Ground-truth manifest of the dataset.
        real: Real DataFrames the profile was computed from, used to check claims about unplanted structure;
            defaults to the catalog instance at the manifest's structure seed and `sample_seed`.
        sample_seed: Catalog sample seed of the real data; defaults to the structure seed.
        seed: Seed of the strings generated to test regex equivalence and of the dependency samples.
        config: Metrics configuration; defaults to the packaged one.

    Raises:
        RuntimeError: If `real` is not given and no Spark session is active.
    """
    config = config or load_config()
    if real is None:
        session = SparkSession.getActiveSession()
        if session is None:
            raise RuntimeError("score_structure needs `real` or an active SparkSession")
        sample = manifest.structure_seed if sample_seed is None else sample_seed
        real = load_instance(
            manifest.dataset, manifest.tier, structure_seed=manifest.structure_seed, sample_seed=sample, spark=session
        )
    verifier = Verifier(real, seed, config)
    profiled = {t.name: t for t in profile.tables}
    tables = [_score_table(t, profiled.get(t.name), profile, manifest, verifier, seed, config) for t in manifest.tables]
    return StructureScore(
        dataset=manifest.dataset,
        config_version=config.version,
        config_hash=config.hash(),
        tables=tables,
        score=mean([t.score for t in tables]) or 0.0,
    )


def _score_table(
    truth: Table,
    table: TableProfile | None,
    profile: Profile,
    manifest: Manifest,
    verifier: Verifier,
    seed: int,
    config: MetricsConfig,
) -> TableStructure:
    tol = config.structure_tolerances
    categories = {
        "keys": _keys(truth, table, verifier),
        "foreign_keys": _foreign_keys(truth, profile, manifest, verifier),
        "checks": _checks(truth, table, tol.range_relative, verifier),
        "functional_dependencies": _functional_dependencies(truth, table),
        "dependencies": _dependencies(truth, table, tol.dependency, verifier),
        "patterns": _patterns(truth, table, seed, config),
        "null_rates": _null_rates(truth, table),
        "cardinality": _cardinality(truth, table, tol.distinct_log_ratio),
    }
    present = {name: c for name, c in categories.items() if c is not None}
    _, weights, score = weighted({name: c.score for name, c in present.items()}, config.structure_weights)
    return TableStructure(name=truth.name, categories=present, weights=weights, score=score)


def _category(
    items: list[StructureItem], claims: Sequence[float] | None, verified: Sequence[bool] = ()
) -> StructureCategory | None:
    """Returns a category's recall, precision, and score; recall is 1 when nothing is planted but claims were scored,
    so false claims about a table without planted structure still cost precision."""
    if not items and not claims:
        return None
    recall = sum(i.credit for i in items) / len(items) if items else 1.0
    precision = sum(claims) / len(claims) if claims else None
    if precision is None:
        score = recall
    else:
        score = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return StructureCategory(
        recall=recall,
        precision=precision,
        score=score,
        items=items,
        claims=len(claims) if claims else 0,
        verified_true=sum(verified),
        verified_false=len(verified) - sum(verified),
    )


def _item(target: str, credit: float, note: str | None = None) -> StructureItem:
    status = "recovered" if credit >= 1.0 else "partial" if credit > 0 else "not recovered"
    return StructureItem(target=target, status=status, credit=credit, note=None if credit >= 1.0 else note)


def _unplanted(verified: Sequence[bool]) -> list[float]:
    """Returns the precision entries of claims about unplanted structure: none for each true one, 0 for each false."""
    return [0.0] * (len(verified) - sum(verified))


def _keys(truth: Table, table: TableProfile | None, verifier: Verifier) -> StructureCategory | None:
    primary = frozenset(truth.primary_key)
    unique = {frozenset(k) for k in truth.unique_keys}
    unique |= {frozenset({c.name}) for c in truth.columns if c.unique and frozenset({c.name}) != primary}
    claimed_primary = frozenset(table.primary_key.columns) if table and table.primary_key else None
    claimed_unique = {frozenset(k.columns) for k in (table.unique_keys or [])} if table else set()
    claimed_all = claimed_unique | ({claimed_primary} if claimed_primary else set())
    items = []
    if primary:
        items.append(_item(f"primary key {sorted(primary)}", float(claimed_primary == primary), "not claimed"))
    items += [
        _item(f"unique key {sorted(k)}", float(k in claimed_all), "not claimed") for k in sorted(unique, key=sorted)
    ]
    planted = unique | ({primary} if primary else set())
    claims = [1.0 for k in claimed_all if k in planted]
    verified = [
        verifier.key(truth.name, sorted(key), primary=key == claimed_primary)
        for key in sorted(claimed_all - planted, key=sorted)
    ]
    return _category(items, claims + _unplanted(verified), verified)


def _foreign_keys(truth: Table, profile: Profile, manifest: Manifest, verifier: Verifier) -> StructureCategory | None:
    def signature(child: str, cc: Sequence[str], parent: str, pc: Sequence[str]) -> tuple[Any, ...]:
        return (child, tuple(cc), parent, tuple(pc))

    planted = {
        signature(fk.child_table, fk.child_columns, fk.parent_table, fk.parent_columns)
        for fk in manifest.foreign_keys
        if fk.child_table == truth.name
    }
    relationships = {
        signature(r.child_table, r.child_columns, r.parent_table, r.parent_columns): r
        for r in profile.relationships or []
        if r.child_table == truth.name
    }
    items = [
        _item(f"{s[0]}{list(s[1])} -> {s[2]}{list(s[3])}", float(s in relationships), "not claimed")
        for s in sorted(planted)
    ]
    claims = [1.0 for s in relationships if s in planted]
    verified = [verifier.foreign_key(relationships[s]) for s in sorted(set(relationships) - planted)]
    return _category(items, claims + _unplanted(verified), verified)


def _number(value: Bound | bool) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except ValueError:
        pass
    try:
        if len(value) == 10:
            return float(parse_temporal(value, LogicalType.DATE).toordinal())
        moment = parse_temporal(value, LogicalType.TIMESTAMP)
        assert isinstance(moment, datetime)
        moment = moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
        return moment.timestamp() / 86_400
    except ValueError:
        return None


def _bound_credit(planted: Bound | None, claimed: Bound | bool | None, span: float, tolerance: float) -> float:
    if planted is None:
        return 1.0
    p, c = _number(planted), _number(claimed) if claimed is not None else None
    if p is None or c is None:
        return 0.0
    return float(abs(p - c) <= tolerance * span)


def _range_credit(check: RangeCheck, claim: RangeConstraint, tolerance: float) -> float:
    low, high = (_number(b) if b is not None else None for b in (check.min, check.max))
    span = high - low if low is not None and high is not None else max(abs(low or high or 0.0), 1.0)
    span = span or 1.0
    return _bound_credit(check.min, claim.min, span, tolerance) * _bound_credit(check.max, claim.max, span, tolerance)


def _normalize_sql(sql: str) -> str:
    text = re.sub(r"\s+", " ", sql.replace("`", "")).strip().lower()
    while text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return text


def _comparison(left: str, op: str, right: str) -> tuple[str, str, str]:
    return (left, op, right) if left <= right else (right, FLIPPED[op], left)


def _comparison_credit(check: ComparisonCheck, claim: ComparisonConstraint) -> float:
    planted, claimed = _comparison(check.left, check.op, check.right), _comparison(
        claim.left, claim.operator, claim.right
    )
    if planted[0::2] != claimed[0::2]:
        return 0.0
    return float(claimed[1] == planted[1] or planted[1] in IMPLIES.get(claimed[1], set()))


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def _literal(value: object) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def _check_credit(check: Check, claim: CheckConstraint, tolerance: float) -> float | None:
    """Returns the credit `claim` earns for `check`, or None when the claim does not target the same structure."""
    if isinstance(check, RangeCheck) and isinstance(claim, RangeConstraint) and claim.column == check.column:
        return _range_credit(check, claim, tolerance)
    if (
        isinstance(check, EnumerationCheck)
        and isinstance(claim, EnumerationConstraint)
        and claim.column == check.column
    ):
        return _jaccard({_literal(v) for v in check.values}, {_literal(v) for v in claim.values})
    if isinstance(check, ComparisonCheck) and isinstance(claim, ComparisonConstraint):
        return _comparison_credit(check, claim) if {claim.left, claim.right} == {check.left, check.right} else None
    if isinstance(check, ExpressionCheck) and isinstance(claim, ExpressionConstraint):
        return float(_normalize_sql(claim.expression) == _normalize_sql(check.sql))
    return None


def _check_target(check: Check) -> str:
    if isinstance(check, RangeCheck):
        return f"range {check.column} in [{check.min}, {check.max}]"
    if isinstance(check, EnumerationCheck):
        return f"enumeration {check.column}"
    if isinstance(check, ComparisonCheck):
        return f"{check.left} {check.op} {check.right}"
    return check.sql


def _checks(truth: Table, table: TableProfile | None, tolerance: float, verifier: Verifier) -> StructureCategory | None:
    claims = list(table.check_constraints or []) if table else []
    items: list[StructureItem] = []
    best: dict[int, float] = {}
    for check in truth.checks:
        credits = {i: c for i, claim in enumerate(claims) if (c := _check_credit(check, claim, tolerance)) is not None}
        for i, credit in credits.items():
            best[i] = max(best.get(i, 0.0), credit)
        items.append(_item(_check_target(check), max(credits.values(), default=0.0), "no matching constraint"))
    scored = [best[i] for i in sorted(best)]
    verified = [verifier.check(truth.name, claim) for i, claim in enumerate(claims) if i not in best]
    return _category(items, scored + _unplanted(verified), verified)


def _functional_dependencies(truth: Table, table: TableProfile | None) -> StructureCategory | None:
    claimed = {
        (frozenset(fd.determinant), frozenset(fd.dependent))
        for fd in (getattr(table, "functional_dependencies", None) or [] if table else [])
    }
    note = "not claimed" if claimed else "the profile states no functional dependencies"
    items = [
        _item(
            f"{list(fd.determinant)} -> {list(fd.dependent)}",
            float((frozenset(fd.determinant), frozenset(fd.dependent)) in claimed),
            note,
        )
        for fd in truth.functional_dependencies
    ]
    planted = {(frozenset(fd.determinant), frozenset(fd.dependent)) for fd in truth.functional_dependencies}
    return _category(items, [float(c in planted) for c in claimed])


def _dependency_key(columns: Sequence[str], method: str) -> tuple[Any, ...]:
    ordered = method == "correlation_ratio"
    return (method, tuple(columns) if ordered else frozenset(columns))


def _dependencies(
    truth: Table, table: TableProfile | None, tolerance: float, verifier: Verifier
) -> StructureCategory | None:
    claims = [(d, _dependency_key(d.columns, d.method.value)) for d in (table.dependencies or [] if table else [])]
    rows = max(truth.row_count, 1)
    planted = {}
    items = []
    for dep in truth.dependencies:
        key = _dependency_key(dep.columns, dep.measure)
        allowed = tolerance + 4 * (1 - dep.value**2) / math.sqrt(rows)
        planted[key] = (dep.value, allowed)
        values = [c.value for c, k in claims if k == key]
        credit = float(any(abs(v - dep.value) <= allowed for v in values))
        note = f"claimed {values}" if values else "not claimed"
        items.append(_item(f"{dep.measure}{list(dep.columns)} = {dep.value:.3f}", credit, note))
    scored = [float(abs(c.value - planted[k][0]) <= planted[k][1]) for c, k in claims if k in planted]
    verified = [verifier.dependency(truth.name, claim) for claim, key in claims if key not in planted]
    return _category(items, scored + _unplanted(verified), verified)


def _children(column: ColumnProfile | None) -> dict[str, ColumnProfile]:
    return {c.name: c for c in (column.children or [])} if column is not None else {}


def _pattern_pairs(
    columns: Sequence[Column], profiled: dict[str, ColumnProfile], prefix: str
) -> list[tuple[str, Column, ColumnProfile | None]]:
    found = []
    for column in columns:
        claim = profiled.get(column.name)
        if column.formats:
            found.append((f"{prefix}{column.name}", column, claim))
        found += _pattern_pairs(column.children(), _children(claim), f"{prefix}{column.name}.")
    return found


def _patterns(truth: Table, table: TableProfile | None, seed: int, config: MetricsConfig) -> StructureCategory | None:
    profiled = {c.name: c for c in table.columns} if table else {}
    threshold = config.structure_tolerances.pattern_equivalence
    count = config.sampling.pattern_samples
    items, claims = [], []
    for path, column, claim in _pattern_pairs(truth.columns, profiled, ""):
        regexes = [p.regex for p in (claim.string.patterns or [])] if claim and claim.string else []
        scores = [[_equivalence(f.regex, r, count, seed) for r in regexes] for f in column.formats]
        for fmt, row in zip(column.formats, scores, strict=True):
            value = max(row, default=0.0)
            items.append(_item(f"{path} ~ {fmt.regex}", float(value >= threshold), f"best two-way match {value:.3f}"))
        claims += [
            float(max((scores[i][j] for i in range(len(scores))), default=0.0) >= threshold)
            for j in range(len(regexes))
        ]
    return _category(items, claims)


def _equivalence(reference: str, candidate: str, count: int, seed: int) -> float:
    try:
        forward = _match_rate(candidate, _samples(reference, count, seed))
        backward = _match_rate(reference, _samples(candidate, count, seed))
    except (regex.UnsupportedRegex, re.error):
        return 0.0
    return min(forward, backward)


def _samples(pattern: str, count: int, seed: int) -> list[str]:
    return regex.generate(pattern, count, np.random.default_rng([seed, zlib.crc32(pattern.encode())]))


def _match_rate(pattern: str, values: list[str]) -> float:
    compiled = re.compile(pattern)
    return sum(compiled.fullmatch(v) is not None for v in values) / len(values) if values else 0.0


def _null_rates(truth: Table, table: TableProfile | None) -> StructureCategory | None:
    profiled = {c.name: c for c in table.columns} if table else {}
    items = []
    for column in truth.columns:
        claim = profiled.get(column.name)
        if claim is None:
            items.append(_item(f"{column.name} null rate {column.null_rate}", 0.0, "column not profiled"))
            continue
        failures = _rate("null rate", claim.null_fraction, column.null_rate, max(truth.row_count, 1))
        items.append(_item(f"{column.name} null rate {column.null_rate}", float(not failures), "; ".join(failures)))
    return _category(items, None)


def _cardinality(truth: Table, table: TableProfile | None, tolerance: float) -> StructureCategory | None:
    profiled = {c.name: c for c in table.columns} if table else {}
    items = []
    for column in truth.columns:
        claim = profiled.get(column.name)
        distinct = claim.distinct_count.value if claim and claim.distinct_count else None
        present = round(truth.row_count * (1 - column.null_rate))
        if column.distinct is not None:
            target, credit = f"{column.name} distinct = {column.distinct}", _close(distinct, column.distinct, tolerance)
        elif column.unique:
            unique = claim is not None and claim.cardinality == Cardinality.UNIQUE
            target, credit = f"{column.name} unique", float(unique or _close(distinct, present, tolerance) == 1.0)
        elif column.max_distinct is not None:
            limit = column.max_distinct * math.exp(tolerance)
            target, credit = f"{column.name} distinct <= {column.max_distinct}", float(
                distinct is not None and distinct <= limit
            )
        else:
            continue
        items.append(_item(target, credit, "no distinct count" if distinct is None else f"profile distinct {distinct}"))
    return _category(items, None)


def _close(value: int | None, target: int, tolerance: float) -> float:
    if value is None:
        return 0.0
    return float(abs(math.log((value + 1) / (target + 1))) <= tolerance)
