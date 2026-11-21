"""Keyed join-consistent substitution of sensitive Spark columns."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import math
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Any

from pyspark.broadcast import Broadcast
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from dbldatagen_profiler.model import ColumnProfile, PiiClass, Profile, Semantic, Sensitivity
from dbldatagen_profiler.pii.policy import (
    PiiAction,
    PiiPolicy,
    PiiRule,
    expected_correlation_attenuation,
)


MAX_NESTED_BROADCAST_VALUES = 100_000


@dataclass(frozen=True)
class SubstitutionOptions:
    """Controls domains, format preservation, generalization, and date shifts."""

    domains: dict[str, str | list[str] | tuple[str, ...]] = field(default_factory=dict)
    keep_email_domain: bool = False
    date_shift_days: int = 30
    generalize: dict[str, str] = field(default_factory=dict)
    surrogate_bits: int = 256


@dataclass(frozen=True)
class SubstitutionResult:
    """Contains substituted DataFrames and a value-free count report."""

    tables: dict[str, DataFrame]
    report: dict[str, Any]


def substitute(  # noqa: PLR0912, PLR0915
    tables: dict[str, DataFrame],
    profile: Profile,
    *,
    key: bytes,
    options: SubstitutionOptions | None = None,
    policy: PiiPolicy | None = None,
) -> SubstitutionResult:
    """Replaces sensitive values with deterministic keyed surrogates."""
    if not key:
        raise ValueError("key must not be empty")
    opts = options or SubstitutionOptions()
    handling = policy or PiiPolicy()
    if opts.date_shift_days < 0:
        raise ValueError("date_shift_days must not be negative")
    if not 1 <= opts.surrogate_bits <= 256:
        raise ValueError("surrogate_bits must be in [1, 256]")
    profiles = {table.name: table for table in profile.tables}
    missing = sorted(set(tables) - set(profiles))
    if missing:
        raise ValueError(f"tables are absent from profile: {missing}")
    domains = _domains(profile, opts)
    decisions = _decisions(profile, handling, opts)
    decisions = _domain_decisions(domains, decisions)
    _validate_perturb_decisions(profile, tables, domains, decisions)
    domain_semantics: dict[str, Semantic] = {}
    for table_name, path, column in _profile_columns(profile):
        decision = decisions.get(f"{table_name}.{path}")
        semantic = column.semantic or (_rule_semantic(decision[0]) if decision is not None else None)
        if semantic is not None and decision is not None and decision[0].action == PiiAction.SUBSTITUTE:
            domain_semantics.setdefault(domains[f"{table_name}.{path}"], semantic)
    mappings: dict[str, DataFrame] = {}
    nested_mappings: dict[str, Broadcast[dict[str, object]]] = {}
    collision_counts: dict[str, int] = {}
    nested_domains = {
        domains[f"{table}.{path}"]
        for table, path, column in _profile_columns(profile)
        if "." in path
        and decisions.get(f"{table}.{path}", (PiiRule(action=PiiAction.KEEP), "default"))[0].action
        == PiiAction.SUBSTITUTE
        and _decision_pii_class(column, decisions[f"{table}.{path}"][0])
        not in {PiiClass.FREE_TEXT_PII, PiiClass.DATE_OF_BIRTH}
    }
    for domain, semantic in domain_semantics.items():
        if semantic.pii_class in {PiiClass.FREE_TEXT_PII, PiiClass.DATE_OF_BIRTH}:
            continue
        members = [reference for reference, member_domain in domains.items() if member_domain == domain]
        mapping, collisions = _build_mapping(tables, members, semantic, domain, key, opts)
        mappings[domain], collision_counts[domain] = mapping, collisions
        if domain in nested_domains:
            rows = mapping.limit(MAX_NESTED_BROADCAST_VALUES + 1).collect()
            if len(rows) > MAX_NESTED_BROADCAST_VALUES:
                raise ValueError(f"nested substitution domain {domain!r} exceeds the broadcast limit")
            session = next(iter(tables.values())).sparkSession
            nested_mappings[domain] = session.sparkContext.broadcast({str(row.original): row.surrogate for row in rows})
    output: dict[str, DataFrame] = {}
    reports: list[dict[str, Any]] = []
    for table_name, frame in tables.items():
        result = frame
        columns = {column.name: column for column in profiles[table_name].columns}
        entity_columns = _entity_columns(profile, table_name)
        entity_keys = _capture_entity_keys(result, entity_columns)
        for entity_column, hidden in entity_keys.items():
            result = result.withColumn(hidden, F.col(entity_column))
        row_entity = (
            _capture_row_entity(result, frame.columns)
            if not entity_columns and _table_has_perturb(table_name, frame.columns, decisions)
            else {}
        )
        for hidden, expression in row_entity.items():
            result = result.withColumn(hidden, expression)
        ordered_names = sorted(
            frame.columns,
            key=lambda name: _is_drop(decisions.get(f"{table_name}.{name}")),
        )
        for name in ordered_names:
            reference = f"{table_name}.{name}"
            current_column = columns.get(name)
            current_semantic = current_column.semantic if current_column is not None else None
            effective_semantic = domain_semantics.get(domains[reference], current_semantic)
            decision = decisions.get(reference)
            if decision is None and effective_semantic is not None:
                decision = handling.resolve(reference, effective_semantic)
            rule, rule_source = decision or (PiiRule(action=PiiAction.KEEP), "default")
            nested_all = [
                (
                    path.removeprefix(f"{name}."),
                    column.semantic or _rule_semantic(decisions[f"{table_name}.{path}"][0]),
                    domains[f"{table_name}.{path}"],
                    decisions[f"{table_name}.{path}"],
                )
                for nested_table, path, column in _profile_columns(profile)
                if nested_table == table_name and path.startswith(f"{name}.") and f"{table_name}.{path}" in decisions
            ]
            nested_active = [item for item in nested_all if item[3][0].action != PiiAction.KEEP]
            if rule.action == PiiAction.DROP:
                result = result.drop(name)
                count = frame.where(F.col(name).isNotNull()).count()
                reports.append(
                    _report(table_name, name, current_semantic, domains[reference], count, 0, rule, rule_source)
                )
            elif rule.action == PiiAction.SUPPRESS:
                result = result.withColumn(name, F.lit(None).cast(frame.schema[name].dataType))
                count = frame.where(F.col(name).isNotNull()).count()
                reports.append(
                    _report(table_name, name, current_semantic, domains[reference], count, 0, rule, rule_source)
                )
            elif rule.action == PiiAction.GENERALIZE:
                result = result.withColumn(name, _generalize(F.col(name), rule.parameters, frame.schema[name].dataType))
                count = frame.where(F.col(name).isNotNull()).count()
                reports.append(
                    _report(table_name, name, current_semantic, domains[reference], count, 0, rule, rule_source)
                )
            elif rule.action == PiiAction.PERTURB:
                data_type = frame.schema[name].dataType
                stddev = _perturb_stddev(frame, name, data_type)
                result = result.withColumn(
                    name,
                    _perturb_column(
                        F.col(name),
                        data_type=data_type,
                        domain=domains[reference],
                        key=key,
                        rule=rule,
                        stddev=stddev,
                        entity_columns=[entity_keys[item] for item in entity_columns],
                        row_entity_columns=list(row_entity),
                    ),
                )
                count = frame.where(F.col(name).isNotNull()).count()
                reports.append(
                    _report(table_name, name, current_semantic, domains[reference], count, 0, rule, rule_source)
                )
            elif nested_active:
                nested_schema = _nested_output_type(frame.schema[name].dataType, nested_all)
                result = result.withColumn(
                    name,
                    _nested_policy_udf(
                        frame.schema[name].dataType, nested_schema, nested_all, nested_mappings, key, opts
                    )(name),
                )
                for path, semantic, domain, (nested_rule, nested_source) in nested_all:
                    count = _leaf_values(frame, f"{name}.{path}").count()
                    report = _report(
                        table_name,
                        f"{name}.{path}",
                        semantic,
                        domain,
                        count if nested_rule.action != PiiAction.KEEP else 0,
                        collision_counts.get(domain, 0),
                        nested_rule,
                        nested_source,
                    )
                    if nested_rule.action == PiiAction.KEEP:
                        report["skipped_reason"] = (
                            "skipped_low_confidence"
                            if semantic is not None
                            and semantic.confidence < handling.min_confidence
                            and not handling.has_override(f"{table_name}.{name}.{path}", semantic)
                            else "kept_by_policy"
                        )
                    reports.append(report)
            elif nested_all:
                for path, semantic, domain, (nested_rule, nested_source) in nested_all:
                    report = _report(table_name, f"{name}.{path}", semantic, domain, 0, 0, nested_rule, nested_source)
                    report["skipped_reason"] = (
                        "skipped_low_confidence"
                        if semantic is not None
                        and semantic.confidence < handling.min_confidence
                        and not handling.has_override(f"{table_name}.{name}.{path}", semantic)
                        else "kept_by_policy"
                    )
                    reports.append(report)
            elif effective_semantic is not None and rule.action == PiiAction.SUBSTITUTE:
                date_shift_key: str | None = None
                if effective_semantic.pii_class == PiiClass.DATE_OF_BIRTH:
                    date_shift_key = "entity" if entity_columns else "value"
                    transform = _date_shift_udf(domains[reference], key, opts.date_shift_days)
                    result = result.withColumn(
                        name,
                        transform(F.col(name), *(F.col(entity_keys[item]) for item in entity_columns)),
                    )
                elif effective_semantic.pii_class == PiiClass.FREE_TEXT_PII:
                    transform = _substitution_udf(
                        effective_semantic, domains[reference], key, opts, frame.schema[name].dataType
                    )
                    result = result.withColumn(
                        name,
                        F.when(F.col(name).isNull() | (F.col(name).cast("string") == ""), F.col(name)).otherwise(
                            transform(F.col(name))
                        ),
                    )
                else:
                    mapping = mappings[domains[reference]]
                    original = f"__pii_original_{len(reports)}"
                    surrogate = f"__pii_surrogate_{len(reports)}"
                    generic = _is_generic_semantic(effective_semantic)
                    mapping = mapping.select(F.col("original").alias(original), F.col("surrogate").alias(surrogate))
                    source = F.col(name)
                    converted = F.col(surrogate) if generic else F.col(surrogate).cast(frame.schema[name].dataType)
                    join_source = source if generic else source.cast("string")
                    joined = result.join(mapping, join_source == F.col(original), "left")
                    invalid = source.isNotNull() & (source.cast("string") != "") & converted.isNull()
                    if joined.where(invalid).limit(1).count() != 0:
                        raise ValueError(f"surrogate could not be represented for column {reference!r}")
                    result = joined.withColumn(
                        name,
                        F.when(source.isNull() | (source.cast("string") == ""), source).otherwise(converted),
                    ).drop(original, surrogate)
                count = frame.where(F.col(name).isNotNull() & (F.col(name).cast("string") != "")).count()
                report = _report(
                    table_name,
                    name,
                    effective_semantic,
                    domains[reference],
                    count,
                    collision_counts.get(domains[reference], 0),
                    rule,
                    rule_source,
                )
                if date_shift_key is not None:
                    report["date_shift_key"] = date_shift_key
                reports.append(report)
            elif current_semantic is not None:
                reports.append(
                    {
                        **_report(table_name, name, current_semantic, domains[reference], 0, 0, rule, rule_source),
                        "skipped_reason": (
                            "skipped_low_confidence"
                            if current_semantic.confidence < handling.min_confidence
                            and not handling.has_override(reference, current_semantic)
                            else "kept_by_policy"
                        ),
                    }
                )
        output[table_name] = result.drop(*entity_keys.values(), *row_entity)
    return SubstitutionResult(tables=output, report={"columns": reports})


def _decisions(profile: Profile, policy: PiiPolicy, options: SubstitutionOptions) -> dict[str, tuple[PiiRule, str]]:
    decisions: dict[str, tuple[PiiRule, str]] = {}
    references = {f"{table}.{path}" for table, path, _ in _profile_columns(profile)}
    policy.validate_columns(references)
    for table, path, column in _profile_columns(profile):
        reference = f"{table}.{path}"
        if reference in options.generalize or path.rsplit(".", maxsplit=1)[-1] in options.generalize:
            method = options.generalize.get(reference) or options.generalize[path.rsplit(".", maxsplit=1)[-1]]
            decisions[reference] = (PiiRule(action=PiiAction.GENERALIZE, parameters={"method": str(method)}), "column")
        elif reference in policy.columns:
            decisions[reference] = (policy.columns[reference], "column")
        elif column.semantic is None:
            continue
        elif (
            column.semantic.sensitivity not in {Sensitivity.DIRECT_IDENTIFIER, Sensitivity.SECRET}
            and column.semantic.confidence < policy.min_confidence
            and not policy.has_override(reference, column.semantic)
        ):
            decisions[reference] = (PiiRule(action=PiiAction.KEEP), "min_confidence")
        else:
            decisions[reference] = policy.resolve(reference, column.semantic)
    return decisions


def _rule_semantic(rule: PiiRule) -> Semantic | None:
    """Creates value-free semantics for an explicitly substituted undetected column."""
    if rule.action != PiiAction.SUBSTITUTE:
        return None
    pii_class = rule.pii_class or PiiClass.OTHER
    if pii_class == PiiClass.SECRET_TOKEN:
        sensitivity = Sensitivity.SECRET
    elif pii_class == PiiClass.FREE_TEXT_PII:
        sensitivity = Sensitivity.FREE_TEXT
    elif pii_class in {PiiClass.CITY, PiiClass.POSTAL_CODE, PiiClass.DATE_OF_BIRTH}:
        sensitivity = Sensitivity.QUASI_IDENTIFIER
    else:
        sensitivity = Sensitivity.DIRECT_IDENTIFIER
    method = "policy_column" if rule.pii_class is not None else "policy_column_generic"
    return Semantic(
        pii_class=pii_class,
        sensitivity=sensitivity,
        confidence=1.0,
        evidence=[{"method": method, "match_rate": 1.0}],
    )


def _decision_pii_class(column: ColumnProfile, rule: PiiRule) -> PiiClass | None:
    semantic = column.semantic or _rule_semantic(rule)
    return semantic.pii_class if semantic is not None else None


def _profile_columns(profile: Profile) -> Iterator[tuple[str, str, ColumnProfile]]:
    def descendants(table_name: str, path: str, column: ColumnProfile) -> Iterator[tuple[str, str, ColumnProfile]]:
        yield table_name, path, column
        for child in column.children or []:
            yield from descendants(table_name, f"{path}.{child.name}", child)

    for table in profile.tables:
        for column in table.columns:
            yield from descendants(table.name, column.name, column)


def _domain_decisions(
    domains: dict[str, str],
    decisions: dict[str, tuple[PiiRule, str]],
) -> dict[str, tuple[PiiRule, str]]:
    """Makes one join-safe decision per substitution domain."""
    by_domain: dict[str, list[str]] = {}
    for reference, domain in domains.items():
        if reference in decisions:
            by_domain.setdefault(domain, []).append(reference)
    rank = {
        PiiAction.KEEP: 0,
        PiiAction.GENERALIZE: 1,
        PiiAction.PERTURB: 2,
        PiiAction.SUBSTITUTE: 3,
        PiiAction.SUPPRESS: 4,
        PiiAction.DROP: 5,
    }
    resolved = dict(decisions)
    for members in by_domain.values():
        rules = {reference: decisions[reference] for reference in members}
        explicit = [reference for reference, (_, source) in rules.items() if source in {"column", "pii_class"}]
        domain = domains[members[0]]
        domain_members = [reference for reference, member_domain in domains.items() if member_domain == domain]
        dropped = [reference for reference, (rule, _) in rules.items() if rule.action == PiiAction.DROP]
        if dropped and len(dropped) != len(members):
            missing = sorted(set(members) - set(dropped))
            raise ValueError(f"cannot drop only part of an FK-linked column set; also drop {missing}")
        if len(explicit) >= 2 and len({rules[reference][0].model_dump_json() for reference in explicit}) > 1:
            descriptions = ", ".join(f"{reference}={rules[reference][0].action.value}" for reference in sorted(rules))
            raise ValueError(f"conflicting explicit PII rules for linked columns: {descriptions}")
        winner_reference = max(members, key=lambda reference: rank[rules[reference][0].action])
        winner_rule = rules[winner_reference][0]
        winner = (winner_rule, f"fk_group:{winner_reference}") if len(domain_members) > 1 else rules[winner_reference]
        for reference, member_domain in domains.items():
            if member_domain == domain:
                resolved[reference] = winner
    return resolved


def _entity_columns(profile: Profile, table_name: str) -> list[str]:
    tables = {table.name: table for table in profile.tables}
    table = tables[table_name]
    if table.primary_key is not None:
        return table.primary_key.columns
    for relationship in profile.relationships or []:
        parent = tables[relationship.parent_table]
        if (
            relationship.child_table == table_name
            and parent.primary_key is not None
            and relationship.parent_columns == parent.primary_key.columns
        ):
            return relationship.child_columns
    return []


def _capture_entity_keys(frame: DataFrame, entity_columns: list[str]) -> dict[str, str]:
    used = {name.lower() for name in frame.columns}
    captured: dict[str, str] = {}
    for index, column in enumerate(entity_columns):
        hidden = f"__pii_entity_{index}"
        while hidden.lower() in used:
            hidden = f"_{hidden}"
        captured[column] = hidden
        used.add(hidden.lower())
    return captured


def _is_drop(decision: tuple[PiiRule, str] | None) -> bool:
    return decision is not None and decision[0].action == PiiAction.DROP


def _build_mapping(
    tables: dict[str, DataFrame],
    members: list[str],
    semantic: Semantic,
    domain: str,
    key: bytes,
    options: SubstitutionOptions,
) -> tuple[DataFrame, int]:
    generic = _is_generic_semantic(semantic)
    data_types = [
        _leaf_data_type(tables[reference.split(".", maxsplit=1)[0]], reference.split(".", maxsplit=1)[1])
        for reference in members
    ]
    data_type = data_types[0]
    if generic and any(kind != data_type for kind in data_types[1:]):
        raise ValueError(f"generic substitution domain {domain!r} has incompatible Spark types")
    if generic:
        _validate_generic_type(data_type, members[0])
    values: DataFrame | None = None
    for reference in members:
        table, column = reference.split(".", maxsplit=1)
        selected = _leaf_values(tables[table], column, cast_string=not generic)
        values = selected if values is None else values.union(selected)
    if values is None:
        raise ValueError(f"domain {domain!r} has no columns")
    mapping = values.distinct().withColumn("counter", F.lit(0))
    if generic:
        _validate_generic_values(mapping, data_type, members[0])
    else:
        _check_surrogate_capacity(mapping, semantic, members)
    make = F.udf(
        lambda value, counter: _mapping_value(semantic, value, data_type, domain, key, options, counter),
        data_type if generic else T.StringType(),
    )
    candidate = mapping.withColumn("surrogate", make("original", "counter"))
    duplicates = candidate.groupBy("surrogate").count().where(F.col("count") > 1).select("surrogate")
    originals = values.select(F.col("original").alias("existing_original")).distinct()
    survivors = candidate.join(originals, F.col("surrogate") == F.col("existing_original"), "inner").select("surrogate")
    if duplicates.union(survivors).limit(1).count() == 0:
        return candidate.select("original", "surrogate"), 0
    return _resolve_collisions(values, semantic, data_type, domain, key, options, members[0])


def _resolve_collisions(
    values: DataFrame,
    semantic: Semantic,
    data_type: T.DataType,
    domain: str,
    key: bytes,
    options: SubstitutionOptions,
    column: str,
) -> tuple[DataFrame, int]:
    originals = values.distinct().localCheckpoint(eager=True)
    generic = _is_generic_semantic(semantic)
    make = F.udf(
        lambda value, counter: _mapping_value(semantic, value, data_type, domain, key, options, counter),
        data_type if generic else T.StringType(),
    )
    initial = originals.withColumn("counter", F.lit(0)).withColumn("surrogate", make("original", "counter"))
    duplicate_surrogates = initial.groupBy("surrogate").count().where(F.col("count") > 1).select("surrogate")
    duplicate_originals = initial.join(duplicate_surrogates, "surrogate", "inner").select("original")
    survivor_originals = initial.join(
        originals.select(F.col("original").alias("existing")), F.col("surrogate") == F.col("existing"), "inner"
    ).select("original")
    rehash = duplicate_originals.union(survivor_originals).distinct().localCheckpoint(eager=True)
    collisions = rehash.count()
    assigned = initial.join(rehash, "original", "left_anti").select("original", "surrogate").localCheckpoint(eager=True)
    unresolved = rehash.withColumn("counter", F.lit(1)).select("original", "counter").localCheckpoint(eager=True)
    attempts = min(100_000, max(256, 2 ** min(options.surrogate_bits, 16)))
    for _ in range(attempts):
        if unresolved.limit(1).count() == 0:
            return assigned, collisions
        proposals = unresolved.withColumn("surrogate", make("original", "counter"))
        proposals = proposals.join(
            originals.select(F.col("original").alias("existing")),
            F.col("surrogate") == F.col("existing"),
            "left_anti",
        ).join(assigned.select("surrogate"), "surrogate", "left_anti")
        winners = (
            proposals.withColumn("rank", F.row_number().over(Window.partitionBy("surrogate").orderBy("original")))
            .where(F.col("rank") == 1)
            .select("original", "surrogate")
            .localCheckpoint(eager=True)
        )
        next_unresolved = (
            unresolved.join(winners.select("original"), "original", "left_anti")
            .withColumn("counter", F.col("counter") + 1)
            .localCheckpoint(eager=True)
        )
        next_assigned = assigned.union(winners).localCheckpoint(eager=True)
        unresolved.unpersist()
        assigned.unpersist()
        unresolved, assigned = next_unresolved, next_assigned
    raise ValueError(f"surrogate space is too small to preserve distinct values for column {column!r}")


def _leaf_values(frame: DataFrame, path: str, *, cast_string: bool = True) -> DataFrame:
    parts = path.split(".")
    data_type = frame.schema[parts[0]].dataType
    values = frame.select(F.col(parts[0]).alias("__value"))
    for part in parts[1:]:
        if isinstance(data_type, T.StructType):
            data_type = data_type[part].dataType
            values = values.select(F.col("__value").getField(part).alias("__value"))
        elif isinstance(data_type, T.ArrayType) and part == "element":
            data_type = data_type.elementType
            values = values.select(F.explode_outer("__value").alias("__value"))
        elif isinstance(data_type, T.MapType) and part in {"key", "value"}:
            data_type = data_type.keyType if part == "key" else data_type.valueType
            values = values.select(F.explode_outer(F.map_entries("__value")).alias("__entry")).select(
                F.col("__entry").getField(part).alias("__value")
            )
        else:
            raise ValueError(f"invalid nested column path {path!r}")
    original = F.col("__value").cast("string") if cast_string else F.col("__value")
    selected = values.select(original.alias("original")).where(F.col("original").isNotNull())
    if cast_string or isinstance(data_type, (T.StringType, T.BinaryType)):
        selected = selected.where(F.length("original") > 0)
    return selected


def _leaf_data_type(frame: DataFrame, path: str) -> T.DataType:
    data_type = frame.schema[path.split(".")[0]].dataType
    for part in path.split(".")[1:]:
        if isinstance(data_type, T.StructType):
            data_type = data_type[part].dataType
        elif isinstance(data_type, T.ArrayType) and part == "element":
            data_type = data_type.elementType
        elif isinstance(data_type, T.MapType) and part in {"key", "value"}:
            data_type = data_type.keyType if part == "key" else data_type.valueType
        else:
            raise ValueError(f"invalid nested column path {path!r}")
    return data_type


def _validate_generic_type(data_type: T.DataType, reference: str) -> None:
    if isinstance(data_type, T.BooleanType):
        raise ValueError(f"generic substitute is not supported for boolean column {reference!r}; use suppress instead")
    supported = (
        T.ByteType,
        T.ShortType,
        T.IntegerType,
        T.LongType,
        T.FloatType,
        T.DoubleType,
        T.DecimalType,
        T.DateType,
        T.TimestampType,
        T.StringType,
        T.BinaryType,
    )
    timestamp_ntz = getattr(T, "TimestampNTZType", None)
    if not isinstance(data_type, supported) and (timestamp_ntz is None or not isinstance(data_type, timestamp_ntz)):
        raise ValueError(
            f"generic substitute is not supported for Spark type {data_type.simpleString()!r} on {reference!r}"
        )


def _validate_generic_values(mapping: DataFrame, data_type: T.DataType, reference: str) -> None:
    if isinstance(data_type, (T.FloatType, T.DoubleType)):
        non_finite = F.isnan("original") | (F.abs("original") == F.lit(float("inf")))
        if mapping.where(non_finite).limit(1).count() != 0:
            raise ValueError(f"generic float substitution requires finite values for column {reference!r}")
    space: int | None = None
    if isinstance(data_type, T.ByteType):
        space = 2**8
    elif isinstance(data_type, T.ShortType):
        space = 2**16
    elif isinstance(data_type, T.IntegerType):
        space = 2**32
    elif isinstance(data_type, T.DecimalType) and data_type.precision <= 9:
        space = 2 * 10**data_type.precision - 1
    if space is not None:
        count = mapping.count()
        if count > space - count:
            raise ValueError(f"surrogate space is too small to preserve distinct values for column {reference!r}")


def _check_surrogate_capacity(mapping: DataFrame, semantic: Semantic, members: list[str]) -> None:
    if semantic.pii_class not in {PiiClass.ACCOUNT_ID, PiiClass.SECRET_TOKEN}:
        return
    stats = mapping.agg(
        F.count("original").alias("count"),
        F.min(F.length("original")).alias("min_length"),
        F.max(F.length("original")).alias("max_length"),
        F.min(F.col("original").rlike(r"^\d+$").cast("int")).alias("all_digits"),
    ).first()
    if stats is None or not stats.all_digits or stats.min_length != stats.max_length:
        return
    count = int(stats["count"])
    available = 10 ** int(stats["min_length"]) - count
    if count > available:
        raise ValueError(f"surrogate space is too small to preserve distinct values for column {members[0]!r}")


def _mapping_surrogate(
    kind: PiiClass,
    value: str,
    domain: str,
    key: bytes,
    options: SubstitutionOptions,
    counter: int,
) -> str:
    digest = _digest(key, domain, value, options.surrogate_bits, counter)
    if kind == PiiClass.DATE_OF_BIRTH:
        original = date.fromisoformat(value)
        return (original + timedelta(days=_date_offset(digest, options.date_shift_days))).isoformat()
    return _surrogate(kind, value, domain, key, options, counter)


def _is_generic_semantic(semantic: Semantic) -> bool:
    return any(item.method == "policy_column_generic" for item in semantic.evidence)


def _mapping_value(  # noqa: PLR0911, PLR0912
    semantic: Semantic,
    value: object,
    data_type: T.DataType,
    domain: str,
    key: bytes,
    options: SubstitutionOptions,
    counter: int,
) -> object:
    text = str(value)
    if not _is_generic_semantic(semantic):
        return _mapping_surrogate(semantic.pii_class, text, domain, key, options, counter)
    digest = _digest(key, domain, text, options.surrogate_bits, counter)
    number = int.from_bytes(digest, "big")
    integer_ranges: tuple[tuple[type[T.DataType], int, int], ...] = (
        (T.ByteType, -(2**7), 2**7 - 1),
        (T.ShortType, -(2**15), 2**15 - 1),
        (T.IntegerType, -(2**31), 2**31 - 1),
        (T.LongType, -(2**63), 2**63 - 1),
    )
    for kind, minimum, maximum in integer_ranges:
        if isinstance(data_type, kind):
            return minimum + number % (maximum - minimum + 1)
    if isinstance(data_type, (T.FloatType, T.DoubleType)):
        if not isinstance(value, (int, float)):
            raise ValueError(f"expected numeric value for {data_type.simpleString()}")
        original = float(value)
        magnitude = max(abs(original), 1.0)
        factor = 0.5 + (number % 1_000_003) / 1_000_003
        return math.copysign(magnitude * factor, original or 1.0)
    if isinstance(data_type, T.DecimalType):
        limit = 10**data_type.precision
        unscaled = number % (limit * 2 - 1) - (limit - 1)
        return Decimal(unscaled).scaleb(-data_type.scale)
    if isinstance(data_type, T.DateType):
        if not isinstance(value, date):
            raise ValueError("expected date value")
        return value + timedelta(days=_date_offset(digest, options.date_shift_days))
    timestamp_ntz = getattr(T, "TimestampNTZType", None)
    if isinstance(data_type, T.TimestampType) or (timestamp_ntz is not None and isinstance(data_type, timestamp_ntz)):
        if not isinstance(value, datetime):
            raise ValueError("expected timestamp value")
        maximum = max(1, options.date_shift_days * 24 * 60 * 60)
        return value + timedelta(seconds=_date_offset(digest, maximum))
    if isinstance(data_type, T.StringType):
        return _surrogate(PiiClass.OTHER, text, domain, key, options, counter)
    if isinstance(data_type, T.BinaryType):
        if not isinstance(value, (bytes, bytearray)):
            raise ValueError("expected binary value")
        raw = bytes(value)
        return (digest * ((len(raw) + len(digest) - 1) // len(digest)))[: len(raw)]
    raise ValueError(f"generic substitute is not supported for Spark type {data_type.simpleString()!r}")


def _domains(profile: Profile, options: SubstitutionOptions) -> dict[str, str]:
    references = {f"{table}.{path}" for table, path, _ in _profile_columns(profile)}
    parent = {reference: reference for reference in references}

    def find(reference: str) -> str:
        while parent[reference] != reference:
            parent[reference] = parent[parent[reference]]
            reference = parent[reference]
        return reference

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            first, second = sorted((left_root, right_root))
            parent[second] = first

    for relationship in profile.relationships or []:
        for child, parent_column in zip(relationship.child_columns, relationship.parent_columns, strict=True):
            union(f"{relationship.child_table}.{child}", f"{relationship.parent_table}.{parent_column}")
    explicit: dict[str, str] = {}
    for name, value in options.domains.items():
        if isinstance(value, str):
            explicit[name] = value
        else:
            for reference in value:
                explicit[reference] = name
    groups: dict[str, list[str]] = {}
    for reference, domain in explicit.items():
        if reference not in references:
            raise ValueError(f"unknown domain column {reference!r}")
        groups.setdefault(domain, []).append(reference)
    for members in groups.values():
        for member in members[1:]:
            union(members[0], member)
    root_labels: dict[str, str] = {}
    for reference, label in explicit.items():
        root_labels.setdefault(find(reference), label)
    return {reference: root_labels.get(find(reference), find(reference)) for reference in references}


def _substitution_udf(
    semantic: Semantic, domain: str, key: bytes, options: SubstitutionOptions, data_type: T.DataType
) -> Callable[..., Column]:
    def replace(value: object) -> object:
        if semantic.pii_class == PiiClass.FREE_TEXT_PII:
            return _replace_spans(str(value), domain, key, options)
        if semantic.pii_class == PiiClass.DATE_OF_BIRTH and isinstance(value, date):
            digest = _digest(key, domain, value.isoformat(), options.surrogate_bits)
            return value + timedelta(days=_date_offset(digest, options.date_shift_days))
        return _surrogate(semantic.pii_class, str(value), domain, key, options)

    return F.udf(replace, data_type, useArrow=False)


def _date_shift_udf(domain: str, key: bytes, date_shift_days: int) -> Callable[..., Column]:
    def replace(value: date | None, *entity: object) -> date | None:
        if value is None:
            return None
        identity = (
            "\u001f".join(str(item) for item in entity)
            if entity and all(item is not None for item in entity)
            else value.isoformat()
        )
        digest = _digest(key, f"{domain}:entity", identity)
        return value + timedelta(days=_date_offset(digest, date_shift_days))

    return F.udf(replace, T.DateType(), useArrow=False)


def _nested_output_type(
    data_type: T.DataType,
    specifications: list[tuple[str, Semantic | None, str, tuple[PiiRule, str]]],
) -> T.DataType:
    rules = {tuple(path.split(".")): decision[0] for path, _, _, decision in specifications}

    def transform(kind: T.DataType, path: tuple[str, ...]) -> T.DataType:
        if isinstance(kind, T.StructType):
            return T.StructType(
                [
                    T.StructField(
                        field.name, transform(field.dataType, (*path, field.name)), field.nullable, field.metadata
                    )
                    for field in kind.fields
                    if rules.get((*path, field.name), PiiRule(action=PiiAction.KEEP)).action != PiiAction.DROP
                ]
            )
        if isinstance(kind, T.ArrayType):
            return T.ArrayType(transform(kind.elementType, (*path, "element")), kind.containsNull)
        if isinstance(kind, T.MapType):
            if any(
                rules.get((*path, part), PiiRule(action=PiiAction.KEEP)).action == PiiAction.DROP
                for part in ("key", "value")
            ):
                raise ValueError("drop is not supported for a map key or value; drop the containing column")
            return T.MapType(
                transform(kind.keyType, (*path, "key")),
                transform(kind.valueType, (*path, "value")),
                kind.valueContainsNull,
            )
        return kind

    return transform(data_type, ())


def _nested_policy_udf(
    data_type: T.DataType,
    output_type: T.DataType,
    specifications: list[tuple[str, Semantic | None, str, tuple[PiiRule, str]]],
    mappings: dict[str, Broadcast[dict[str, object]]],
    key: bytes,
    options: SubstitutionOptions,
) -> Callable[..., Column]:
    semantics = {
        tuple(path.split(".")): (semantic, domain, decision[0]) for path, semantic, domain, decision in specifications
    }

    def transform(value: object, kind: T.DataType, path: tuple[str, ...]) -> object:  # noqa: PLR0911, PLR0912
        if value is None:
            return None
        specification = semantics.get(path)
        if specification is not None:
            semantic, domain, rule = specification
            if rule.action == PiiAction.KEEP:
                return value
            if rule.action == PiiAction.SUPPRESS:
                return None
            if rule.action == PiiAction.GENERALIZE:
                return _generalize_nested(value, rule.parameters)
            if rule.action == PiiAction.PERTURB:
                raise ValueError(
                    f"perturb is not supported for nested path {'.'.join(path)!r}; "
                    "perturb only applies to top-level numeric and temporal columns"
                )
            if semantic is None:
                raise ValueError(f"substitute rule for nested path {'.'.join(path)!r} has no PII class")
            if str(value) == "":
                return value
            if semantic.pii_class == PiiClass.FREE_TEXT_PII:
                return _replace_spans(str(value), domain, key, options)
            mapping = mappings.get(domain)
            if mapping is not None:
                surrogate = mapping.value.get(str(value))
                if surrogate is None:
                    raise ValueError(f"nested value is absent from resolved domain {domain!r}")
                return _coerce_nested_value(surrogate, kind)
            surrogate = _mapping_value(semantic, value, kind, domain, key, options, 0)
            return _coerce_nested_value(surrogate, kind)
        if isinstance(kind, T.StructType):
            source = value.asDict(recursive=False) if hasattr(value, "asDict") else value
            return {
                field.name: transform(source[field.name], field.dataType, (*path, field.name))
                for field in kind.fields
                if semantics.get((*path, field.name), (None, None, PiiRule(action=PiiAction.KEEP)))[2].action
                != PiiAction.DROP
            }
        if isinstance(kind, T.ArrayType):
            return (
                [transform(item, kind.elementType, (*path, "element")) for item in value]
                if isinstance(value, list | tuple)
                else value
            )
        if isinstance(kind, T.MapType):
            if isinstance(value, dict):
                transformed = {
                    transform(item_key, kind.keyType, (*path, "key")): transform(
                        item_value, kind.valueType, (*path, "value")
                    )
                    for item_key, item_value in value.items()
                }
                if len(transformed) != len(value):
                    raise ValueError("nested map substitution changed the entry count")
                return transformed
            return value
        return value

    def replace(value: object) -> object:
        return None if value is None else transform(value, data_type, ())

    return F.udf(replace, output_type, useArrow=False)


def _generalize_nested(value: object, parameters: dict[str, str | int | float | bool]) -> object:
    if "prefix_length" in parameters:
        return str(value)[: int(parameters["prefix_length"])]
    granularity = str(parameters.get("date_granularity", parameters.get("method", "")))
    if granularity in {"month", "date_month"} and isinstance(value, date):
        return value.replace(day=1)
    if granularity in {"year", "date_year"} and isinstance(value, date):
        return value.replace(month=1, day=1)
    if "bucket_width" in parameters:
        width = float(parameters["bucket_width"])
        if width <= 0:
            raise ValueError("numeric bucket_width must be positive")
        result = math.floor(float(str(value)) / width) * width
        return int(result) if isinstance(value, int) else result
    raise ValueError(f"unknown nested generalization parameters {parameters!r}")


def _coerce_nested_value(value: object, data_type: T.DataType) -> object:
    if not isinstance(value, str):
        return value
    if isinstance(data_type, (T.ByteType, T.ShortType, T.IntegerType, T.LongType)):
        return int(value)
    if isinstance(data_type, (T.FloatType, T.DoubleType)):
        return float(value)
    if isinstance(data_type, T.DateType):
        return date.fromisoformat(value)
    return value


def _digest(key: bytes, domain: str, value: str, bits: int = 256, counter: int = 0) -> bytes:
    message = f"{domain}\u001f{value}\u001f{counter}".encode()
    digest = bytearray(hmac.new(key, message, hashlib.sha256).digest())
    size = (bits + 7) // 8
    digest = digest[:size]
    if bits % 8:
        digest[-1] &= (1 << (bits % 8)) - 1
    return bytes(digest)


def _date_offset(digest: bytes, maximum: int) -> int:
    """Returns a deterministic non-zero offset within the configured bound."""
    if maximum == 0:
        return 0
    position = int.from_bytes(digest, "big") % (maximum * 2)
    return position - maximum if position < maximum else position - maximum + 1


def _surrogate(  # noqa: PLR0911
    kind: PiiClass, value: str, domain: str, key: bytes, options: SubstitutionOptions, counter: int = 0
) -> str:
    digest = _digest(key, domain, value, options.surrogate_bits, counter)
    number = int.from_bytes(digest, "big")
    token = digest.hex()
    if kind == PiiClass.EMAIL:
        domain_part = value.rpartition("@")[2] if options.keep_email_domain and "@" in value else "masked.invalid"
        if len(value) - len(domain_part) - 1 < 1:
            domain_part = "x.co"
        local_length = max(1, len(value) - len(domain_part) - 1)
        local = ("u" + token * local_length)[:local_length]
        return f"{local}@{domain_part}"
    if kind == PiiClass.PHONE:
        digits = re.sub(r"\D", "", value)
        generated = str(number).zfill(len(digits))[-len(digits) :]
        country_length = 0
        if value.startswith("+"):
            country_length = max(1, len(digits) - 10)
        area_end = min(len(digits), country_length + 3)
        generated = digits[:area_end] + generated[area_end:]
        nanp_offset = len(generated) - 10
        if len(generated) >= 10 and (len(generated) == 10 or generated[:nanp_offset] == "1"):
            generated = (
                generated[: nanp_offset + 3]
                + str(int(generated[nanp_offset + 3]) % 8 + 2)
                + generated[nanp_offset + 4 :]
            )
        return _apply_digit_format(value, generated)
    if kind == PiiClass.CREDIT_CARD:
        length = len(re.sub(r"\D", "", value))
        body = str(number).zfill(length - 1)[-(length - 1) :]
        digits = body + str(_luhn_check_digit(body))
        return _apply_digit_format(value, digits)
    if kind == PiiClass.NATIONAL_ID:
        digits = f"{number % 799 + 100:03d}{(number // 799) % 99 + 1:02d}{(number // 79101) % 9999 + 1:04d}"
        if digits[:3] == "666":
            digits = "665" + digits[3:]
        return _apply_digit_format(value, digits)
    if kind == PiiClass.IBAN:
        compact = re.sub(r"\s", "", value).upper()
        country, length = compact[:2], len(compact)
        body_length = length - 4
        body = "".join(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"[(number >> (index * 5)) % 36] for index in range(body_length)
        )
        rearranged = body + country + "00"
        numeric = "".join(str(ord(char) - 55) if char.isalpha() else char for char in rearranged)
        result = f"{country}{98 - int(numeric) % 97:02d}{body}"
        return " ".join(result[index : index + 4] for index in range(0, len(result), 4)) if " " in value else result
    if kind == PiiClass.IP_ADDRESS:
        original = ipaddress.ip_address(value)
        return str(ipaddress.ip_address(number % (2**original.max_prefixlen)))
    if kind == PiiClass.PERSON_NAME:
        syllables = ("ba", "de", "fi", "ko", "lu", "mi", "na", "ra", "so", "ve")
        parts = [syllables[(number >> (index * 4)) % len(syllables)] for index in range(max(2, len(value) // 3))]
        return "".join(parts)[: max(2, len(value))].title()
    if kind in {PiiClass.ACCOUNT_ID, PiiClass.SECRET_TOKEN} and value.isdigit():
        return str(number).zfill(len(value))[-len(value) :]
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(alphabet[(number >> (index * 6)) % len(alphabet)] for index in range(max(1, len(value))))


def _apply_digit_format(template: str, digits: str) -> str:
    iterator = iter(digits)
    return "".join(next(iterator) if char.isdigit() else char for char in template)


def _luhn_check_digit(body: str) -> int:
    total = 0
    for index, digit in enumerate(map(int, reversed(body))):
        value = digit * 2 if index % 2 == 0 else digit
        total += value - 9 if value > 9 else value
    return (-total) % 10


def _replace_spans(text: str, domain: str, key: bytes, options: SubstitutionOptions) -> str:
    patterns = (
        (PiiClass.EMAIL, re.compile(r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9-]+(?:\.[A-Z0-9-]+)+", re.I)),
        (PiiClass.NATIONAL_ID, re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}(?!\d)")),
        (PiiClass.PHONE, re.compile(r"\+[1-9]\d{7,14}")),
        (PiiClass.IBAN, re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
        (PiiClass.IP_ADDRESS, re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    )
    result = text
    for kind, pattern in patterns:

        def replace(match: re.Match[str], pii_class: PiiClass = kind) -> str:
            return _safe_span_surrogate(pii_class, match.group(), domain, key, options)

        result = pattern.sub(
            replace,
            result,
        )
    card = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
    result = card.sub(
        lambda match: (
            _surrogate(PiiClass.CREDIT_CARD, match.group(), domain, key, options)
            if _is_luhn(match.group())
            else match.group()
        ),
        result,
    )
    return result


def _safe_span_surrogate(kind: PiiClass, value: str, domain: str, key: bytes, options: SubstitutionOptions) -> str:
    if kind == PiiClass.IP_ADDRESS:
        try:
            ipaddress.ip_address(value)
        except ValueError:
            return value
    if kind == PiiClass.IBAN:
        compact = re.sub(r"\s", "", value).upper()
        rearranged = compact[4:] + compact[:4]
        numeric = "".join(str(ord(char) - 55) if char.isalpha() else char for char in rearranged)
        if not numeric.isdigit() or int(numeric) % 97 != 1:
            return value
    return _surrogate(kind, value, domain, key, options)


def _is_luhn(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(map(int, reversed(digits))):
        number = digit * 2 if index % 2 else digit
        total += number - 9 if number > 9 else number
    return total % 10 == 0


def _generalize(column: Column, parameters: dict[str, str | int | float | bool], data_type: T.DataType) -> Column:
    method = str(parameters.get("method", ""))
    if "prefix_length" in parameters or method == "postal_code_prefix":
        length = int(parameters.get("prefix_length", 3))
        return F.substring(column.cast("string"), 1, length)
    if method == "postal_prefix":
        return F.concat(F.substring(column.cast("string"), 1, 3), F.lit("**"))
    granularity = str(parameters.get("date_granularity", method))
    if granularity in {"date_month", "month"}:
        return F.date_trunc("month", column) if isinstance(data_type, T.TimestampType) else F.trunc(column, "month")
    if granularity in {"date_year", "year"}:
        return F.date_trunc("year", column) if isinstance(data_type, T.TimestampType) else F.trunc(column, "year")
    if "bucket_width" in parameters or method in {"numeric_bucket", "bucket"}:
        width = float(parameters.get("bucket_width", 0))
        if width <= 0:
            raise ValueError("numeric bucket_width must be positive")
        return (F.floor(column / F.lit(width)) * F.lit(width)).cast(data_type)
    raise ValueError(f"unknown generalization {method!r}")


def _report(
    table: str,
    column: str,
    semantic: Semantic | None,
    domain: str,
    count: int,
    collisions: int,
    rule: PiiRule,
    rule_source: str,
) -> dict[str, Any]:
    source, separator, winner = rule_source.partition(":")
    parameters = dict(rule.parameters)
    if rule.action == PiiAction.PERTURB:
        parameters.setdefault("mode", rule.perturb_mode())
        if rule.perturb_mode() == "noise":
            parameters.setdefault("noise_ratio", rule.perturb_noise_ratio())
    report: dict[str, Any] = {
        "table": table,
        "column": column,
        "class": semantic.pii_class.value if semantic else None,
        "domain": domain,
        "rule_source": source,
        "action": rule.action.value,
        "parameters": parameters,
        "values_affected": count,
        "values_substituted": count if rule.action == PiiAction.SUBSTITUTE else 0,
        "values_perturbed": count if rule.action == PiiAction.PERTURB else 0,
        "collisions_resolved": collisions,
        "generalizations": (
            [rule.parameters["method"]]
            if rule.action == PiiAction.GENERALIZE and set(rule.parameters) == {"method"}
            else ([rule.parameters] if rule.action == PiiAction.GENERALIZE else [])
        ),
    }
    if rule.action == PiiAction.PERTURB:
        report["mode"] = rule.perturb_mode()
        report["expected_correlation_attenuation"] = expected_correlation_attenuation(rule)
        report["equality_preserved"] = False
    if separator:
        report["winning_column"] = winner
    return report


def _table_has_perturb(table_name: str, columns: list[str], decisions: dict[str, tuple[PiiRule, str]]) -> bool:
    return any(
        decisions.get(f"{table_name}.{name}", (PiiRule(action=PiiAction.KEEP), "default"))[0].action
        == PiiAction.PERTURB
        for name in columns
    )


def _capture_row_entity(frame: DataFrame, columns: list[str]) -> dict[str, Column]:
    """Builds deterministic row-identity columns when no primary key is available."""
    used = {name.lower() for name in frame.columns}
    hash_name = "__pii_row_hash"
    while hash_name.lower() in used:
        hash_name = f"_{hash_name}"
    used.add(hash_name.lower())
    occ_name = "__pii_row_occ"
    while occ_name.lower() in used:
        occ_name = f"_{occ_name}"
    parts = [F.coalesce(F.col(name).cast("string"), F.lit("\0")) for name in columns]
    row_hash = F.xxhash64(*parts) if parts else F.lit(0)
    occurrence = F.row_number().over(Window.partitionBy(row_hash).orderBy(F.monotonically_increasing_id()))
    return {hash_name: row_hash, occ_name: occurrence}


def _key_column_names(profile: Profile, table_name: str) -> set[str]:
    table = next(item for item in profile.tables if item.name == table_name)
    names = set(table.primary_key.columns if table.primary_key is not None else [])
    for key in table.unique_keys or []:
        names.update(key.columns)
    return names


def _validate_perturb_decisions(
    profile: Profile,
    tables: dict[str, DataFrame],
    domains: dict[str, str],
    decisions: dict[str, tuple[PiiRule, str]],
) -> None:
    key_references = {
        f"{table.name}.{column}" for table in profile.tables for column in _key_column_names(profile, table.name)
    }
    domain_members: dict[str, list[str]] = {}
    for reference, domain in domains.items():
        domain_members.setdefault(domain, []).append(reference)
    for reference, (rule, _) in decisions.items():
        if rule.action != PiiAction.PERTURB:
            continue
        table_name, path = reference.split(".", maxsplit=1)
        if "." in path:
            raise ValueError(
                f"perturb is not supported for nested column {reference!r}; "
                "perturb only applies to top-level numeric and temporal columns"
            )
        frame = tables[table_name]
        if path not in frame.columns:
            raise ValueError(f"perturb column {reference!r} is absent from the input table")
        column = next(col for t, p, col in _profile_columns(profile) if t == table_name and p == path)
        semantic = column.semantic
        if semantic is not None and semantic.sensitivity in {
            Sensitivity.DIRECT_IDENTIFIER,
            Sensitivity.SECRET,
            Sensitivity.FREE_TEXT,
        }:
            raise ValueError(
                f"perturb is not allowed for {semantic.sensitivity.value} column {reference!r}; "
                "use it only on quasi_identifier columns or explicit column rules for undetected columns"
            )
        members = domain_members[domains[reference]]
        if len(members) > 1:
            raise ValueError(
                f"perturb is not allowed on FK-linked column {reference!r}; "
                "perturb cannot be applied to any column in an FK-linked substitution domain"
            )
        if reference in key_references:
            raise ValueError(
                f"perturb is not allowed on key column {reference!r}; "
                "perturb cannot be applied to primary keys or unique keys"
            )
        data_type = frame.schema[path].dataType
        if not _is_perturbable_type(data_type):
            raise ValueError(
                f"perturb is not supported for Spark type {data_type.simpleString()!r} on {reference!r}; "
                "allowed types are numeric, date, timestamp, and timestamp_ntz"
            )


def _is_perturbable_type(data_type: T.DataType) -> bool:
    if isinstance(
        data_type,
        (
            T.ByteType,
            T.ShortType,
            T.IntegerType,
            T.LongType,
            T.FloatType,
            T.DoubleType,
            T.DecimalType,
            T.DateType,
            T.TimestampType,
        ),
    ):
        return True
    timestamp_ntz = getattr(T, "TimestampNTZType", None)
    return timestamp_ntz is not None and isinstance(data_type, timestamp_ntz)


def _is_exact_numeric_type(data_type: T.DataType) -> bool:
    return isinstance(data_type, (T.ByteType, T.ShortType, T.IntegerType, T.LongType, T.DecimalType))


def _perturb_numeric_expression(column: Column, data_type: T.DataType) -> Column:
    if isinstance(data_type, T.DateType):
        return F.datediff(column, F.lit(date(1970, 1, 1)).cast("date")).cast("double")
    if isinstance(data_type, T.TimestampType):
        return column.cast("double")
    timestamp_ntz = getattr(T, "TimestampNTZType", None)
    if timestamp_ntz is not None and isinstance(data_type, timestamp_ntz):
        # TIMESTAMP_NTZ cannot cast directly to double; go through timestamp first.
        return column.cast("timestamp").cast("double")
    return column.cast("double")


def _perturb_stddev(frame: DataFrame, name: str, data_type: T.DataType) -> float:
    """Population stddev via Spark aggregates (double; scales sigma only, so approx is fine)."""
    numeric = _perturb_numeric_expression(F.col(name), data_type)
    if isinstance(data_type, (T.FloatType, T.DoubleType)):
        numeric = F.when(F.isnan(name) | (F.abs(F.col(name)) == F.lit(float("inf"))), F.lit(None)).otherwise(numeric)
    stats = frame.select(F.stddev_pop(numeric).alias("std")).first()
    if stats is None or stats["std"] is None:
        return 0.0
    return float(stats["std"])


def _unit_from_digest(digest: bytes, offset: int = 0) -> float:
    """Maps 8 digest bytes to a uniform open unit interval (0, 1)."""
    chunk = digest[offset : offset + 8]
    if len(chunk) < 8:
        chunk = (chunk + digest)[:8]
    value = int.from_bytes(chunk, "big") % (2**53 - 1) + 1
    return value / 2**53


def _gaussian_from_digest(digest: bytes) -> float:
    """Returns a deterministic standard-normal sample via Box-Muller."""
    u1 = _unit_from_digest(digest, 0)
    u2 = _unit_from_digest(digest, 8)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _digest_expand(seed: bytes, nbytes: int) -> bytes:
    """Deterministically expands a seed into nbytes via SHA-256 counter mode."""
    out = bytearray()
    counter = 0
    while len(out) < nbytes:
        out.extend(hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
        counter += 1
    return bytes(out[:nbytes])


def _exact_numeric_quantum(data_type: T.DataType) -> Decimal:
    """Returns the column quantum for exact integer/decimal types."""
    if isinstance(data_type, T.DecimalType):
        return Decimal(1).scaleb(-data_type.scale)
    return Decimal(1)


def _gaussian_noise_decimal(digest: bytes, sigma: Decimal, *, quantum: Decimal) -> Decimal:
    """Approx N(0, sigma^2); digits below float precision are HMAC-randomized down to quantum.

    Float Box-Muller only yields ~53 bits. For large sigma that leaves low-order digits of
    ``value + noise`` unchanged (a leak). Keep the float gaussian for the high significand,
    then draw the unresolved lower digits uniformly from expanded HMAC bytes.
    """
    if sigma == 0 or quantum <= 0:
        return Decimal(0)
    z = _gaussian_from_digest(digest)
    with localcontext() as context:
        context.prec = max(50, abs(sigma).adjusted() - quantum.adjusted() + 40)
        context.rounding = ROUND_HALF_EVEN
        coarse = Decimal(str(z)) * sigma
        ref = abs(coarse) if coarse != 0 else abs(sigma)
        # Trust ~15 significant digits from the float draw; randomize everything finer.
        trusted_place = ref.adjusted() - 14
        coarse_step = max(Decimal(1).scaleb(trusted_place), quantum)
        coarse_q = (coarse / coarse_step).to_integral_value(rounding=ROUND_HALF_EVEN) * coarse_step
        steps = int((coarse_step / quantum).to_integral_value(rounding=ROUND_HALF_EVEN))
        if steps <= 1:
            return coarse_q
        n_bytes = max(8, (steps.bit_length() + 7) // 8)
        raw = int.from_bytes(_digest_expand(digest + b"\x00tail", n_bytes), "big")
        centered = (raw % steps) - (steps // 2)
        return coarse_q + Decimal(centered) * quantum


_AFFINE_SCALE_DENOMINATOR = 1000


def _affine_coefficients_float(key: bytes, domain: str, stddev: float) -> tuple[float, float]:
    digest = _digest(key, f"{domain}:affine", "coefficients")
    unit = _unit_from_digest(digest, 0)
    scale = 0.5 + 1.5 * unit
    shift = _gaussian_from_digest(digest[8:] + digest[:8]) * stddev
    return scale, shift


def _affine_coefficients_decimal(
    key: bytes, domain: str, stddev: Decimal, *, quantum: Decimal
) -> tuple[Decimal, Decimal]:
    """Returns (a, b) with a = p/1000 in [0.5, 2.0] and b = z * stddev in high-precision Decimal."""
    digest = _digest(key, f"{domain}:affine", "coefficients")
    # Inclusive range 500/1000 .. 2000/1000 => 1501 distinct rationals.
    index = int.from_bytes(digest[:8], "big") % 1501
    scale = (Decimal(500) + Decimal(index)) / Decimal(_AFFINE_SCALE_DENOMINATOR)
    shift_digest = digest[8:] + digest[:8]
    shift = _gaussian_noise_decimal(shift_digest, stddev, quantum=quantum)
    return scale, shift


def _integer_bounds(data_type: T.DataType) -> tuple[int, int] | None:
    bounds = {
        T.ByteType: (-(2**7), 2**7 - 1),
        T.ShortType: (-(2**15), 2**15 - 1),
        T.IntegerType: (-(2**31), 2**31 - 1),
        T.LongType: (-(2**63), 2**63 - 1),
    }
    for kind, span in bounds.items():
        if isinstance(data_type, kind):
            return span
    return None


def _decimal_bounds(data_type: T.DecimalType) -> tuple[Decimal, Decimal]:
    with localcontext() as context:
        context.prec = max(data_type.precision + 8, 50)
        quantum = Decimal(1).scaleb(-data_type.scale)
        maximum = (Decimal(10) ** (data_type.precision - data_type.scale)) - quantum
        return -maximum, maximum


def _clamp_integer(value: int, data_type: T.DataType) -> int:
    bounds = _integer_bounds(data_type)
    if bounds is None:
        return value
    minimum, maximum = bounds
    return minimum if value < minimum else maximum if value > maximum else value


def _clamp_decimal(value: Decimal, data_type: T.DecimalType) -> Decimal:
    """Clamps via the unscaled integer domain so Spark accepts the Decimal(p,s) encoding."""
    minimum, maximum = _decimal_bounds(data_type)
    with localcontext() as context:
        context.prec = max(data_type.precision + 8, 50)
        context.rounding = ROUND_HALF_EVEN
        if value > maximum:
            value = maximum
        elif value < minimum:
            value = minimum
        scale_factor = Decimal(10) ** data_type.scale
        unscaled = int((value * scale_factor).to_integral_value(rounding=ROUND_HALF_EVEN))
        limit = 10**data_type.precision - 1
        if unscaled > limit:
            unscaled = limit
        elif unscaled < -limit:
            unscaled = -limit
        # scaleb avoids Decimal division rounding at default/low precision.
        return Decimal(unscaled).scaleb(-data_type.scale)


def _serialize_entity_parts(parts: Sequence[object]) -> str:
    """Serializes entity-key parts so nulls and embedded separators cannot collide."""

    def encode(value: object) -> object:  # noqa: PLR0911
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return {"float": str(value)}
            return value
        if isinstance(value, Decimal):
            return {"decimal": str(value)}
        if isinstance(value, datetime):
            return {"datetime": value.isoformat()}
        if isinstance(value, date):
            return {"date": value.isoformat()}
        if isinstance(value, (bytes, bytearray)):
            return {"bytes": bytes(value).hex()}
        return {"string": str(value)}

    return json.dumps([encode(part) for part in parts], separators=(",", ":"), ensure_ascii=False)


def _perturb_value(  # noqa: PLR0911
    value: object,
    data_type: T.DataType,
    *,
    noise: float | Decimal,
    scale: float | Decimal,
    shift: float | Decimal,
    mode: str,
) -> object:
    if value is None:
        return None
    if isinstance(data_type, (T.FloatType, T.DoubleType)):
        number = float(str(value)) if not isinstance(value, (int, float)) else float(value)
        if math.isnan(number) or math.isinf(number):
            return number
        scale_f, shift_f, noise_f = float(scale), float(shift), float(noise)
        perturbed = scale_f * number + shift_f if mode == "affine" else number + noise_f
        return float(perturbed)
    if isinstance(data_type, T.DecimalType):
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
        scale_d = scale if isinstance(scale, Decimal) else Decimal(str(scale))
        shift_d = shift if isinstance(shift, Decimal) else Decimal(str(shift))
        noise_d = noise if isinstance(noise, Decimal) else Decimal(str(noise))
        with localcontext() as context:
            context.prec = max(data_type.precision + 16, 50)
            context.rounding = ROUND_HALF_EVEN
            decimal_result = amount * scale_d + shift_d if mode == "affine" else amount + noise_d
        return _clamp_decimal(decimal_result, data_type)
    if isinstance(data_type, (T.ByteType, T.ShortType, T.IntegerType, T.LongType)):
        number = int(str(value)) if not isinstance(value, int) else value
        scale_d = scale if isinstance(scale, Decimal) else Decimal(str(scale))
        shift_d = shift if isinstance(shift, Decimal) else Decimal(str(shift))
        noise_d = noise if isinstance(noise, Decimal) else Decimal(str(noise))
        with localcontext() as context:
            context.prec = 50
            context.rounding = ROUND_HALF_EVEN
            exact = Decimal(number) * scale_d + shift_d if mode == "affine" else Decimal(number) + noise_d
            integer_result = int(exact.to_integral_value(rounding=ROUND_HALF_EVEN))
        return _clamp_integer(integer_result, data_type)
    if isinstance(data_type, T.DateType):
        current = (
            value if isinstance(value, date) and not isinstance(value, datetime) else date.fromisoformat(str(value))
        )
        epoch = date(1970, 1, 1)
        days = (current - epoch).days
        scale_f, shift_f, noise_f = float(scale), float(shift), float(noise)
        mapped_days = round(scale_f * days + shift_f) if mode == "affine" else days + round(noise_f)
        return epoch + timedelta(days=mapped_days)
    timestamp_ntz = getattr(T, "TimestampNTZType", None)
    if isinstance(data_type, T.TimestampType) or (timestamp_ntz is not None and isinstance(data_type, timestamp_ntz)):
        current = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        epoch = datetime(1970, 1, 1, tzinfo=current.tzinfo) if current.tzinfo is not None else datetime(1970, 1, 1)
        seconds = (current - epoch).total_seconds()
        scale_f, shift_f, noise_f = float(scale), float(shift), float(noise)
        mapped_seconds = scale_f * seconds + shift_f if mode == "affine" else seconds + noise_f
        return epoch + timedelta(seconds=float(mapped_seconds))
    raise ValueError(f"perturb is not supported for Spark type {data_type.simpleString()!r}")


def _perturb_column(
    column: Column,
    *,
    data_type: T.DataType,
    domain: str,
    key: bytes,
    rule: PiiRule,
    stddev: float,
    entity_columns: list[str],
    row_entity_columns: list[str],
) -> Column:
    mode = rule.perturb_mode()
    noise_ratio = rule.perturb_noise_ratio()
    exact = _is_exact_numeric_type(data_type)
    quantum = _exact_numeric_quantum(data_type) if exact else Decimal(1)
    if mode == "affine":
        affine_scale: float | Decimal
        affine_shift: float | Decimal
        if exact:
            affine_scale, affine_shift = _affine_coefficients_decimal(
                key, domain, Decimal(str(stddev)), quantum=quantum
            )
        else:
            affine_scale, affine_shift = _affine_coefficients_float(key, domain, stddev)

        def affine_replace(value: object) -> object:
            return _perturb_value(value, data_type, noise=0, scale=affine_scale, shift=affine_shift, mode="affine")

        return F.udf(affine_replace, data_type, useArrow=False)(column)

    sigma_exact: Decimal | None = None
    sigma_float: float | None = None
    if exact:
        sigma_exact = Decimal(str(stddev)) * Decimal(str(noise_ratio))
    else:
        sigma_float = stddev * noise_ratio

    def noise_replace(value: object, *entity: object) -> object:
        if value is None:
            return None
        identity = (
            _serialize_entity_parts(entity)
            if entity_columns or row_entity_columns
            else _serialize_entity_parts((value,))
        )
        digest = _digest(key, f"{domain}:perturb", identity)
        if sigma_exact is not None:
            noise: float | Decimal = _gaussian_noise_decimal(digest, sigma_exact, quantum=quantum)
        else:
            noise = _gaussian_from_digest(digest) * float(sigma_float or 0.0)
        return _perturb_value(value, data_type, noise=noise, scale=1, shift=0, mode="noise")

    extras = [F.col(name) for name in (entity_columns or row_entity_columns)]
    return (
        F.udf(noise_replace, data_type, useArrow=False)(column, *extras)
        if extras
        else F.udf(noise_replace, data_type, useArrow=False)(column)
    )
