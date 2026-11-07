"""Keyed join-consistent substitution of sensitive Spark columns."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from dbldatagen_profiler.model import PiiClass, Profile, Semantic, Sensitivity


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


def substitute(  # noqa: PLR0912
    tables: dict[str, DataFrame],
    profile: Profile,
    *,
    key: bytes,
    options: SubstitutionOptions | None = None,
) -> SubstitutionResult:
    """Replaces sensitive values with deterministic keyed surrogates."""
    if not key:
        raise ValueError("key must not be empty")
    opts = options or SubstitutionOptions()
    if opts.date_shift_days < 0:
        raise ValueError("date_shift_days must not be negative")
    if not 1 <= opts.surrogate_bits <= 256:
        raise ValueError("surrogate_bits must be in [1, 256]")
    profiles = {table.name: table for table in profile.tables}
    missing = sorted(set(tables) - set(profiles))
    if missing:
        raise ValueError(f"tables are absent from profile: {missing}")
    domains = _domains(profile, opts)
    domain_semantics: dict[str, Semantic] = {}
    for table in profile.tables:
        for column in table.columns:
            if column.semantic is not None and column.semantic.sensitivity != Sensitivity.QUASI_IDENTIFIER:
                domain_semantics.setdefault(domains[f"{table.name}.{column.name}"], column.semantic)
    mappings: dict[str, DataFrame] = {}
    collision_counts: dict[str, int] = {}
    for domain, semantic in domain_semantics.items():
        if semantic.pii_class == PiiClass.FREE_TEXT_PII:
            continue
        members = [reference for reference, member_domain in domains.items() if member_domain == domain]
        mapping, collisions = _build_mapping(tables, members, semantic, domain, key, opts)
        mappings[domain], collision_counts[domain] = mapping, collisions
    output: dict[str, DataFrame] = {}
    reports: list[dict[str, Any]] = []
    for table_name, frame in tables.items():
        result = frame
        columns = {column.name: column for column in profiles[table_name].columns}
        for name in frame.columns:
            reference = f"{table_name}.{name}"
            current_column = columns.get(name)
            current_semantic = current_column.semantic if current_column is not None else None
            effective_semantic = domain_semantics.get(domains[reference], current_semantic)
            generalization = opts.generalize.get(reference) or opts.generalize.get(name)
            if generalization:
                result = result.withColumn(name, _generalize(F.col(name), generalization, frame.schema[name].dataType))
                count = frame.where(F.col(name).isNotNull()).count()
                reports.append(
                    _report(table_name, name, current_semantic, domains[reference], count, 0, generalization)
                )
            elif effective_semantic is not None and effective_semantic.sensitivity != Sensitivity.QUASI_IDENTIFIER:
                if effective_semantic.pii_class == PiiClass.FREE_TEXT_PII:
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
                    mapping = mapping.select(F.col("original").alias(original), F.col("surrogate").alias(surrogate))
                    result = (
                        result.join(mapping, F.col(name).cast("string") == F.col(original), "left")
                        .withColumn(name, F.coalesce(F.col(surrogate).cast(frame.schema[name].dataType), F.col(name)))
                        .drop(original, surrogate)
                    )
                count = frame.where(F.col(name).isNotNull() & (F.col(name).cast("string") != "")).count()
                reports.append(
                    _report(
                        table_name,
                        name,
                        effective_semantic,
                        domains[reference],
                        count,
                        collision_counts.get(domains[reference], 0),
                        None,
                    )
                )
        output[table_name] = result
    return SubstitutionResult(tables=output, report={"columns": reports})


def _build_mapping(
    tables: dict[str, DataFrame],
    members: list[str],
    semantic: Semantic,
    domain: str,
    key: bytes,
    options: SubstitutionOptions,
) -> tuple[DataFrame, int]:
    values: DataFrame | None = None
    for reference in members:
        table, column = reference.split(".", maxsplit=1)
        selected = (
            tables[table]
            .select(F.col(column).cast("string").alias("original"))
            .where(F.col("original").isNotNull() & (F.col("original") != ""))
        )
        values = selected if values is None else values.union(selected)
    if values is None:
        raise ValueError(f"domain {domain!r} has no columns")
    mapping = values.distinct().withColumn("counter", F.lit(0)).cache()
    make = F.udf(
        lambda value, counter: _mapping_surrogate(semantic.pii_class, value, domain, key, options, counter),
        T.StringType(),
    )
    collisions_resolved = 0
    for _ in range(100):
        candidate = mapping.withColumn("surrogate", make("original", "counter"))
        duplicates = candidate.groupBy("surrogate").count().where(F.col("count") > 1).select("surrogate")
        originals = values.select(F.col("original").alias("existing_original")).distinct()
        survivors = candidate.join(originals, F.col("surrogate") == F.col("existing_original"), "inner").select(
            "surrogate"
        )
        collided = duplicates.union(survivors).distinct()
        collision_count = collided.count()
        if collision_count == 0:
            mapping.unpersist()
            return candidate.select("original", "surrogate"), collisions_resolved
        updated = candidate.join(collided.withColumn("collided", F.lit(True)), "surrogate", "left").withColumn(
            "__rank", F.row_number().over(Window.partitionBy("surrogate").orderBy("original"))
        )
        updated = updated.withColumn(
            "counter",
            F.when(F.col("collided") & (F.col("__rank") > 1), F.col("counter") + F.col("__rank") - 1).otherwise(
                F.col("counter")
            ),
        )
        collisions_resolved += updated.where(F.col("collided") & (F.col("__rank") > 1)).count()
        mapping.unpersist()
        mapping = updated.select("original", "counter").cache()
    mapping.unpersist()
    raise RuntimeError(f"could not resolve surrogate collisions in domain {domain!r}")


def _mapping_surrogate(
    kind: PiiClass,
    value: str,
    domain: str,
    key: bytes,
    options: SubstitutionOptions,
    counter: int,
) -> str:
    if kind == PiiClass.DATE_OF_BIRTH:
        original = date.fromisoformat(value)
        digest = _digest(key, domain, value, options.surrogate_bits, counter)
        width = options.date_shift_days * 2 + 1
        return (original + timedelta(days=int.from_bytes(digest, "big") % width - options.date_shift_days)).isoformat()
    return _surrogate(kind, value, domain, key, options, counter)


def _domains(profile: Profile, options: SubstitutionOptions) -> dict[str, str]:
    references = {f"{table.name}.{column.name}" for table in profile.tables for column in table.columns}
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
            width = options.date_shift_days * 2 + 1
            return value + timedelta(days=int.from_bytes(digest, "big") % width - options.date_shift_days)
        return _surrogate(semantic.pii_class, str(value), domain, key, options)

    return F.udf(replace, data_type)


def _digest(key: bytes, domain: str, value: str, bits: int = 256, counter: int = 0) -> bytes:
    message = f"{domain}\u001f{value}\u001f{counter}".encode()
    digest = bytearray(hmac.new(key, message, hashlib.sha256).digest())
    size = (bits + 7) // 8
    digest = digest[:size]
    if bits % 8:
        digest[-1] &= (1 << (bits % 8)) - 1
    return bytes(digest)


def _surrogate(  # noqa: PLR0911
    kind: PiiClass, value: str, domain: str, key: bytes, options: SubstitutionOptions, counter: int = 0
) -> str:
    digest = _digest(key, domain, value, options.surrogate_bits, counter)
    number = int.from_bytes(digest, "big")
    token = digest.hex()
    if kind == PiiClass.EMAIL:
        domain_part = value.rpartition("@")[2] if options.keep_email_domain and "@" in value else "masked.invalid"
        return f"u{token[: max(8, min(24, len(value) - len(domain_part) - 1))]}@{domain_part}"
    if kind == PiiClass.PHONE:
        digits = re.sub(r"\D", "", value)
        generated = str(number).zfill(len(digits))[-len(digits) :]
        if value.startswith("+"):
            country_length = max(1, len(digits) - 10)
            generated = digits[:country_length] + generated[country_length:]
        nanp_offset = len(generated) - 10
        if len(generated) >= 10 and (len(generated) == 10 or generated[:nanp_offset] == "1"):
            generated = (
                generated[:nanp_offset]
                + str(int(generated[nanp_offset]) % 8 + 2)
                + generated[nanp_offset + 1 : nanp_offset + 3]
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


def _generalize(column: Column, method: str, data_type: T.DataType) -> Column:
    if method in {"postal_prefix", "postal_code_prefix"}:
        return F.concat(F.substring(column.cast("string"), 1, 3), F.lit("**"))
    if method in {"date_month", "month"}:
        return F.date_trunc("month", column) if isinstance(data_type, T.TimestampType) else F.trunc(column, "month")
    raise ValueError(f"unknown generalization {method!r}")


def _report(
    table: str,
    column: str,
    semantic: Semantic | None,
    domain: str,
    count: int,
    collisions: int,
    generalization: str | None,
) -> dict[str, Any]:
    return {
        "table": table,
        "column": column,
        "class": semantic.pii_class.value if semantic else None,
        "domain": domain,
        "values_substituted": count,
        "collisions_resolved": collisions,
        "generalizations": [generalization] if generalization else [],
    }
