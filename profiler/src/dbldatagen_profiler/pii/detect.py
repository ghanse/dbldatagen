"""Bounded deterministic PII detection for Spark DataFrames."""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from itertools import combinations

from pyspark.sql import DataFrame, Row
from pyspark.sql import functions as F
from pyspark.sql import types as T

from dbldatagen_profiler.model import PiiClass, Semantic, SemanticEvidence, Sensitivity


VALUE_MATCH_THRESHOLD = 0.80
FREE_TEXT_MATCH_THRESHOLD = 0.20
NAME_HINT_CONFIDENCE = 0.30
VALUE_CONFIDENCE_FLOOR = 0.80
ACCOUNT_UNIQUENESS_THRESHOLD = 0.90
QUASI_UNIQUE_THRESHOLD = 0.80
MAX_QUASI_CANDIDATES = 12

_EMAIL = re.compile(r"^[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9-]+(?:\.[A-Z0-9-]+)+$", re.I)
_EMAIL_SPAN = re.compile(r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9-]+(?:\.[A-Z0-9-]+)+", re.I)
_PHONE = re.compile(r"^(?:\+[1-9]\d{7,14}|(?:\+?1[-. ]?)?\(?[2-9]\d{2}\)?[-. ]?[2-9]\d{2}[-. ]?\d{4})$")
_PHONE_SPAN = re.compile(r"(?<!\d)(?:\+[1-9]\d{7,14}|(?:\+?1[-. ]?)?\(?[2-9]\d{2}\)?[-. ]?[2-9]\d{2}[-. ]?\d{4})(?!\d)")
_SSN = re.compile(r"^(?!000|666|9\d\d)\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}$")
_SSN_SPAN = re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}(?!\d)")
_CARD_SPAN = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_IBAN = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)
_POSTAL = re.compile(r"^\d{5}(?:-\d{4})?$")

_HINTS: tuple[tuple[PiiClass, Sensitivity, re.Pattern[str]], ...] = (
    (PiiClass.EMAIL, Sensitivity.DIRECT_IDENTIFIER, re.compile(r"(^|_)(e_?mail|email_?address)($|_)")),
    (PiiClass.PHONE, Sensitivity.DIRECT_IDENTIFIER, re.compile(r"(^|_)(phone|mobile|telephone)($|_)")),
    (
        PiiClass.PERSON_NAME,
        Sensitivity.DIRECT_IDENTIFIER,
        re.compile(r"(^|_)(first_?name|last_?name|full_?name|person_?name)($|_)"),
    ),
    (PiiClass.STREET_ADDRESS, Sensitivity.DIRECT_IDENTIFIER, re.compile(r"(^|_)(street|address|address_?line)($|_)")),
    (PiiClass.CITY, Sensitivity.QUASI_IDENTIFIER, re.compile(r"(^|_)(city|town)($|_)")),
    (PiiClass.POSTAL_CODE, Sensitivity.QUASI_IDENTIFIER, re.compile(r"(^|_)(zip|zipcode|postal_?code)($|_)")),
    (PiiClass.NATIONAL_ID, Sensitivity.DIRECT_IDENTIFIER, re.compile(r"(^|_)(ssn|national_?id|tax_?id)($|_)")),
    (PiiClass.CREDIT_CARD, Sensitivity.DIRECT_IDENTIFIER, re.compile(r"(^|_)(credit_?card|card_?number|pan)($|_)")),
    (PiiClass.IBAN, Sensitivity.DIRECT_IDENTIFIER, re.compile(r"(^|_)iban($|_)")),
    (PiiClass.IP_ADDRESS, Sensitivity.DIRECT_IDENTIFIER, re.compile(r"(^|_)(ip|ip_?address)($|_)")),
    (PiiClass.DATE_OF_BIRTH, Sensitivity.QUASI_IDENTIFIER, re.compile(r"(^|_)(dob|birth_?date|date_?of_?birth)($|_)")),
    (PiiClass.ACCOUNT_ID, Sensitivity.DIRECT_IDENTIFIER, re.compile(r"(^|_)(account|customer|user|member)_?id($|_)")),
    (PiiClass.SECRET_TOKEN, Sensitivity.SECRET, re.compile(r"(^|_)(secret|token|api_?key|password|credential)($|_)")),
    (PiiClass.FREE_TEXT_PII, Sensitivity.FREE_TEXT, re.compile(r"(^|_)(notes?|comments?|description|free_?text)($|_)")),
    (PiiClass.OTHER, Sensitivity.DIRECT_IDENTIFIER, re.compile(r"(^|_)other_?pii($|_)")),
)


def detect(
    tables: Mapping[str, DataFrame], *, sample_rows: int = 10_000, seed: int = 0
) -> dict[str, dict[str, Semantic]]:
    """Detects column semantics from a bounded deterministic sample of each table."""
    if sample_rows < 1:
        raise ValueError("sample_rows must be positive")
    return {name: _detect_table(frame, sample_rows, seed) for name, frame in tables.items()}


def _detect_table(frame: DataFrame, sample_rows: int, seed: int) -> dict[str, Semantic]:
    rows = _sample(frame, sample_rows, seed)
    return _detect_rows(frame, rows)


def _detect_rows(frame: DataFrame, rows: Sequence[Row]) -> dict[str, Semantic]:
    result: dict[str, Semantic] = {}
    for field in frame.schema:
        values = [row[field.name] for row in rows if row[field.name] is not None]
        _detect_field(field, values, field.name, result)
    return result


def _detect_field(field: T.StructField, values: list[object], path: str, result: dict[str, Semantic]) -> None:
    data_type = field.dataType
    if not isinstance(data_type, (T.StructType, T.ArrayType, T.MapType)):
        semantic = _classify(field, values)
        if semantic is not None:
            result[path] = semantic
    if isinstance(data_type, T.StructType):
        for child in data_type.fields:
            child_values = [_struct_value(value, child.name) for value in values]
            _detect_field(child, [value for value in child_values if value is not None], f"{path}.{child.name}", result)
    elif isinstance(data_type, T.ArrayType):
        elements = [element for value in values if isinstance(value, list | tuple) for element in value]
        child = T.StructField("element", data_type.elementType, data_type.containsNull)
        _detect_field(child, [value for value in elements if value is not None], f"{path}.element", result)
    elif isinstance(data_type, T.MapType):
        mappings = [value for value in values if isinstance(value, dict)]
        keys = [key for value in mappings for key in value]
        map_values = [item for value in mappings for item in value.values()]
        _detect_field(T.StructField("key", data_type.keyType, False), keys, f"{path}.key", result)
        _detect_field(
            T.StructField("value", data_type.valueType, data_type.valueContainsNull),
            [value for value in map_values if value is not None],
            f"{path}.value",
            result,
        )


def _struct_value(value: object, name: str) -> object:
    if isinstance(value, Row):
        return value[name]
    if isinstance(value, dict):
        return value.get(name)
    return None


def analyze_table(
    frame: DataFrame, sample_rows: int = 10_000, seed: int = 0
) -> tuple[dict[str, Semantic], list[tuple[list[str], float, int]]]:
    """Detects semantics and quasi-identifiers from one collected sample."""
    rows = _sample(frame, sample_rows, seed)
    semantics = _detect_rows(frame, rows)
    return semantics, _find_quasi_identifier_sets(frame, semantics, rows)


def _sample(frame: DataFrame, sample_rows: int, seed: int) -> list[Row]:
    columns = [F.coalesce(F.col(f.name).cast("string"), F.lit("\u0000")) for f in frame.schema]
    order = F.xxhash64(F.lit(seed), *columns)
    return frame.orderBy(order, *columns).limit(sample_rows).collect()


def _classify(field: T.StructField, values: list[object]) -> Semantic | None:
    name = re.sub(r"[^a-z0-9]+", "_", field.name.lower()).strip("_")
    hint = next(((kind, sensitivity) for kind, sensitivity, pattern in _HINTS if pattern.search(name)), None)
    evidence: list[SemanticEvidence] = []
    if hint:
        evidence.append(SemanticEvidence(method="column_name", match_rate=1.0))
    validators: list[tuple[PiiClass, Sensitivity, str, Callable[[object], bool]]] = [
        (PiiClass.EMAIL, Sensitivity.DIRECT_IDENTIFIER, "email", lambda value: bool(_EMAIL.fullmatch(str(value)))),
        (PiiClass.PHONE, Sensitivity.DIRECT_IDENTIFIER, "phone", lambda value: bool(_PHONE.fullmatch(str(value)))),
        (PiiClass.CREDIT_CARD, Sensitivity.DIRECT_IDENTIFIER, "luhn", _is_card),
        (PiiClass.IBAN, Sensitivity.DIRECT_IDENTIFIER, "iban_mod97", _is_iban),
        (PiiClass.NATIONAL_ID, Sensitivity.DIRECT_IDENTIFIER, "us_ssn", lambda value: bool(_SSN.fullmatch(str(value)))),
        (PiiClass.IP_ADDRESS, Sensitivity.DIRECT_IDENTIFIER, "ip_address", _is_ip),
        (PiiClass.ACCOUNT_ID, Sensitivity.DIRECT_IDENTIFIER, "uuid", _is_uuid),
        (PiiClass.SECRET_TOKEN, Sensitivity.SECRET, "opaque_token_entropy", _is_opaque_token),
    ]
    matches = [
        (kind, sensitivity, method, _rate(values, validator)) for kind, sensitivity, method, validator in validators
    ]
    value_match = max(matches, key=lambda item: item[3], default=None)
    if value_match and value_match[3] >= VALUE_MATCH_THRESHOLD:
        kind, sensitivity, method, rate = value_match
        if kind == PiiClass.CREDIT_CARD and not (hint and hint[0] == kind) and "sku" in name:
            value_match = None
        else:
            evidence.append(SemanticEvidence(method=method, match_rate=rate))
            confidence = min(1.0, VALUE_CONFIDENCE_FLOOR + 0.2 * rate + (0.05 if hint and hint[0] == kind else 0.0))
            return Semantic(pii_class=kind, sensitivity=sensitivity, confidence=confidence, evidence=evidence)
    if isinstance(field.dataType, T.DateType) and hint and hint[0] == PiiClass.DATE_OF_BIRTH:
        rate = _rate(values, _plausible_birth_date)
        if rate >= VALUE_MATCH_THRESHOLD:
            evidence.append(SemanticEvidence(method="plausible_age", match_rate=rate))
            return Semantic(
                pii_class=PiiClass.DATE_OF_BIRTH,
                sensitivity=Sensitivity.QUASI_IDENTIFIER,
                confidence=min(1.0, 0.75 + 0.2 * rate),
                evidence=evidence,
            )
    if hint and hint[0] == PiiClass.ACCOUNT_ID and values:
        uniqueness = len({str(value) for value in values}) / len(values)
        evidence.append(SemanticEvidence(method="sample_uniqueness", match_rate=uniqueness))
        if uniqueness >= ACCOUNT_UNIQUENESS_THRESHOLD:
            return Semantic(pii_class=hint[0], sensitivity=hint[1], confidence=0.85, evidence=evidence)
    span_rate = _rate(values, _contains_pii_span)
    if span_rate >= FREE_TEXT_MATCH_THRESHOLD:
        evidence.append(SemanticEvidence(method="pii_span", match_rate=span_rate))
        return Semantic(
            pii_class=PiiClass.FREE_TEXT_PII,
            sensitivity=Sensitivity.FREE_TEXT,
            confidence=min(0.95, 0.6 + span_rate * 0.35),
            evidence=evidence,
        )
    if hint:
        return Semantic(pii_class=hint[0], sensitivity=hint[1], confidence=NAME_HINT_CONFIDENCE, evidence=evidence)
    return None


def find_quasi_identifier_sets(
    frame: DataFrame, semantics: Mapping[str, Semantic], sample_rows: int, seed: int
) -> list[tuple[list[str], float, int]]:
    """Finds bounded pairs and triples whose sampled combinations are mostly unique."""
    rows = _sample(frame, sample_rows, seed)
    return _find_quasi_identifier_sets(frame, semantics, rows)


def _find_quasi_identifier_sets(
    frame: DataFrame, semantics: Mapping[str, Semantic], rows: Sequence[Row]
) -> list[tuple[list[str], float, int]]:
    direct = {
        name
        for name, semantic in semantics.items()
        if semantic.sensitivity in {Sensitivity.DIRECT_IDENTIFIER, Sensitivity.SECRET}
    }
    candidates = []
    for field in frame.schema:
        values = [row[field.name] for row in rows if row[field.name] is not None]
        distinct = len({repr(value) for value in values})
        semantic = semantics.get(field.name)
        preferred = isinstance(field.dataType, (T.DateType, T.TimestampType)) or (
            semantic is not None and semantic.pii_class == PiiClass.POSTAL_CODE
        )
        if field.name not in direct and values and (preferred or 2 <= distinct <= 0.8 * len(values)):
            candidates.append(field.name)
    found = []
    for size in (2, 3):
        for columns in combinations(candidates[:MAX_QUASI_CANDIDATES], size):
            keys = [tuple(row[column] for column in columns) for row in rows]
            keys = [key for key in keys if all(value is not None for value in key)]
            keys = [tuple(repr(value) for value in key) for key in keys]
            counts = Counter(keys)
            if not keys:
                continue
            unique_fraction = sum(count == 1 for count in counts.values()) / len(keys)
            if unique_fraction >= QUASI_UNIQUE_THRESHOLD:
                found.append((list(columns), unique_fraction, min(counts.values())))
    return found


def _rate(values: list[object], validator: Callable[[object], bool]) -> float:
    return sum(validator(value) for value in values) / len(values) if values else 0.0


def _digits(value: object) -> str:
    return re.sub(r"\D", "", str(value))


def _is_card(value: object) -> bool:
    digits = _digits(value)
    if not 13 <= len(digits) <= 19:
        return False
    total = sum(
        (digit * 2 - 9 if digit * 2 > 9 else digit * 2) if index % 2 else digit
        for index, digit in enumerate(map(int, reversed(digits)))
    )
    return total % 10 == 0


def _is_iban(value: object) -> bool:
    compact = re.sub(r"\s", "", str(value)).upper()
    if not _IBAN.fullmatch(compact):
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(ord(char) - 55) if char.isalpha() else char for char in rearranged)
    return int(numeric) % 97 == 1


def _is_ip(value: object) -> bool:
    try:
        ipaddress.ip_address(str(value))
    except ValueError:
        return False
    return True


def _is_opaque_token(value: object) -> bool:
    text = str(value)
    if len(text) < 20 or any(char.isspace() for char in text):
        return False
    entropy = -sum((count / len(text)) * math.log2(count / len(text)) for count in Counter(text).values())
    return entropy >= 3.5 and len(set(text)) >= 10


def _is_uuid(value: object) -> bool:
    return _UUID.fullmatch(str(value)) is not None


def _plausible_birth_date(value: object) -> bool:
    moment = value.date() if isinstance(value, datetime) else value
    if not isinstance(moment, date):
        return False
    today = date.today()
    age = today.year - moment.year - ((today.month, today.day) < (moment.month, moment.day))
    return 0 <= age <= 120


def _contains_pii_span(value: object) -> bool:
    text = str(value)
    basic_matches = [*_EMAIL_SPAN.finditer(text), *_PHONE_SPAN.finditer(text), *_SSN_SPAN.finditer(text)]
    basic_matches.extend(match for match in _CARD_SPAN.finditer(text) if _is_card(match.group()))
    basic = any(match.span() != (0, len(text)) for match in basic_matches)
    if basic:
        return True
    tokens = re.findall(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b|\b(?:\d{1,3}\.){3}\d{1,3}\b", text, re.I)
    return any((_is_iban(token) or _is_ip(token)) and token != text for token in tokens)
