"""Regex helpers: random strings matching a regex, and the literal text a regex embeds.

Both walk the parse tree of Python's `re` module. Generation supports literals, character classes, `.`, groups,
alternation, and repeats; lookarounds and back-references are not supported.
"""

from __future__ import annotations

import importlib
import math
import re
import string
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


_parser: Any = importlib.import_module("re._parser" if sys.version_info >= (3, 11) else "sre_parse")

PRINTABLE = string.ascii_letters + string.digits + string.punctuation + " "
CATEGORIES = {
    "CATEGORY_DIGIT": string.digits,
    "CATEGORY_WORD": string.ascii_letters + string.digits + "_",
    "CATEGORY_SPACE": " ",
}
UNBOUNDED_EXTRA = 8
REPEATS = ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT")


class UnsupportedRegex(ValueError):
    """The regex uses a construct strings cannot be generated for."""


def generate(regex: str, count: int, rng: np.random.Generator) -> list[str]:
    """Returns `count` random strings that fully match `regex`.

    Raises:
        UnsupportedRegex: If the regex does not parse or uses an unsupported construct.
    """
    try:
        items = _parser.parse(regex)
    except re.error as error:
        raise UnsupportedRegex(str(error)) from error
    return ["".join(_generate(items, rng)) for _ in range(count)]


def _generate(items: Sequence[tuple[Any, Any]], rng: np.random.Generator) -> list[str]:
    out: list[str] = []
    for op, av in items:
        name = op.name
        if name == "LITERAL":
            out.append(chr(av))
        elif name == "NOT_LITERAL":
            out.append(_choice([c for c in PRINTABLE if c != chr(av)], rng))
        elif name == "ANY":
            out.append(_choice(PRINTABLE, rng))
        elif name == "IN":
            out.append(_choice(_class(av), rng))
        elif name == "CATEGORY":
            out.append(_choice(_category(av.name), rng))
        elif name == "BRANCH":
            out += _generate(av[1][int(rng.integers(len(av[1])))], rng)
        elif name == "SUBPATTERN":
            out += _generate(av[-1], rng)
        elif name in REPEATS:
            low, high, inner = av
            high = low + UNBOUNDED_EXTRA if high == _parser.MAXREPEAT else high
            for _ in range(int(rng.integers(low, high + 1))):
                out += _generate(inner, rng)
        elif name != "AT":
            raise UnsupportedRegex(f"cannot generate {name}")
    return out


def _choice(characters: str | list[str], rng: np.random.Generator) -> str:
    if not characters:
        raise UnsupportedRegex("empty character class")
    return characters[int(rng.integers(len(characters)))]


def _category(name: str) -> str:
    if name.startswith("CATEGORY_NOT_"):
        excluded = CATEGORIES.get(name.replace("_NOT", ""), "")
        return "".join(c for c in PRINTABLE if c not in excluded)
    if name not in CATEGORIES:
        raise UnsupportedRegex(f"cannot generate {name}")
    return CATEGORIES[name]


def _class(items: Sequence[tuple[Any, Any]]) -> str:
    members: set[str] = set()
    negate = False
    for op, av in items:
        name = op.name
        if name == "NEGATE":
            negate = True
        elif name == "LITERAL":
            members.add(chr(av))
        elif name == "RANGE":
            members.update(chr(c) for c in range(av[0], av[1] + 1))
        elif name == "CATEGORY":
            members.update(_category(av.name))
        else:
            raise UnsupportedRegex(f"cannot generate {name} in a class")
    if negate:
        return "".join(c for c in PRINTABLE if c not in members)
    return "".join(sorted(members))


def literals(regex: str) -> list[str]:
    """Returns the runs of literal characters in `regex`, one per alternative; the regex itself if it does not parse."""
    try:
        items = _parser.parse(regex)
    except re.error:
        return [regex]
    runs: list[str] = []
    _literals(items, runs)
    return [r for r in runs if r]


def _literals(items: Sequence[tuple[Any, Any]], runs: list[str]) -> None:
    current: list[str] = []
    for op, av in items:
        name = op.name
        single = name == "IN" and len(av) == 1 and av[0][0].name == "LITERAL"
        if name == "LITERAL" or single:
            current.append(chr(av if name == "LITERAL" else av[0][1]))
            continue
        runs.append("".join(current))
        current = []
        if name == "BRANCH":
            for branch in av[1]:
                _literals(branch, runs)
        elif name == "SUBPATTERN":
            _literals(av[-1], runs)
        elif name in REPEATS:
            _literals(av[2], runs)
    runs.append("".join(current))


UNICODE = 2**16
MAX_EXPONENT = 300
CLASS_SIZES = {"CATEGORY_DIGIT": 10, "CATEGORY_WORD": 63, "CATEGORY_SPACE": 6}


def language_size(regex: str) -> float:
    """Returns an estimate of how many strings fully match `regex`; `inf` when a repeat is unbounded.

    Character classes count their members, with `.`, negated classes, and negated categories counted as 2**16
    characters, `\\d` as 10, `\\w` as 63, and `\\s` as 6. Anchors and lookarounds count as one. A regex that does not
    parse counts as the single literal string it is.
    """
    try:
        items = _parser.parse(regex)
    except re.error:
        return 1.0
    return _size(items)


def _size(items: Sequence[tuple[Any, Any]]) -> float:
    total = 1.0
    for item in items:
        total *= _item_size(item)
        if total == float("inf"):
            return total
    return total


def _repeat_size(av: tuple[Any, Any, Any]) -> float:
    low, high, inner = av
    if high == _parser.MAXREPEAT:
        return float("inf")
    base = _size(inner)
    if base <= 1:
        return 1.0
    if high * math.log10(base) > MAX_EXPONENT:
        return float("inf")
    return float(sum(base**count for count in range(low, high + 1)))


ONE_SIZED = frozenset({"LITERAL", "AT", "ASSERT", "ASSERT_NOT", "GROUPREF", "GROUPREF_EXISTS"})
SIZES: dict[str, Callable[[Any], float]] = {
    "CATEGORY": lambda av: float(CLASS_SIZES.get(av.name, UNICODE)),
    "IN": lambda av: _class_size(av),
    "BRANCH": lambda av: sum(_size(branch) for branch in av[1]),
    "SUBPATTERN": lambda av: _size(av[-1]),
    "ATOMIC_GROUP": lambda av: _size(av[-1]),
    **dict.fromkeys(REPEATS, _repeat_size),
}


def _item_size(item: tuple[Any, Any]) -> float:
    op, av = item
    if op.name in ONE_SIZED:
        return 1.0
    return SIZES[op.name](av) if op.name in SIZES else float(UNICODE)


def _class_size(items: Sequence[tuple[Any, Any]]) -> float:
    if any(op.name == "NEGATE" for op, _ in items):
        return float(UNICODE)
    size = 0.0
    for op, av in items:
        name = op.name
        if name == "LITERAL":
            size += 1
        elif name == "RANGE":
            size += av[1] - av[0] + 1
        elif name == "CATEGORY":
            size += CLASS_SIZES.get(av.name, UNICODE)
        else:
            size += UNICODE
    return size
