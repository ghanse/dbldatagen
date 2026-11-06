"""Small deterministic regex-to-string generator used by the sampler."""

from __future__ import annotations

import re
import sre_parse
import string
from typing import Any

import numpy as np


_WORD = string.ascii_letters + string.digits + "_"
_SPACE = " \t"
_PRINTABLE = string.ascii_letters + string.digits + " _-."


class UnsupportedRegex(ValueError):
    """Indicates that a pattern uses syntax outside the practical subset."""


def generate_regex(pattern: str, rng: np.random.Generator, *, max_length: int = 32) -> str:
    """Generates a full-matching string for the supported regex subset."""
    parsed = sre_parse.parse(pattern)
    value = _emit(parsed, rng, max(1, max_length))
    if re.fullmatch(pattern, value) is None:
        raise UnsupportedRegex(pattern)
    return value


def _emit(tokens: Any, rng: np.random.Generator, budget: int) -> str:  # noqa: ANN401
    result: list[str] = []
    for op, arg in tokens:
        remaining = max(1, budget - sum(map(len, result)))
        if op is sre_parse.LITERAL:
            result.append(chr(arg))
        elif op is sre_parse.NOT_LITERAL:
            result.append("x" if arg != ord("x") else "y")
        elif op is sre_parse.ANY:
            result.append(_pick(_PRINTABLE, rng))
        elif op is sre_parse.IN:
            result.append(_emit_class(arg, rng))
        elif op is sre_parse.CATEGORY:
            result.append(_emit_category(arg, rng))
        elif op is sre_parse.SUBPATTERN:
            result.append(_emit(arg[-1], rng, remaining))
        elif op is sre_parse.BRANCH:
            branches = arg[1]
            result.append(_emit(branches[int(rng.integers(len(branches)))], rng, remaining))
        elif op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
            low, high, child = arg
            upper = min(high, remaining) if high != sre_parse.MAXREPEAT else min(low + 4, remaining)
            count = low if upper <= low else int(rng.integers(low, upper + 1))
            result.extend(_emit(child, rng, remaining) for _ in range(count))
        elif op is sre_parse.AT:
            continue
        else:
            raise UnsupportedRegex(f"unsupported regex operation: {op}")
    return "".join(result)


def _emit_class(tokens: Any, rng: np.random.Generator) -> str:  # noqa: ANN401
    negate = any(op is sre_parse.NEGATE for op, _ in tokens)
    choices: list[str] = []
    for op, arg in tokens:
        if op is sre_parse.LITERAL:
            choices.append(chr(arg))
        elif op is sre_parse.RANGE:
            choices.extend(chr(value) for value in range(arg[0], arg[1] + 1))
        elif op is sre_parse.CATEGORY:
            choices.extend(_category_chars(arg))
    if negate:
        choices = [char for char in _PRINTABLE if char not in choices]
    if not choices:
        raise UnsupportedRegex("empty or unsupported character class")
    return _pick(choices, rng)


def _category_chars(category: Any) -> str:  # noqa: ANN401
    if category is sre_parse.CATEGORY_DIGIT:
        return string.digits
    if category is sre_parse.CATEGORY_WORD:
        return _WORD
    if category is sre_parse.CATEGORY_SPACE:
        return _SPACE
    if category is sre_parse.CATEGORY_NOT_DIGIT:
        return string.ascii_letters + " _"
    if category is sre_parse.CATEGORY_NOT_WORD:
        return " -.!"
    if category is sre_parse.CATEGORY_NOT_SPACE:
        return string.ascii_letters + string.digits
    raise UnsupportedRegex(f"unsupported category: {category}")


def _emit_category(category: Any, rng: np.random.Generator) -> str:  # noqa: ANN401
    return _pick(_category_chars(category), rng)


def _pick(values: Any, rng: np.random.Generator) -> str:  # noqa: ANN401
    return str(values[int(rng.integers(len(values)))])
