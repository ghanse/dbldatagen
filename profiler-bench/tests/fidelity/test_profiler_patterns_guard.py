"""Profiler string patterns checked by the frozen privacy guard's own regex functions."""

import re

import pytest

from dbldatagen_profiler.engines.spark.patterns import COARSE, FINE, Shape, patterns_for
from profiler_bench.fidelity import regex
from profiler_bench.fidelity.config import load_config


K = 10

CASES = {
    "sku": [Shape(FINE, "A-9", (3, 1, 4), (3, 1, 4), 50)],
    "metacharacters": [
        Shape(FINE, "9(9)9.a+A", (3, 1, 3, 1, 2, 1, 4, 1, 2), (3, 1, 3, 1, 2, 1, 6, 1, 2), 20),
        Shape(FINE, "a[9]a{9}a", (2, 1, 3, 1, 2, 1, 3, 1, 2), (5, 1, 3, 1, 5, 1, 3, 1, 5), 20),
        Shape(FINE, "9|9\\a^9$9*a?", (2, 1, 2, 1, 3, 1, 2, 1, 2, 1, 3, 1), (2, 1, 2, 1, 3, 1, 2, 1, 2, 1, 3, 1), 20),
        Shape(FINE, "a-9", (4, 1, 4), (4, 2, 4), 20),
    ],
    "unicode": [
        Shape(FINE, "a\u00e9a", (4, 1, 4), (7, 1, 7), 30),
        Shape(FINE, "A\u4e2d9", (3, 1, 4), (3, 2, 4), 30),
        Shape(FINE, "a\u00a0a", (5, 1, 5), (8, 1, 8), 30),
    ],
    "variable_runs": [
        Shape(FINE, "Aa9", (1, 1, 4), (1, 12, 9), 40),
        Shape(FINE, "a_a9", (2, 1, 2, 1), (12, 1, 12, 5), 40),
    ],
    "email": [
        Shape(FINE, "a.a9@a.a", (4, 1, 4, 2, 1, 4, 1, 3), (6, 1, 6, 2, 1, 7, 1, 4), 30),
        Shape(FINE, "a_a9@a.a", (4, 1, 4, 2, 1, 4, 1, 3), (6, 1, 6, 2, 1, 7, 1, 4), 30),
    ],
    "hex": [Shape(COARSE, "x", (32,), (32,), 40, frozenset({"digit", "lower_af"}))],
    "uuid": [
        Shape(
            COARSE,
            "x-x-x-x-x",
            (8, 1, 4, 1, 4, 1, 4, 1, 12),
            (8, 1, 4, 1, 4, 1, 4, 1, 12),
            40,
            frozenset({"digit", "lower_af"}),
        )
    ],
}


def _emitted() -> list[tuple[str, bool, str]]:
    found = []
    for name, shapes in CASES.items():
        for literal_free in (False, True):
            for pattern in patterns_for(shapes, sum(s.count for s in shapes), K, literal_free=literal_free) or []:
                found.append((name, literal_free, pattern.regex))
    return found


EMITTED = _emitted()


def test_every_case_emits_patterns_in_both_modes() -> None:
    assert {(name, free) for name, free, _ in EMITTED} == {(name, free) for name in CASES for free in (False, True)}


@pytest.mark.parametrize(("case", "literal_free", "pattern"), EMITTED)
def test_pattern_passes_the_frozen_guard_functions(case: str, literal_free: bool, pattern: str) -> None:
    limits = load_config().privacy
    limit = max(K, limits.min_regex_language)

    assert regex.in_dialect(pattern), pattern
    re.compile(pattern)
    if literal_free:
        assert regex.is_generalized(pattern), pattern
    probed = regex.count_matches(pattern, 0, limits.max_regex_length, max_dfa_states=limits.max_dfa_states)
    assert probed is not regex.BUDGET_EXCEEDED, pattern
    shortest = next(
        n
        for n in range(limits.max_regex_length + 1)
        if regex.count_matches(pattern, n, n, max_dfa_states=limits.max_dfa_states)
    )
    counted = regex.count_matches(pattern, 0, shortest, max_dfa_states=limits.max_dfa_states)
    assert isinstance(counted, int) and counted >= limit, (pattern, shortest, counted)
