"""Regex helpers: random strings matching a regex, and the literal text a regex embeds.

Both walk the parse tree of Python's `re` module. Generation supports literals, character classes, `.`, groups,
alternation, and repeats; lookarounds and back-references are not supported.
"""

from __future__ import annotations

import importlib
import re
import string
import sys
from collections.abc import Sequence
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
COUNT_CAP = 10**12
WIDE_RANGE = 256
DEFAULT_MAX_DFA_STATES = 10_000
DEFAULT_MAX_REGEX_LENGTH = 1_000
# Work units allotted per allowed automaton state; NFA build, epsilon-closure, DFA steps, and the length loop share it.
WORK_PER_STATE = 64
CATEGORY_CHARS = {
    "CATEGORY_DIGIT": string.digits,
    "CATEGORY_WORD": string.ascii_letters + string.digits + "_",
    "CATEGORY_SPACE": " \t\n\r\f\v",
}

CharSet = tuple[bool, frozenset[str]]


class UnsupportedForCount(ValueError):
    """The regex uses a construct (such as a back-reference) whose language cannot be counted."""


class _BudgetExceeded:
    """Sentinel returned when language counting exhausts its shared work budget."""

    def __repr__(self) -> str:
        return "BUDGET_EXCEEDED"


BUDGET_EXCEEDED = _BudgetExceeded()


class _Exhausted(Exception):
    """Internal: the shared work budget was exhausted."""


class _WorkBudget:
    """Single counter for NFA construction, epsilon-closure, DFA subset construction, and length iteration."""

    def __init__(self, max_states: int) -> None:
        self.max_states = max_states
        self.limit = max_states * WORK_PER_STATE
        self.used = 0
        self.nfa_states = 0
        self.dfa_states = 0

    def charge(self, amount: int = 1) -> None:
        self.used += amount
        if self.used > self.limit:
            raise _Exhausted()


def count_matches(
    regex: str,
    min_length: int,
    max_length: int,
    cap: int = COUNT_CAP,
    max_dfa_states: int = DEFAULT_MAX_DFA_STATES,
    budget: _WorkBudget | None = None,
) -> int | _BudgetExceeded:
    """Returns how many distinct strings with a length in [min_length, max_length] fully match `regex`, at most `cap`.

    The regex is compiled to an automaton over character classes and determinized, so each string is one path and
    syntactic alternatives that match the same strings are not counted twice. Characters outside the regex's explicit
    characters count as one class of 2**16 minus those; a range wider than 256 characters, `.`, and lookarounds are
    widened to every character, which can only overcount. NFA construction (including expanded repeats), epsilon
    closure, DFA subset construction, and every length-loop iteration charge one shared `_WorkBudget` sized from
    `max_dfa_states`; when it is exhausted the sentinel `BUDGET_EXCEEDED` is returned.

    Raises:
        UnsupportedForCount: If the regex does not parse or uses back-references.
    """
    try:
        items = _parser.parse(regex)
    except re.error as error:
        raise UnsupportedForCount(str(error)) from error
    work = budget if budget is not None else _WorkBudget(max_dfa_states)
    try:
        work.charge(1)
        nfa = _Nfa(max_length, work)
        start = nfa.state()
        end = nfa.build(items, start)
        classes = _classes(nfa.sets)
        return _count(nfa, start, end, classes, min_length, max_length, cap, work)
    except _Exhausted:
        return BUDGET_EXCEEDED


class _Nfa:
    def __init__(self, max_length: int, budget: _WorkBudget) -> None:
        self.max_length = max_length
        self.budget = budget
        self.epsilon: list[list[int]] = []
        self.edges: list[list[tuple[int, int]]] = []
        self.sets: list[CharSet] = []

    def state(self) -> int:
        self.budget.charge(1)
        self.budget.nfa_states += 1
        if self.budget.nfa_states > self.budget.max_states:
            raise _Exhausted()
        self.epsilon.append([])
        self.edges.append([])
        return len(self.epsilon) - 1

    def char(self, start: int, charset: CharSet) -> int:
        end = self.state()
        self.sets.append(charset)
        self.edges[start].append((len(self.sets) - 1, end))
        return end

    def build(self, items: Sequence[tuple[Any, Any]], start: int) -> int:
        current = start
        for op, av in items:
            current = self._item((op, av), current)
        return current

    def _item(self, item: tuple[Any, Any], start: int) -> int:
        self.budget.charge(1)
        op, av = item
        name = op.name
        if name in ("AT", "ASSERT", "ASSERT_NOT"):
            return start
        if name in ("SUBPATTERN", "ATOMIC_GROUP"):
            return self.build(av[-1], start)
        if name == "BRANCH":
            end = self.state()
            for branch in av[1]:
                self.epsilon[self.build(branch, start)].append(end)
            return end
        if name in REPEATS:
            return self._repeat(av, start)
        charset = _charset(item)
        if charset is None:
            raise UnsupportedForCount(f"cannot count {name}")
        return self.char(start, charset)

    def _repeat(self, av: tuple[Any, Any, Any], start: int) -> int:
        low, high, inner = av
        # Charge the syntactic repeat size before truncating to max_length so a{100000} fails without building it.
        span = low if high == _parser.MAXREPEAT else high
        self.budget.charge(max(span, 1))
        low = min(low, self.max_length + 1)
        current = start
        for _ in range(low):
            current = self.build(inner, current)
        if high == _parser.MAXREPEAT or high > self.max_length:
            loop_end = self.build(inner, current)
            self.epsilon[loop_end].append(current)
            return current
        end = self.state()
        self.epsilon[current].append(end)
        for _ in range(high - low):
            current = self.build(inner, current)
            self.epsilon[current].append(end)
        return end


def _charset(item: tuple[Any, Any]) -> CharSet | None:
    op, av = item
    name = op.name
    if name == "LITERAL":
        return False, frozenset(chr(av))
    if name == "NOT_LITERAL":
        return True, frozenset(chr(av))
    if name == "ANY":
        return True, frozenset()
    if name == "CATEGORY":
        return _category_set(av.name)
    if name == "IN":
        return _class_set(av)
    return None


def _category_set(name: str) -> CharSet:
    negated = name.startswith("CATEGORY_NOT_")
    return negated, frozenset(CATEGORY_CHARS.get(name.replace("_NOT", ""), ""))


def _class_set(items: Sequence[tuple[Any, Any]]) -> CharSet:
    members: set[str] = set()
    excluded: set[str] | None = None
    negate = False
    for op, av in items:
        name = op.name
        if name == "NEGATE":
            negate = True
        elif name == "LITERAL":
            members.add(chr(av))
        elif name == "RANGE":
            if av[1] - av[0] >= WIDE_RANGE:
                excluded = set()
            else:
                members.update(chr(c) for c in range(av[0], av[1] + 1))
        elif name == "CATEGORY":
            category_negated, chars = _category_set(av.name)
            if category_negated:
                excluded = (excluded or set(chars)) & set(chars)
            else:
                members.update(chars)
        else:
            excluded = set()
    if excluded is not None:
        result: CharSet = (True, frozenset(excluded - members))
    else:
        result = (False, frozenset(members))
    return (not result[0], result[1]) if negate else result


def _classes(sets: Sequence[CharSet]) -> list[tuple[int, tuple[bool, ...]]]:
    """Returns each class of characters that every set treats alike, as its size and per-set membership."""
    explicit = sorted({c for _, chars in sets for c in chars})
    groups: dict[tuple[bool, ...], int] = {}
    for c in explicit:
        signature = tuple((c in chars) != negated for negated, chars in sets)
        groups[signature] = groups.get(signature, 0) + 1
    rest = tuple(negated for negated, _ in sets)
    groups[rest] = groups.get(rest, 0) + UNICODE - len(explicit)
    return [(size, signature) for signature, size in groups.items() if size > 0]


def _closure(nfa: _Nfa, states: set[int], budget: _WorkBudget) -> frozenset[int]:
    stack, seen = list(states), set(states)
    while stack:
        budget.charge(1)
        for target in nfa.epsilon[stack.pop()]:
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return frozenset(seen)


def _count(
    nfa: _Nfa,
    start: int,
    end: int,
    classes: Sequence[tuple[int, tuple[bool, ...]]],
    min_length: int,
    max_length: int,
    cap: int,
    budget: _WorkBudget,
) -> int | _BudgetExceeded:
    transitions: dict[frozenset[int], list[tuple[frozenset[int], int]]] = {}

    def step(state: frozenset[int]) -> list[tuple[frozenset[int], int]]:
        if state in transitions:
            return transitions[state]
        budget.dfa_states += 1
        if budget.dfa_states > budget.max_states:
            raise _Exhausted()
        budget.charge(1)
        moves: dict[frozenset[int], int] = {}
        for size, member in classes:
            budget.charge(max(1, len(state)))
            targets = {t for s in state for setid, t in nfa.edges[s] if member[setid]}
            if targets:
                target = _closure(nfa, targets, budget)
                moves[target] = moves.get(target, 0) + size
        transitions[state] = list(moves.items())
        return transitions[state]

    counts = {_closure(nfa, {start}, budget): 1}
    total = 0
    for length in range(max_length + 1):
        budget.charge(1)
        if length >= min_length:
            total += sum(n for state, n in counts.items() if end in state)
            if total >= cap:
                return cap
        following: dict[frozenset[int], int] = {}
        for state, n in counts.items():
            budget.charge(1)
            for target, size in step(state):
                following[target] = min(cap, following.get(target, 0) + n * size)
        counts = following
        if not counts:
            break
    return min(total, cap)


def required_literals(regex: str) -> list[str]:
    """Returns literal strings that every full match of `regex` contains, longest runs first.

    Groups are flattened, single-character classes such as `[j]` become literals, anchors are dropped, and a repeat
    with a fixed count of an exact string becomes that string repeated. A repeat that may occur zero times, a class
    of several characters, or an alternation ends a run; an alternation contributes the runs every branch shares.
    A regex that does not parse requires itself.
    """
    try:
        items = _parser.parse(regex)
    except re.error:
        return [regex]
    return sorted({f for f in _required(items) if f}, key=lambda f: (-len(f), f))


def _flatten(items: Sequence[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    flat: list[tuple[Any, Any]] = []
    for op, av in items:
        if op.name in ("SUBPATTERN", "ATOMIC_GROUP"):
            flat += _flatten(av[-1])
        else:
            flat.append((op, av))
    return flat


def _exact(item: tuple[Any, Any]) -> str | None:
    op, av = item
    name = op.name
    if name == "LITERAL":
        return chr(av)
    if name in ("AT", "ASSERT", "ASSERT_NOT"):
        return ""
    if name == "IN" and len(av) == 1 and av[0][0].name == "LITERAL":
        return chr(av[0][1])
    if name == "BRANCH":
        options = {_exact_sequence(branch) for branch in av[1]}
        return options.pop() if len(options) == 1 and None not in options else None
    if name in REPEATS and av[0] == av[1]:
        inner = _exact_sequence(av[2])
        return inner * av[0] if inner is not None else None
    return None


def _exact_sequence(items: Sequence[tuple[Any, Any]]) -> str | None:
    parts = [_exact(item) for item in _flatten(items)]
    return None if any(p is None for p in parts) else "".join(p for p in parts if p is not None)


def _required(items: Sequence[tuple[Any, Any]]) -> list[str]:
    factors: list[str] = []
    run = ""
    for op, av in _flatten(items):
        literal = _exact((op, av))
        if literal is not None:
            run += literal
            continue
        factors.append(run)
        run = ""
        if op.name in REPEATS and av[0] >= 1:
            factors += _required(av[2])
        elif op.name == "BRANCH":
            branches = [_required(branch) for branch in av[1]]
            factors += [f for f in branches[0] if all(any(f in g for g in other) for other in branches[1:])]
    factors.append(run)
    return factors
