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

    Groups are flattened, single-character classes such as `[j]` become literals, and a class of the two cases of one
    letter such as `[jJ]` becomes that letter in lowercase (matched case-insensitively by callers). Anchors are
    dropped, and a repeat with a fixed count of an exact string becomes that string repeated. A repeat that may occur
    zero times, a class of several characters, or an alternation ends a run; an alternation contributes the runs every
    branch shares. A regex that does not parse requires itself.
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


def _exact_class(items: Sequence[tuple[Any, Any]]) -> str | None:
    """Returns the literal a character class always matches, or None if it can match more than one character."""
    if any(op.name == "NEGATE" for op, _ in items):
        return None
    chars: list[str] = []
    for op, av in items:
        if op.name == "LITERAL":
            chars.append(chr(av))
        else:
            return None
    if len(chars) == 1:
        return chars[0]
    if len(chars) == 2 and chars[0].isalpha() and chars[0].lower() == chars[1].lower():
        return chars[0].lower()
    return None


def _exact(item: tuple[Any, Any]) -> str | None:
    op, av = item
    name = op.name
    if name == "LITERAL":
        return chr(av)
    if name in ("AT", "ASSERT", "ASSERT_NOT"):
        return ""
    if name == "IN":
        return _exact_class(av)
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


def is_generalized(regex: str) -> bool:
    """Returns whether `regex` is literal-free after normalization (spec 0.2.0 sensitive-column patterns).

    Groups and anchors are ignored. Single-character classes and case-only classes such as `[jJ]` count as literals.
    Any literal character, alternation, or back-reference fails. Allowed atoms are character classes with at least two
    distinct case-folded members, `.`, `\\d` / `\\w` / `\\s` (and their negations), with any quantifier.
    """
    try:
        items = _parser.parse(regex)
    except re.error:
        return False
    try:
        _assert_generalized(items)
    except _NotGeneralized:
        return False
    return True


class _NotGeneralized(Exception):
    """Internal: the pattern embeds a literal, alternation, or back-reference."""


def _assert_generalized(items: Sequence[tuple[Any, Any]]) -> None:
    for op, av in items:
        name = op.name
        if name in ("AT",):
            continue
        if name in ("SUBPATTERN", "ATOMIC_GROUP"):
            _assert_generalized(av[-1])
            continue
        if name in REPEATS:
            _assert_generalized(av[2])
            continue
        if name == "ANY":
            continue
        if name == "CATEGORY" and _category_allowed(av.name):
            continue
        if name == "IN" and _class_generalized(av):
            continue
        if name in ("BRANCH", "LITERAL", "NOT_LITERAL", "GROUPREF", "GROUPREF_EXISTS", "ASSERT", "ASSERT_NOT"):
            raise _NotGeneralized()
        raise _NotGeneralized()


def _category_allowed(name: str) -> bool:
    base = name.replace("CATEGORY_NOT_", "CATEGORY_") if name.startswith("CATEGORY_NOT_") else name
    return base in ("CATEGORY_DIGIT", "CATEGORY_WORD", "CATEGORY_SPACE")


def _class_generalized(items: Sequence[tuple[Any, Any]]) -> bool:
    """True when a character class is a broad set, not a folded literal."""
    members: set[str] = set()
    broad = False
    for op, av in items:
        name = op.name
        if name == "NEGATE":
            continue
        if name == "CATEGORY" and _category_allowed(av.name):
            broad = True
        elif name == "RANGE":
            if av[1] - av[0] >= 1:
                broad = True
            else:
                members.add(_fold_char(chr(av[0])))
        elif name == "LITERAL":
            members.add(_fold_char(chr(av)))
        else:
            return False
    return broad or len(members) >= 2


def _fold_char(char: str) -> str:
    return char.lower() if char.isalpha() else char


def expand_branches(
    regex: str, max_dfa_states: int = DEFAULT_MAX_DFA_STATES, budget: _WorkBudget | None = None
) -> list[str] | _BudgetExceeded:
    """Returns one regex per alternation branch of `regex`, charging `budget` (created from `max_dfa_states` if omitted).

    Nested alternations are expanded into a cartesian product. Exhausting the shared work budget returns
    `BUDGET_EXCEEDED`. A pattern with no alternation returns a one-element list containing `regex` itself when the
    expansion equals the original parse, otherwise the serialized form of the normalized items.
    """
    try:
        items = _parser.parse(regex)
    except re.error as error:
        raise UnsupportedForCount(str(error)) from error
    work = budget if budget is not None else _WorkBudget(max_dfa_states)
    try:
        work.charge(1)
        expanded = _expand_items(list(items), work)
        return [_items_to_regex(branch) for branch in expanded]
    except _Exhausted:
        return BUDGET_EXCEEDED


def _expand_items(items: list[tuple[Any, Any]], budget: _WorkBudget) -> list[list[tuple[Any, Any]]]:
    results: list[list[tuple[Any, Any]]] = [[]]
    for op, av in items:
        budget.charge(1)
        name = op.name
        if name in ("SUBPATTERN", "ATOMIC_GROUP"):
            inners = _expand_items(list(av[-1]), budget)
            results = [prefix + inner for prefix in results for inner in inners]
        elif name == "BRANCH":
            alts: list[list[tuple[Any, Any]]] = []
            for branch in av[1]:
                alts.extend(_expand_items(list(branch), budget))
            results = [prefix + alt for prefix in results for alt in alts]
        elif name in REPEATS:
            inners = _expand_items(list(av[2]), budget)
            results = [[*prefix, (op, (av[0], av[1], inner))] for prefix in results for inner in inners]
        else:
            results = [[*prefix, (op, av)] for prefix in results]
        budget.charge(len(results))
        if len(results) > budget.max_states:
            raise _Exhausted()
    return results


def _items_to_regex(items: Sequence[tuple[Any, Any]]) -> str:
    parts: list[str] = []
    for op, av in items:
        name = op.name
        if name == "LITERAL":
            parts.append(re.escape(chr(av)))
        elif name == "NOT_LITERAL":
            parts.append(f"[^{re.escape(chr(av))}]")
        elif name == "ANY":
            parts.append(".")
        elif name == "IN":
            parts.append(_class_to_regex(av))
        elif name == "CATEGORY":
            parts.append(_category_to_regex(av.name))
        elif name in ("SUBPATTERN", "ATOMIC_GROUP"):
            parts.append(f"(?:{_items_to_regex(av[-1])})")
        elif name == "BRANCH":
            alts = "|".join(f"(?:{_items_to_regex(branch)})" for branch in av[1])
            parts.append(f"(?:{alts})")
        elif name in REPEATS:
            inner = _items_to_regex(av[2])
            wrapped = inner if len(av[2]) == 1 and av[2][0][0].name in ("IN", "ANY", "CATEGORY") else f"(?:{inner})"
            parts.append(wrapped + _quantifier(av[0], av[1]))
        elif name == "AT":
            continue
        else:
            raise UnsupportedForCount(f"cannot serialize {name}")
    return "".join(parts)


def _quantifier(low: int, high: int) -> str:
    if low == 0 and high == _parser.MAXREPEAT:
        return "*"
    if low == 1 and high == _parser.MAXREPEAT:
        return "+"
    if low == 0 and high == 1:
        return "?"
    if high == _parser.MAXREPEAT:
        return f"{{{low},}}"
    if low == high:
        return f"{{{low}}}"
    return f"{{{low},{high}}}"


def _category_to_regex(name: str) -> str:
    return {
        "CATEGORY_DIGIT": r"\d",
        "CATEGORY_NOT_DIGIT": r"\D",
        "CATEGORY_WORD": r"\w",
        "CATEGORY_NOT_WORD": r"\W",
        "CATEGORY_SPACE": r"\s",
        "CATEGORY_NOT_SPACE": r"\S",
    }.get(name, ".")


def _class_to_regex(items: Sequence[tuple[Any, Any]]) -> str:
    negate = ""
    body: list[str] = []
    for op, av in items:
        name = op.name
        if name == "NEGATE":
            negate = "^"
        elif name == "LITERAL":
            body.append(re.escape(chr(av)))
        elif name == "RANGE":
            body.append(f"{re.escape(chr(av[0]))}-{re.escape(chr(av[1]))}")
        elif name == "CATEGORY":
            body.append(_category_to_regex(av.name))
        else:
            raise UnsupportedForCount(f"cannot serialize class {name}")
    return f"[{negate}{''.join(body)}]"


# --- Allowlisted portable dialect (raw-source tokenizer; not sre_parse) ---

_DIALECT_CATEGORIES = frozenset("dDwWsS")
_DIALECT_META = frozenset(".^$*+?{}[]\\|()")
_DIALECT_ESCAPE_LITERALS = frozenset(".^$*+?{}[]\\|()-")


class _DialectError(Exception):
    """The pattern is outside the allowlisted portable dialect."""


def in_dialect(source: str) -> bool:
    """Returns whether `source` is exactly one pattern in the allowlisted portable dialect.

    The check uses a strict tokenizer over the raw string (not Python's `sre_parse`). Allowed forms are literals and
    escaped literals, classes `[...]` (including negation and ranges), `.`, `\\d`/`\\D`/`\\w`/`\\W`/`\\s`/`\\S`,
    quantifiers `?`/`*`/`+`/`{m}`/`{m,}`/`{m,n}` (no lazy forms), groups `(...)` and `(?:...)`, alternation, and
    anchors `^`/`$`. Comments, inline flags, lookarounds, named groups, backreferences, `\\p`/`\\N`/`\\x`/`\\u`,
    verbose whitespace, conditionals, and NUL/control characters fail. The source must equal the concatenation of the
    token texts so ignored or hidden bytes cannot carry data.
    """
    if any(ord(c) < 32 or ord(c) == 127 for c in source):
        return False
    try:
        tokens, pos = _dialect_disjunction(source, 0)
    except _DialectError:
        return False
    return pos == len(source) and "".join(tokens) == source


def _dialect_disjunction(source: str, pos: int) -> tuple[list[str], int]:
    tokens, pos = _dialect_term(source, pos)
    while pos < len(source) and source[pos] == "|":
        tokens.append("|")
        pos += 1
        more, pos = _dialect_term(source, pos)
        tokens.extend(more)
    return tokens, pos


def _dialect_term(source: str, pos: int) -> tuple[list[str], int]:
    tokens: list[str] = []
    while pos < len(source) and source[pos] not in ")|":
        atom, pos = _dialect_atom(source, pos)
        tokens.extend(atom)
        if pos < len(source) and source[pos] in "?*+{":
            quant, pos = _dialect_quantifier(source, pos)
            tokens.append(quant)
            if pos < len(source) and source[pos] == "?":
                raise _DialectError("lazy quantifier")
    return tokens, pos


def _dialect_atom(source: str, pos: int) -> tuple[list[str], int]:
    if pos >= len(source):
        raise _DialectError("truncated")
    char = source[pos]
    if char in " \t\r\n\f\v":
        raise _DialectError("whitespace")
    if char in "^$.":
        return [char], pos + 1
    if char == "\\":
        return _dialect_escape(source, pos, in_class=False)
    if char == "[":
        return _dialect_class(source, pos)
    if char == "(":
        return _dialect_group(source, pos)
    if char in _DIALECT_META:
        raise _DialectError(f"bare metacharacter {char}")
    return [char], pos + 1


def _dialect_escape(source: str, pos: int, *, in_class: bool) -> tuple[list[str], int]:
    if pos + 1 >= len(source):
        raise _DialectError("trailing backslash")
    nxt = source[pos + 1]
    if ord(nxt) < 32 or ord(nxt) == 127:
        raise _DialectError("control escape")
    if nxt in _DIALECT_CATEGORIES:
        return [source[pos : pos + 2]], pos + 2
    if nxt in _DIALECT_ESCAPE_LITERALS or (in_class and nxt in "^-]"):
        return [source[pos : pos + 2]], pos + 2
    raise _DialectError(f"unsupported escape \\{nxt}")


def _dialect_class(source: str, pos: int) -> tuple[list[str], int]:
    start = pos
    pos += 1
    if pos < len(source) and source[pos] == "^":
        pos += 1
    if pos >= len(source):
        raise _DialectError("unterminated class")
    # First body character may be ] or - without closing/range meaning.
    if source[pos] in "]-":
        pos += 1
    while pos < len(source) and source[pos] != "]":
        if source[pos] == "\\":
            _, pos = _dialect_escape(source, pos, in_class=True)
            continue
        if source[pos] == "[":
            raise _DialectError("nested class or POSIX class")
        if ord(source[pos]) < 32 or ord(source[pos]) == 127:
            raise _DialectError("control in class")
        pos += 1
    if pos >= len(source) or source[pos] != "]":
        raise _DialectError("unterminated class")
    return [source[start : pos + 1]], pos + 1


def _dialect_group(source: str, pos: int) -> tuple[list[str], int]:
    start = pos
    pos += 1
    if pos < len(source) and source[pos] == "?":
        if pos + 1 < len(source) and source[pos + 1] == ":":
            pos += 2
        else:
            raise _DialectError("unsupported group extension")
    _, pos = _dialect_disjunction(source, pos)
    if pos >= len(source) or source[pos] != ")":
        raise _DialectError("unterminated group")
    return [source[start : pos + 1]], pos + 1


def _dialect_quantifier(source: str, pos: int) -> tuple[str, int]:
    char = source[pos]
    if char in "?*+":
        return char, pos + 1
    # {m}, {m,}, {m,n}
    if char != "{":
        raise _DialectError("quantifier")
    start = pos
    pos += 1
    if pos >= len(source) or not source[pos].isdigit():
        raise _DialectError("braced quantifier")
    while pos < len(source) and source[pos].isdigit():
        pos += 1
    if pos < len(source) and source[pos] == ",":
        pos += 1
        while pos < len(source) and source[pos].isdigit():
            pos += 1
    if pos >= len(source) or source[pos] != "}":
        raise _DialectError("braced quantifier")
    return source[start : pos + 1], pos + 1
