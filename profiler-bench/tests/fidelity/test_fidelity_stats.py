import re
import time

import numpy as np
import pandas as pd
import pytest

from profiler_bench.fidelity import regex, stats


def test_identical_samples_have_zero_distance() -> None:
    values = np.random.default_rng(0).normal(size=1000)
    assert stats.ks_wasserstein(values, values.copy()) == (0.0, 0.0)


def test_shifted_samples_move_both_distances() -> None:
    values = np.random.default_rng(0).normal(size=1000)
    near = stats.ks_wasserstein(values, values + 0.1)
    far = stats.ks_wasserstein(values, values + 1.0)
    assert 0 < near[0] < far[0] <= 1
    assert 0 < near[1] < far[1] <= 1


def test_wasserstein_is_bounded_for_extreme_values() -> None:
    real = np.array([-1.7e308, 0.0, 1.0, 1.7e308])
    ks, w1 = stats.ks_wasserstein(real, np.array([1e300, 1e300]))
    assert 0 <= ks <= 1
    assert 0 <= w1 <= 1


def test_constant_columns_use_exact_equality() -> None:
    assert stats.ks_wasserstein(np.ones(5), np.ones(3)) == (0.0, 0.0)
    assert stats.ks_wasserstein(np.ones(5), np.full(3, 2.0)) == (1.0, 1.0)


def test_weighted_distances_match_expanded_samples() -> None:
    expanded = stats.ks_wasserstein(np.array([1.0, 1.0, 2.0]), np.array([2.0, 3.0]))
    weighted = stats.ks_wasserstein(np.array([1.0, 2.0]), np.array([2.0, 3.0]), np.array([2.0, 1.0]), np.ones(2))
    assert np.allclose(expanded, weighted)


def test_tvd_counts_unseen_values_as_other() -> None:
    real = pd.Series(["a"] * 6 + ["b"] * 4)
    assert stats.tvd(real, real, ["a", "b"]) == 0
    assert stats.tvd(real, pd.Series(["c"] * 10), ["a", "b"]) == 1


def test_auc_counts_ties_as_half() -> None:
    assert stats.auc(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5
    assert stats.auc(np.array([0.1, 0.9]), np.array([0, 1])) == 1.0


def test_discriminator_cannot_tell_identical_samples_apart() -> None:
    x = np.random.default_rng(0).normal(size=(400, 3))
    assert stats.discriminator_auc(x, x.copy(), np.zeros(3, dtype=bool), 0, 50) == 0.5


def test_discriminator_separates_shifted_samples() -> None:
    rng = np.random.default_rng(0)
    real, synth = rng.normal(size=(400, 2)), rng.normal(size=(400, 2)) + 3
    assert stats.discriminator_auc(real, synth, np.zeros(2, dtype=bool), 0, 50) > 0.95


def test_discriminator_uses_categorical_codes() -> None:
    rng = np.random.default_rng(0)
    real, synth = rng.integers(0, 3, (400, 1)).astype(float), rng.integers(3, 6, (400, 1)).astype(float)
    assert stats.discriminator_auc(real, synth, np.ones(1, dtype=bool), 0, 50) > 0.95


def test_generated_strings_match_their_regex() -> None:
    pattern = r"[A-Z]{3}-\d{4}(x|yy)?[^a-z]\w+"
    values = regex.generate(pattern, 100, np.random.default_rng(0))
    assert all(re.fullmatch(pattern, v) for v in values)


def test_literals_split_alternatives() -> None:
    assert regex.literals(r"john\.doe@(example\.com|corp)") == ["john.doe@", "example.com", "corp"]


@pytest.mark.parametrize(
    ("pattern", "count"),
    [
        (r"(j)(o)(h)(n)@x\.com", 1),
        (r"[j][o]hn", 1),
        (r"(j|J)(o)(h)(n)", 2),
        (r"(j|j)ohn", 1),
        (r"a|aa|a{1,2}", 2),
        (r"[A-Z]{3}-\d{4}", 26**3 * 10**4),
        (r"[ab]{1,2}", 6),
        (r"[a-z]+", regex.COUNT_CAP),
    ],
)
def test_count_matches_counts_distinct_strings(pattern: str, count: int) -> None:
    assert regex.count_matches(pattern, 0, 40) == count


def test_count_matches_respects_length_bounds() -> None:
    assert regex.count_matches(r"[ab]+", 2, 3) == 4 + 8


def test_count_matches_returns_budget_sentinel_deterministically() -> None:
    pattern = r"(a|b)*a(a|b){25}"
    assert regex.count_matches(pattern, 0, 64, max_dfa_states=100) is regex.BUDGET_EXCEEDED
    assert regex.count_matches(pattern, 0, 64, max_dfa_states=100) is regex.BUDGET_EXCEEDED
    assert regex.count_matches(r"[A-Z]{3}-\d{4}", 0, 40) == 26**3 * 10**4


def test_count_matches_budget_covers_nfa_and_length_loop() -> None:
    started = time.perf_counter()
    assert regex.count_matches(r"a{100000}", 0, 100000) is regex.BUDGET_EXCEEDED
    assert regex.count_matches(r"(a{1000}){1000}", 0, 1000000) is regex.BUDGET_EXCEEDED
    assert time.perf_counter() - started < 1.0
    assert regex.count_matches(r"[A-Z]{3}-\d{4}", 0, 8) == 26**3 * 10**4
    email_count = regex.count_matches(r"[a-z]+\.[a-z]+[0-9]+@(example\.com|mail\.test)", 0, 40)
    assert isinstance(email_count, int) and email_count >= 1


def test_every_count_entry_point_takes_the_budget() -> None:
    empty = regex._WorkBudget(0)
    assert regex.count_matches("a", 0, 1, budget=empty) is regex.BUDGET_EXCEEDED
    spent = regex._WorkBudget(10_000)
    assert regex.count_matches("a", 0, 1, budget=spent) == 1
    assert spent.used > 0
    assert spent.nfa_states > 0
    assert spent.dfa_states > 0


@pytest.mark.parametrize(
    ("pattern", "required"),
    [
        (r"(j)(o)(h)(n)[a-z]{0,20}@x\.com", ["@x.com", "john"]),
        (r"^[j][o](h)n$", ["john"]),
        (r"(j|J)(o)(h)(n)", ["john"]),
        (r"(ab){2}c|xababc", ["ababc"]),
        (r"[A-Z]{3}-\d{4}", ["-"]),
        (r"x?(abcd)+", ["abcd"]),
        (r"[jJ][oO][hH][nN][@][xX][.][cC][oO][mM]", ["john@x.com"]),
        (r"[jJ][oO][hH][nN]@x\.com.*", ["john@x.com"]),
    ],
)
def test_required_literals(pattern: str, required: list[str]) -> None:
    assert regex.required_literals(pattern) == required
