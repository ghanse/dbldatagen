import itertools
import math
import re
import time

import numpy as np
import pandas as pd
import pytest

from profiler_bench.fidelity import regex, stats
from profiler_bench.fidelity.scores import log_loss_gap


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


def _discriminate(real: np.ndarray, synth: np.ndarray, categorical: np.ndarray | None = None) -> stats.Discrimination:
    rows = [pd.Series(np.arange(len(real)))]
    folds = (stats.entity_folds(rows, 0), stats.entity_folds([pd.Series(np.arange(len(synth)))], 0))
    flags = np.zeros(real.shape[1], dtype=bool) if categorical is None else categorical
    result = stats.discriminate(real, synth, flags, folds, 0, 100, 50, 10)
    assert result is not None
    return result


def test_discriminator_cannot_tell_identical_samples_apart() -> None:
    x = np.random.default_rng(0).normal(size=(2000, 3))
    result = _discriminate(x, x.copy())
    assert result.auc == 0.5
    assert result.log_loss == pytest.approx(math.log(2))


def test_discriminator_on_two_samples_of_one_distribution_stays_near_ln2() -> None:
    rng = np.random.default_rng(0)
    losses = [_discriminate(rng.normal(size=(2000, 3)), rng.normal(size=(2000, 3))).log_loss for _ in range(3)]
    assert all(abs(loss - math.log(2)) < 0.02 for loss in losses)


def test_log_loss_gap_is_monotone_and_smooth_in_the_shift() -> None:
    rng = np.random.default_rng(0)
    real = rng.normal(size=(2000, 2))
    shifts = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    results = [_discriminate(real, rng.normal(size=(2000, 2)) + np.array([shift, 0.0])) for shift in shifts]
    gaps = [log_loss_gap(r.log_loss, math.log(2)) for r in results]
    assert gaps[0] < 0.02 and gaps[-1] > 0.99
    assert all(b >= a - 0.02 for a, b in itertools.pairwise(gaps))
    assert max(b - a for a, b in itertools.pairwise(gaps)) < 0.5
    saturated = [r for r in results if r.auc > 0.999]
    assert len({round(log_loss_gap(r.log_loss, math.log(2)), 3) for r in saturated}) > 1


def test_discriminator_separates_shifted_samples_and_stores_only_a_histogram() -> None:
    rng = np.random.default_rng(0)
    result = _discriminate(rng.normal(size=(2000, 2)), rng.normal(size=(2000, 2)) + 3)
    assert result.auc > 0.95
    assert set(result.histogram) == {"real", "synth"}
    assert all(len(counts) == 10 and sum(counts) == result.n_test for counts in result.histogram.values())


def test_discriminator_uses_categorical_codes() -> None:
    rng = np.random.default_rng(0)
    real, synth = rng.integers(0, 3, (2000, 1)).astype(float), rng.integers(3, 6, (2000, 1)).astype(float)
    assert _discriminate(real, synth, np.ones(1, dtype=bool)).auc > 0.95


def test_entity_folds_keep_every_entity_on_one_side() -> None:
    rng = np.random.default_rng(0)
    orders, products = pd.Series(rng.integers(0, 300, 5000)), pd.Series(rng.integers(0, 40, 5000))
    folds = stats.entity_folds([products, orders], 7)
    train = folds == stats.FIT
    train |= folds == stats.VALIDATION
    test = folds == stats.TEST
    for entity in (orders, products):
        assert not set(entity[train]) & set(entity[test])
    assert set(products[folds == stats.VALIDATION]).isdisjoint(products[folds == stats.FIT])
    assert 0.35 < (folds == stats.DROPPED).mean() < 0.65
    assert (stats.entity_folds([products.copy(), orders.copy()], 7) == folds).all()


def test_log_loss_gap_reference_and_clipping() -> None:
    assert log_loss_gap(0.5, 0.5) == 0.0
    assert log_loss_gap(0.9, 0.5) == 0.0
    assert log_loss_gap(0.0, 0.5) == 1.0
    assert log_loss_gap(0.314, 0.232) == 0.0
    assert log_loss_gap(0.25, 0.5) == pytest.approx(0.5)
    assert log_loss_gap(math.log(2) / 2, 1.2) == pytest.approx(0.5)


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
    ("pattern", "generalized"),
    [
        (r"[^\s]+", True),
        (r"[a-z0-9]+", True),
        (r".+", True),
        (r"[a-z0-9]{3,12}@[a-z]{2,8}\.[a-z]{2,3}", False),
        (r"[jJ][oO][hH][nN]@x\.com.*", False),
        (r"(?:(j)(o)(h)(n)[@](x)[.](c)(o)(m)|[a-z0-9@.]{10})", False),
    ],
)
def test_is_generalized(pattern: str, generalized: bool) -> None:
    assert regex.is_generalized(pattern) is generalized


@pytest.mark.parametrize(
    ("pattern", "allowed"),
    [
        (r"[^\s]+", True),
        (r"[A-Z]{3}-\d{4}", True),
        (r"\d{3}-\d{4}", True),
        (r"(?#john@x.com)[^\s]+", False),
        (r"(?i)JOHN@X\.COM", False),
        (r"(?=john)[^\s]+", False),
        (r"(?P<john_x_com>a)", False),
        (r"a*?", False),
        (r"\x41", False),
    ],
)
def test_in_dialect(pattern: str, allowed: bool) -> None:
    assert regex.in_dialect(pattern) is allowed


def test_expand_branches_splits_decoy_alternation() -> None:
    branches = regex.expand_branches(r"(?:(j)(o)(h)(n)[@](x)[.](c)(o)(m)|[a-z0-9@.]{10})")
    assert isinstance(branches, list) and len(branches) == 2
    assert regex.required_literals(branches[0]) == ["john@x.com"]
    assert regex.count_matches(branches[0], 0, 10) == 1


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
