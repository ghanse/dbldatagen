import re

import numpy as np
import pandas as pd

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
