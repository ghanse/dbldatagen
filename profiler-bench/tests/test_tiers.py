import math

import pytest

from profiler_bench.catalog import DATASETS, TIERS, generate
from profiler_bench.catalog.registry import TIER_FILES


def _rows(name: str, tier: str) -> dict[str, int]:
    tables, truth = generate(DATASETS[name], tier, 0)
    assert {t.name: t.row_count for t in truth.tables} == {n: t.num_rows for n, t in tables.items()}
    return {n: t.num_rows for n, t in tables.items()}


def _sizes(name: str, tier: str) -> dict[str, float]:
    sizes = DATASETS[name].sizes
    assert sizes is not None
    return sizes(TIERS[tier])


def test_tiers_are_ordered() -> None:
    assert list(TIERS) == ["xs", "s", "m"]
    assert TIERS["xs"] < TIERS["s"] < TIERS["m"]
    assert list(TIER_FILES) == list(TIERS)


@pytest.mark.parametrize("name", sorted(DATASETS))
@pytest.mark.parametrize("tier", ["xs", "s"])
def test_generated_rows_match_the_size_formulas(name: str, tier: str) -> None:
    rows, expected = _rows(name, tier), _sizes(name, tier)
    assert set(rows) == set(expected)
    for table, count in rows.items():
        assert count == pytest.approx(expected[table], abs=5 * math.sqrt(expected[table]))


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_m_tier_size_formulas_scale(name: str) -> None:
    xs, s, m = (_sizes(name, tier) for tier in TIERS)
    assert max(m.values()) == pytest.approx(TIERS["m"], rel=0.05)
    assert all(xs[t] <= s[t] <= m[t] for t in m)
    assert sum(m.values()) > 10 * sum(s.values())


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_largest_table_tracks_the_tier_size(name: str) -> None:
    for tier in ("xs", "s"):
        assert max(_rows(name, tier).values()) == pytest.approx(TIERS[tier], rel=0.05)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(DATASETS))
def test_m_tier_scales(name: str) -> None:
    rows, expected = _rows(name, "m"), _sizes(name, "m")
    for table, count in rows.items():
        assert count == pytest.approx(expected[table], abs=5 * math.sqrt(expected[table]))
