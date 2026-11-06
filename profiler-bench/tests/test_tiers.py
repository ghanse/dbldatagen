import pytest

from profiler_bench.catalog import DATASETS, TIERS, generate


def _rows(name: str, tier: str) -> dict[str, int]:
    tables, truth = generate(DATASETS[name], tier, 0)
    assert {t.name: t.row_count for t in truth.tables} == {n: t.num_rows for n, t in tables.items()}
    return {n: t.num_rows for n, t in tables.items()}


def test_tiers_are_ordered() -> None:
    assert list(TIERS) == ["xs", "s", "m"]
    assert TIERS["xs"] < TIERS["s"] < TIERS["m"]


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_largest_table_tracks_the_tier_size(name: str) -> None:
    for tier in ("xs", "s"):
        assert max(_rows(name, tier).values()) == pytest.approx(TIERS[tier], rel=0.05)


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_no_table_shrinks_from_xs_to_s(name: str) -> None:
    xs, s = _rows(name, "xs"), _rows(name, "s")
    assert all(s[t] >= xs[t] for t in xs)
    assert sum(s.values()) > 10 * sum(xs.values())


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(DATASETS))
def test_m_tier_scales(name: str) -> None:
    assert max(_rows(name, "m").values()) == pytest.approx(TIERS["m"], rel=0.05)
