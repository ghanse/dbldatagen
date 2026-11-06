from pathlib import Path

import pytest

from profiler_bench.catalog import DATASETS, generate, materialize
from profiler_bench.catalog.gen import rng_factory


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_same_seed_gives_identical_content_hash(name: str, tmp_path: Path) -> None:
    first = materialize(name, "xs", 7, tmp_path / "a")
    second = materialize(name, "xs", 7, tmp_path / "b")
    assert first.content_hash == second.content_hash
    assert first.table_hashes == second.table_hashes
    assert first.manifest == second.manifest


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_different_seeds_give_different_data(name: str) -> None:
    first, _ = generate(DATASETS[name], "xs", 0)
    second, _ = generate(DATASETS[name], "xs", 1)
    assert any(not first[t].equals(second[t]) for t in first if first[t].num_rows)


def test_materialize_reuses_the_cached_instance(tmp_path: Path) -> None:
    first = materialize("numeric", "xs", 0, tmp_path)
    mtime = first.table_path("measurements").stat().st_mtime_ns
    second = materialize("numeric", "xs", 0, tmp_path)
    assert second.path == first.path
    assert second.table_path("measurements").stat().st_mtime_ns == mtime


def test_cache_path_includes_the_generator_fingerprint(tmp_path: Path) -> None:
    instance = materialize("temporal", "xs", 0, tmp_path)
    assert instance.path == tmp_path / "temporal" / "xs" / "seed=0" / DATASETS["temporal"].fingerprint()


def test_streams_are_independent_of_request_order() -> None:
    rng = rng_factory(0, "d")
    a_first = rng("a").random(3)
    rng("b").random(3)
    assert (rng("a").random(3) == a_first).all()
    assert not (rng("b").random(3) == a_first).all()
