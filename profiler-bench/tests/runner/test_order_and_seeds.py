"""Order rotation and paired fidelity seeds."""

from __future__ import annotations

from profiler_bench.runner.fidelity_run import paired_seeds
from profiler_bench.runner.measure import rotate_order


def test_order_rotates_across_samples() -> None:
    assert rotate_order(0, 0) == ("baseline", "candidate")
    assert rotate_order(1, 0) == ("candidate", "baseline")
    assert rotate_order(0, 1) == ("candidate", "baseline")
    assert rotate_order(1, 1) == ("baseline", "candidate")


def test_paired_seeds_are_shared() -> None:
    assert paired_seeds(5) == [0, 1, 2, 3, 4]
    assert paired_seeds(2, start=10) == [10, 11]
