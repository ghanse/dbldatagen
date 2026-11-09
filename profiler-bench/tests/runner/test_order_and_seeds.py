"""Order rotation and paired fidelity seeds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from profiler_bench.runner.fidelity_run import _one_seed, paired_seeds
from profiler_bench.runner.measure import rotate_order
from profiler_bench.runner.scorer import FidelityScores


def test_order_rotates_across_samples() -> None:
    assert rotate_order(0, 0) == ("baseline", "candidate")
    assert rotate_order(1, 0) == ("candidate", "baseline")
    assert rotate_order(0, 1) == ("candidate", "baseline")
    assert rotate_order(1, 1) == ("baseline", "candidate")


def test_paired_seeds_are_shared() -> None:
    assert paired_seeds(5) == [0, 1, 2, 3, 4]
    assert paired_seeds(2, start=10) == [10, 11]


class _BrokenScorer:
    def score(
        self,
        profile: dict[str, Any],
        *,
        dataset: str,
        tier: str,
        seed: int,
        structure_seed: int,
        sample_seed: int,
    ) -> FidelityScores:
        del profile, dataset, tier, seed, structure_seed, sample_seed
        raise ValueError("scoring failed")

    def ceiling(self, dataset: str, tier: str, *, structure_seed: int = 0) -> float:
        del dataset, tier, structure_seed
        raise AssertionError("ceiling must be inside the score failure boundary")


def test_scorer_exception_is_a_failed_sample(tmp_path: Path) -> None:
    record = _one_seed(
        role="baseline",
        dataset="numeric",
        tier="xs",
        seed=0,
        profile={},
        scorer=_BrokenScorer(),
        samples_dir=tmp_path,
        dataset_seed=0,
    )
    assert record == {
        "kind": "fidelity",
        "role": "baseline",
        "dataset": "numeric",
        "tier": "xs",
        "seed": 0,
        "ok": False,
        "error_class": "ValueError",
        "error": "scoring failed",
    }
    saved = json.loads((tmp_path / "fidelity-baseline-numeric-xs-seed-0.json").read_text())
    assert saved == record
