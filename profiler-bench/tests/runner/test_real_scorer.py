"""Real scorer wiring, seeds, and ceiling caching."""

# ruff: noqa: ANN401

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from profiler_bench.fidelity import ReconstructionScore
from profiler_bench.rsi.__main__ import main
from profiler_bench.runner.scorer import FakeScorer, _RealScorer, _score_with_real_modules, try_real_scorer


class _Score:
    def __init__(self, score: float, detail: dict[str, Any]) -> None:
        self.score = score
        self.config_hash = "metrics-a"
        self._detail = detail

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return self._detail

    def relative_to(self, ceiling: Any) -> float:
        assert ceiling.score == 0.9
        return 0.75


def _real_scorer(tmp_path: Path, **overrides: Any) -> _RealScorer:
    values: dict[str, Any] = {
        "sample": lambda *args, **kwargs: None,
        "score_reconstruction": lambda *args, **kwargs: None,
        "score_structure": lambda *args, **kwargs: None,
        "score_privacy": lambda *args, **kwargs: None,
        "fidelity_ceiling": lambda *args, **kwargs: None,
        "cache_dir": tmp_path,
        "harness_hash": "harness-a",
        "metrics_config_hash": "metrics-a",
    }
    values.update(overrides)
    return _RealScorer(**values)


def test_real_scorer_passes_profiled_real_and_explicit_holdout(monkeypatch: Any, tmp_path: Path) -> None:
    profiled = {"table": object()}
    holdout = {"table": object()}
    synth = {"table": object()}
    profile = {"profile": True}
    loaded_profile = object()
    manifest = object()
    ceiling = cast(ReconstructionScore, SimpleNamespace(score=0.9, model_dump=lambda mode: {"score": 0.9}))
    privacy_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def load(dataset: str, tier: str, *, seed: int, spark: Any, sample_seed: int) -> dict[str, object]:
        del dataset, tier, spark
        assert seed == 7
        return profiled if sample_seed == 9 else holdout

    def score_privacy(*args: Any, **kwargs: Any) -> Any:
        privacy_calls.append((args, kwargs))
        return SimpleNamespace(
            passed=False,
            checks=[SimpleNamespace(name="table.unmeasured", detail={"unmeasured": 1.0})],
            model_dump=lambda mode: {"passed": False},
        )

    monkeypatch.setattr("profiler_bench.catalog.load", load)
    monkeypatch.setattr(
        "profiler_bench.catalog.materialize", lambda *args, **kwargs: SimpleNamespace(manifest=manifest)
    )
    monkeypatch.setattr("dbldatagen_profiler.model.load_profile", lambda value, strict: loaded_profile)
    monkeypatch.setattr(
        "pyspark.sql.SparkSession.getActiveSession",
        lambda: SimpleNamespace(conf=SimpleNamespace(set=lambda *args: None)),
    )
    scorer = _real_scorer(
        tmp_path,
        sample=lambda value, spark, seed: SimpleNamespace(tables=synth),
        score_reconstruction=lambda real, generated, truth, seed: _Score(0.8, {"score": 0.8}),
        score_structure=lambda value, truth, **kwargs: _Score(0.7, {"score": 0.7}),
        score_privacy=score_privacy,
    )
    scorer._ceiling_cache[("numeric", "xs", 7, "harness-a", "metrics-a")] = ceiling

    scores = _score_with_real_modules(
        scorer,
        profile,
        dataset="numeric",
        tier="xs",
        seed=3,
        structure_seed=7,
        sample_seed=9,
    )

    args, kwargs = privacy_calls[0]
    assert args == (profiled, synth, loaded_profile, manifest)
    assert kwargs == {"holdout": holdout, "sample_seed": 9, "seed": 3}
    assert scores.privacy_passed is False
    assert scores.privacy_unmeasured == ["table.unmeasured"]
    assert scores.relative_reconstruction == 0.75
    assert scores.detail["relative_to_ceiling"] == 0.75
    assert scores.detail["seeds"] == {
        "sampler": 3,
        "structure": 7,
        "profile_sample": 9,
        "privacy_holdout_sample": 11,
        "ceiling_samples": [7, 8],
    }


def test_ceiling_cache_keys_harness_and_metrics_hashes(tmp_path: Path, spark: Any) -> None:
    assert spark is not None
    calls: list[dict[str, Any]] = []

    def fidelity_ceiling(dataset: str, tier: str, **kwargs: Any) -> ReconstructionScore:
        calls.append(kwargs)
        return ReconstructionScore(
            dataset=dataset,
            tier=tier,
            seed=0,
            config_version="test",
            config_hash="metrics-a",
            tables=[],
            score=0.9,
        )

    first = _real_scorer(tmp_path, fidelity_ceiling=fidelity_ceiling)
    same = _real_scorer(tmp_path, fidelity_ceiling=fidelity_ceiling)
    changed_harness = _real_scorer(tmp_path, fidelity_ceiling=fidelity_ceiling, harness_hash="harness-b")
    changed_metrics = _real_scorer(tmp_path, fidelity_ceiling=fidelity_ceiling, metrics_config_hash="metrics-b")

    assert first.ceiling("numeric", "xs", structure_seed=7) == 0.9
    assert same.ceiling("numeric", "xs", structure_seed=7) == 0.9
    assert changed_harness.ceiling("numeric", "xs", structure_seed=7) == 0.9
    assert changed_metrics.ceiling("numeric", "xs", structure_seed=7) == 0.9
    assert len(calls) == 3
    assert all(call["structure_seed"] == 7 and call["sample_seed"] == 7 for call in calls)
    assert len(list((tmp_path / "ceilings").glob("*/*/*.json"))) == 3


def test_cli_and_factory_default_to_real_scorer(monkeypatch: Any, tmp_path: Path) -> None:
    scorer = try_real_scorer(cache_dir=tmp_path, harness_hash="harness", metrics_config_hash="metrics")
    assert scorer is not None
    assert not isinstance(scorer, FakeScorer)

    captured: list[Any] = []
    output = tmp_path / "profiler" / "rsi" / "experiments" / "cli-default"
    output.mkdir(parents=True)
    (output / "verdict.json").write_text(json.dumps({"verdict": "inconclusive"}))

    def run_experiment(config: Any) -> Path:
        captured.append(config)
        return output

    monkeypatch.setattr("profiler_bench.rsi.__main__.run_experiment", run_experiment)
    assert (
        main(
            [
                "run",
                "--baseline",
                str(tmp_path),
                "--candidate",
                str(tmp_path),
                "--experiment-id",
                "cli-default",
            ]
        )
        == 0
    )
    assert captured[0].scorer is None
