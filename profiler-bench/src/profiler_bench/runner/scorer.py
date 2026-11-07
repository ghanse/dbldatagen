"""Scorer protocol: wraps sampler + fidelity so the runner can inject a fake."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class FidelityScores:
    """JSON-serializable fidelity scores for one (build, dataset, seed).

    Attributes:
        reconstruction: Overall reconstruction score in [0, 1].
        structure: Overall structure-recovery score in [0, 1].
        privacy_passed: Whether every privacy guard check passed.
        ceiling: Real-vs-real ceiling for the dataset, when known.
        relative_reconstruction: `reconstruction` as a fraction of `ceiling`, capped at 1.
        detail: Full per-metric detail from the underlying scorers.
        metrics_config_hash: Hash of the metrics configuration that produced the scores.
    """

    reconstruction: float
    structure: float
    privacy_passed: bool
    ceiling: float | None
    relative_reconstruction: float | None
    detail: dict[str, Any]
    metrics_config_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return asdict(self)


@runtime_checkable
class Scorer(Protocol):
    """Scores a profile by sampling synthetic data and comparing it to the real tables."""

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
        """Returns reconstruction, structure, and privacy scores for one sampler seed."""

    def ceiling(self, dataset: str, tier: str, *, structure_seed: int = 0) -> float:
        """Returns the cached real-vs-real reconstruction ceiling for one dataset."""


@dataclass
class FakeScorer:
    """Deterministic scorer for tests; ignores the profile and returns configured values."""

    reconstruction: float = 0.8
    structure: float = 0.7
    privacy_passed: bool = True
    ceiling_value: float = 0.9
    metrics_config_hash: str = "fake-metrics"
    by_seed: dict[int, dict[str, Any]] = field(default_factory=dict)
    by_role_seed: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    role: str = "baseline"

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
        """Returns configured scores, optionally overridden per seed or (role, seed)."""
        del profile, tier, structure_seed, sample_seed
        values: dict[str, Any] = {
            "reconstruction": self.reconstruction,
            "structure": self.structure,
            "privacy_passed": self.privacy_passed,
        }
        if seed in self.by_seed:
            values.update(self.by_seed[seed])
        if (self.role, seed) in self.by_role_seed:
            values.update(self.by_role_seed[(self.role, seed)])
        reconstruction = float(values["reconstruction"])
        ceiling = self.ceiling_value
        relative = min(1.0, reconstruction / ceiling) if ceiling > 0 else 1.0
        return FidelityScores(
            reconstruction=reconstruction,
            structure=float(values["structure"]),
            privacy_passed=bool(values["privacy_passed"]),
            ceiling=ceiling,
            relative_reconstruction=relative,
            detail={"dataset": dataset, "seed": seed, "values": values},
            metrics_config_hash=self.metrics_config_hash,
        )

    def ceiling(self, dataset: str, tier: str, *, structure_seed: int = 0) -> float:
        """Returns the configured ceiling."""
        del dataset, tier, structure_seed
        return self.ceiling_value

    def with_role(self, role: str) -> FakeScorer:
        """Returns a copy bound to `role` for paired baseline/candidate overrides."""
        return FakeScorer(
            reconstruction=self.reconstruction,
            structure=self.structure,
            privacy_passed=self.privacy_passed,
            ceiling_value=self.ceiling_value,
            metrics_config_hash=self.metrics_config_hash,
            by_seed=dict(self.by_seed),
            by_role_seed=dict(self.by_role_seed),
            role=role,
        )


def try_real_scorer() -> Scorer | None:
    """Returns a real sampler+fidelity scorer when those modules are importable, else None."""
    try:
        from profiler_bench import fidelity, sampler  # noqa: PLC0415
    except ImportError:
        return None
    return _RealScorer(
        sample=sampler.sample,
        score_reconstruction=fidelity.score_reconstruction,
        score_structure=fidelity.score_structure,
        score_privacy=fidelity.score_privacy,
        fidelity_ceiling=fidelity.ceiling,
    )


@dataclass
class _RealScorer:
    sample: Any
    score_reconstruction: Any
    score_structure: Any
    score_privacy: Any
    fidelity_ceiling: Any
    _ceiling_cache: dict[tuple[str, str, int], float] = field(default_factory=dict)

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
        """Samples from the profile and scores reconstruction, structure, and privacy."""
        return _score_with_real_modules(
            self,
            profile,
            dataset=dataset,
            tier=tier,
            seed=seed,
            structure_seed=structure_seed,
            sample_seed=sample_seed,
        )

    def ceiling(self, dataset: str, tier: str, *, structure_seed: int = 0) -> float:
        """Returns a cached real-vs-real ceiling for the harness version."""
        key = (dataset, tier, structure_seed)
        if key not in self._ceiling_cache:
            result = self.fidelity_ceiling(dataset, tier, structure_seed=structure_seed)
            self._ceiling_cache[key] = float(result.score if hasattr(result, "score") else result)
        return self._ceiling_cache[key]


def _score_with_real_modules(
    scorer: _RealScorer,
    profile: dict[str, Any],
    *,
    dataset: str,
    tier: str,
    seed: int,
    structure_seed: int,
    sample_seed: int,
) -> FidelityScores:
    """Implements real scoring once sampler/fidelity are importable."""
    from pyspark.sql import SparkSession  # noqa: PLC0415

    from dbldatagen_profiler.model import load_profile  # noqa: PLC0415
    from profiler_bench.catalog import load, materialize  # noqa: PLC0415

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    real = load(dataset, tier, seed=structure_seed, spark=spark, sample_seed=sample_seed)
    manifest = materialize(dataset, tier, seed=structure_seed, sample_seed=sample_seed).manifest
    loaded = load_profile(profile, strict=False)
    synth = scorer.sample(loaded, spark, seed=seed).tables
    reconstruction = scorer.score_reconstruction(real, synth, manifest, seed=seed)
    structure = scorer.score_structure(loaded, manifest)
    privacy = scorer.score_privacy(real, synth, loaded, manifest, seed=seed)
    ceiling_value = scorer.ceiling(dataset, tier, structure_seed=structure_seed)
    relative = min(1.0, float(reconstruction.score) / ceiling_value) if ceiling_value > 0 else 1.0
    return FidelityScores(
        reconstruction=float(reconstruction.score),
        structure=float(structure.score),
        privacy_passed=bool(privacy.passed),
        ceiling=ceiling_value,
        relative_reconstruction=relative,
        detail={
            "reconstruction": reconstruction.model_dump(mode="json"),
            "structure": structure.model_dump(mode="json"),
            "privacy": privacy.model_dump(mode="json"),
        },
        metrics_config_hash=str(reconstruction.config_hash),
    )
