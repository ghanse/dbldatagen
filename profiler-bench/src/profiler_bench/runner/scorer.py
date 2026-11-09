"""Scorer protocol: wraps sampler + fidelity so the runner can inject a fake."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from profiler_bench.fidelity import ReconstructionScore


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


def try_real_scorer(*, cache_dir: Path, harness_hash: str) -> Scorer | None:
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
        cache_dir=cache_dir,
        harness_hash=harness_hash,
    )


@dataclass
class _RealScorer:
    sample: Any
    score_reconstruction: Any
    score_structure: Any
    score_privacy: Any
    fidelity_ceiling: Any
    cache_dir: Path
    harness_hash: str
    _ceiling_cache: dict[tuple[str, str, int], ReconstructionScore] = field(default_factory=dict)

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
        result = self._ceiling(dataset, tier, structure_seed=structure_seed)
        return float(result.score)

    def _ceiling(self, dataset: str, tier: str, *, structure_seed: int) -> ReconstructionScore:
        """Returns the full cached real-vs-real ceiling score."""
        from pyspark.sql import SparkSession  # noqa: PLC0415

        from profiler_bench.fidelity import ReconstructionScore  # noqa: PLC0415

        key = (dataset, tier, structure_seed)
        if key not in self._ceiling_cache:
            path = self.cache_dir / "ceilings" / self.harness_hash / f"{dataset}-{tier}-seed-{structure_seed}.json"
            if path.is_file():
                result = ReconstructionScore.model_validate_json(path.read_text())
            else:
                spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
                spark.conf.set("spark.sql.session.timeZone", "UTC")
                result = self.fidelity_ceiling(dataset, tier, spark=spark, structure_seed=structure_seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(result.model_dump_json(indent=2) + "\n")
            self._ceiling_cache[key] = result
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
    from profiler_bench.fidelity import load_config  # noqa: PLC0415

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    real = load(dataset, tier, seed=structure_seed, spark=spark, sample_seed=sample_seed)
    manifest = materialize(dataset, tier, seed=structure_seed, sample_seed=sample_seed).manifest
    loaded = load_profile(profile, strict=False)
    synth = scorer.sample(loaded, spark, seed=seed).tables
    config = load_config()
    holdout_seed = sample_seed + config.sampling.holdout_sample_offset
    ceiling_seeds = {structure_seed, structure_seed + config.sampling.ceiling_sample_offset}
    if holdout_seed == sample_seed or holdout_seed in ceiling_seeds:
        raise ValueError("privacy holdout seed overlaps the profiled sample or ceiling samples")
    holdout = load(dataset, tier, seed=structure_seed, spark=spark, sample_seed=holdout_seed)
    reconstruction = scorer.score_reconstruction(real, synth, manifest, seed=seed)
    structure = scorer.score_structure(loaded, manifest, real=real, sample_seed=sample_seed, seed=seed)
    privacy = scorer.score_privacy(
        holdout,
        synth,
        loaded,
        manifest,
        sample_seed=holdout_seed,
        seed=seed,
    )
    if not isinstance(privacy.passed, bool):
        raise RuntimeError("privacy scorer did not produce a pass/fail result")
    ceiling = scorer._ceiling(dataset, tier, structure_seed=structure_seed)
    relative = float(reconstruction.relative_to(ceiling))
    return FidelityScores(
        reconstruction=float(reconstruction.score),
        structure=float(structure.score),
        privacy_passed=bool(privacy.passed),
        ceiling=float(ceiling.score),
        relative_reconstruction=relative,
        detail={
            "reconstruction": reconstruction.model_dump(mode="json"),
            "structure": structure.model_dump(mode="json"),
            "privacy": privacy.model_dump(mode="json"),
            "ceiling": ceiling.model_dump(mode="json"),
            "relative_to_ceiling": relative,
            "seeds": {
                "sampler": seed,
                "structure": structure_seed,
                "profile_sample": sample_seed,
                "privacy_holdout_sample": holdout_seed,
                "ceiling_samples": sorted(ceiling_seeds),
            },
        },
        metrics_config_hash=str(reconstruction.config_hash),
    )
