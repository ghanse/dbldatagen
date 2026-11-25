"""Scorer protocol: wraps sampler + fidelity so the runner can inject a fake."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from profiler_bench.catalog import REAL_DATASETS


if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from profiler_bench.fidelity import ReconstructionScore


@dataclass(frozen=True)
class FidelityScores:
    """JSON-serializable fidelity scores for one (build, dataset, seed).

    Attributes:
        reconstruction: Overall reconstruction score in [0, 1].
        structure: Overall structure-recovery score in [0, 1]; None when not applicable (no planted ground truth).
        privacy_passed: Whether every privacy guard check passed.
        ceiling: Real-vs-real ceiling for the dataset, when known.
        relative_reconstruction: `reconstruction` as a fraction of `ceiling`, capped at 1.
        privacy_unmeasured: Privacy checks that could not be measured.
        detail: Full per-metric detail from the underlying scorers.
        metrics_config_hash: Hash of the metrics configuration that produced the scores.
        privacy_status: `pass`, `fail`, or `unmeasured` on real datasets (see `PrivacyScore.status`); None otherwise.
    """

    reconstruction: float
    structure: float | None
    privacy_passed: bool
    ceiling: float | None
    relative_reconstruction: float | None
    privacy_unmeasured: list[str]
    detail: dict[str, Any]
    metrics_config_hash: str
    privacy_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return asdict(self)


_RealParts = tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, float]]]


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
    """Deterministic scorer for tests; ignores the profile and returns configured values.

    Structure is None (not applicable) on real datasets, as with the real scorer.
    """

    reconstruction: float = 0.8
    structure: float = 0.7
    privacy_passed: bool = True
    ceiling_value: float = 0.9
    metrics_config_hash: str = "fake-metrics"
    by_seed: dict[int, dict[str, Any]] = field(default_factory=dict)
    by_role_seed: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    relative_by_fraction: dict[float, float] = field(default_factory=dict)
    role: str = "baseline"
    raise_message: str | None = None
    crash_exit_code: int | None = None
    fail_datasets: tuple[str, ...] = ()

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
        if not self.fail_datasets or dataset in self.fail_datasets:
            if self.crash_exit_code is not None:
                os._exit(int(self.crash_exit_code))
            if self.raise_message is not None:
                raise RuntimeError(self.raise_message)
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
            structure=None if dataset in REAL_DATASETS else float(values["structure"]),
            privacy_passed=bool(values["privacy_passed"]),
            ceiling=ceiling,
            relative_reconstruction=relative,
            privacy_unmeasured=[],
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
            relative_by_fraction=dict(self.relative_by_fraction),
            role=role,
            raise_message=self.raise_message,
            crash_exit_code=self.crash_exit_code,
            fail_datasets=tuple(self.fail_datasets),
        )

    def to_spec(self) -> dict[str, Any]:
        """Returns a JSON-serializable worker scorer specification."""
        return {
            "kind": "fake",
            "reconstruction": self.reconstruction,
            "structure": self.structure,
            "privacy_passed": self.privacy_passed,
            "ceiling_value": self.ceiling_value,
            "metrics_config_hash": self.metrics_config_hash,
            "by_seed": {str(seed): dict(values) for seed, values in self.by_seed.items()},
            "by_role_seed": {f"{role}:{seed}": dict(values) for (role, seed), values in self.by_role_seed.items()},
            "relative_by_fraction": {str(frac): value for frac, value in self.relative_by_fraction.items()},
            "role": self.role,
            "raise_message": self.raise_message,
            "crash_exit_code": self.crash_exit_code,
            "fail_datasets": list(self.fail_datasets),
        }


def scorer_from_spec(spec: dict[str, Any]) -> Scorer:
    """Rebuilds a scorer from a worker specification (`fake` or `real`)."""
    kind = str(spec.get("kind", "real"))
    if kind == "fake":
        by_role_seed: dict[tuple[str, int], dict[str, Any]] = {}
        for key, values in dict(spec.get("by_role_seed", {})).items():
            role, seed_text = str(key).split(":", 1)
            by_role_seed[(role, int(seed_text))] = dict(values)
        return FakeScorer(
            reconstruction=float(spec.get("reconstruction", 0.8)),
            structure=float(spec.get("structure", 0.7)),
            privacy_passed=bool(spec.get("privacy_passed", True)),
            ceiling_value=float(spec.get("ceiling_value", 0.9)),
            metrics_config_hash=str(spec.get("metrics_config_hash", "fake-metrics")),
            by_seed={int(seed): dict(values) for seed, values in dict(spec.get("by_seed", {})).items()},
            by_role_seed=by_role_seed,
            relative_by_fraction={
                float(frac): float(value) for frac, value in dict(spec.get("relative_by_fraction", {})).items()
            },
            role=str(spec.get("role", "baseline")),
            raise_message=spec.get("raise_message"),
            crash_exit_code=spec.get("crash_exit_code"),
            fail_datasets=tuple(str(name) for name in list(spec.get("fail_datasets", []))),
        )
    if kind != "real":
        raise ValueError(f"unknown scorer kind: {kind}")
    scorer = try_real_scorer(
        cache_dir=Path(str(spec["cache_dir"])),
        harness_hash=str(spec["harness_hash"]),
        metrics_config_hash=str(spec["metrics_config_hash"]),
        real_root=Path(str(spec["real_root"])) if spec.get("real_root") else None,
    )
    if scorer is None:
        raise RuntimeError("real scorer is not importable in the fidelity worker")
    return scorer


def real_scorer_spec(
    *, cache_dir: Path, harness_hash: str, metrics_config_hash: str, real_root: Path | None = None
) -> dict[str, Any]:
    """Returns a worker specification that builds the real sampler+fidelity scorer."""
    spec: dict[str, Any] = {
        "kind": "real",
        "cache_dir": str(cache_dir),
        "harness_hash": harness_hash,
        "metrics_config_hash": metrics_config_hash,
    }
    if real_root is not None:
        spec["real_root"] = str(real_root)
    return spec


def try_real_scorer(
    *, cache_dir: Path, harness_hash: str, metrics_config_hash: str, real_root: Path | None = None
) -> Scorer | None:
    """Returns a real sampler+fidelity scorer when those modules are importable, else None.

    `real_root` is the catalog cache holding the real-dataset exports; it defaults to `catalog.cache_dir()`.
    """
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
        metrics_config_hash=metrics_config_hash,
        real_root=real_root,
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
    metrics_config_hash: str
    real_root: Path | None = None
    _real_parts: dict[tuple[str, str], _RealParts] = field(default_factory=dict)
    _ceiling_cache: dict[tuple[str, str, int, str, str], ReconstructionScore] = field(default_factory=dict)

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
        if dataset in REAL_DATASETS:
            return _score_real_dataset(self, profile, dataset=dataset, tier=tier, seed=seed, sample_seed=sample_seed)
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

        key = (dataset, tier, structure_seed, self.harness_hash, self.metrics_config_hash)
        if key not in self._ceiling_cache:
            path = (
                self.cache_dir
                / "ceilings"
                / self.harness_hash
                / self.metrics_config_hash
                / f"{dataset}-{tier}-seed-{structure_seed}.json"
            )

            def compute() -> ReconstructionScore:
                from profiler_bench.fidelity import ReconstructionScore as ScoreModel  # noqa: PLC0415

                spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
                spark.conf.set("spark.sql.session.timeZone", "UTC")
                if dataset in REAL_DATASETS:
                    return _real_ceiling(self, dataset, tier, spark)
                return cast(
                    ScoreModel,
                    self.fidelity_ceiling(
                        dataset,
                        tier,
                        spark=spark,
                        structure_seed=structure_seed,
                        sample_seed=structure_seed,
                    ),
                )

            self._ceiling_cache[key] = read_or_compute_ceiling(path, compute)
        return self._ceiling_cache[key]


def read_or_compute_ceiling(path: Path, compute: Callable[[], ReconstructionScore]) -> ReconstructionScore:
    """Loads a ceiling JSON under an fcntl lock, computing once on miss/corruption.

    Readers tolerate a missing or partial file by recomputing under the exclusive lock.
    Writes use a temp file + ``os.replace`` so observers never see truncated JSON.
    """
    from profiler_bench.fidelity import ReconstructionScore  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            loaded = _try_load_ceiling(path)
            if loaded is not None:
                return loaded
            result = compute()
            if not isinstance(result, ReconstructionScore):
                raise TypeError("ceiling compute must return ReconstructionScore")
            _atomic_write_text(path, result.model_dump_json(indent=2) + "\n")
            return result
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _try_load_ceiling(path: Path) -> ReconstructionScore | None:
    from profiler_bench.fidelity import ReconstructionScore  # noqa: PLC0415

    try:
        if not path.is_file():
            return None
        return ReconstructionScore.model_validate_json(path.read_text())
    except Exception:
        return None


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


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
        real,
        synth,
        loaded,
        manifest,
        holdout=holdout,
        sample_seed=sample_seed,
        seed=seed,
    )
    if not isinstance(privacy.passed, bool):
        raise RuntimeError("privacy scorer did not produce a pass/fail result")
    ceiling = scorer._ceiling(dataset, tier, structure_seed=structure_seed)
    relative = float(reconstruction.relative_to(ceiling))
    privacy_unmeasured = [check.name for check in privacy.checks if check.detail.get("unmeasured", 0.0) > 0.0]
    return FidelityScores(
        reconstruction=float(reconstruction.score),
        structure=float(structure.score),
        privacy_passed=bool(privacy.passed),
        ceiling=float(ceiling.score),
        relative_reconstruction=relative,
        privacy_unmeasured=privacy_unmeasured,
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


def _real_ceiling(scorer: _RealScorer, dataset: str, tier: str, spark: SparkSession) -> ReconstructionScore:
    """Scores the ceiling part of a real dataset against its profiled part."""
    from profiler_bench.catalog.real import load_part, materialize_part  # noqa: PLC0415

    spec = REAL_DATASETS[dataset]
    manifest = materialize_part(spec, tier, "profiled", scorer.real_root).manifest
    profiled = load_part(spec, tier, "profiled", spark, scorer.real_root)
    stand_in = load_part(spec, tier, "ceiling", spark, scorer.real_root)
    return cast("ReconstructionScore", scorer.score_reconstruction(profiled, stand_in, manifest, seed=0))


def _score_real_dataset(
    scorer: _RealScorer,
    profile: dict[str, Any],
    *,
    dataset: str,
    tier: str,
    seed: int,
    sample_seed: int,
) -> FidelityScores:
    """Scores a real dataset: reconstruction and privacy against its parts; structure is not applicable."""
    from pyspark.sql import SparkSession  # noqa: PLC0415

    from dbldatagen_profiler.model import load_profile  # noqa: PLC0415
    from profiler_bench.catalog.real import SPLIT, materialize_part  # noqa: PLC0415

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    spec = REAL_DATASETS[dataset]
    manifest = materialize_part(spec, tier, "profiled", scorer.real_root).manifest
    from profiler_bench.fidelity.privacy import shuffled_columns  # noqa: PLC0415

    real, holdout, overlap = _real_parts(scorer, dataset, tier, spark)
    chance = {table: shuffled_columns(frame, seed, table) for table, frame in real.items()}
    loaded = load_profile(profile, strict=False)
    synth = scorer.sample(loaded, spark, seed=seed).tables
    reconstruction = scorer.score_reconstruction(real, synth, manifest, seed=seed)
    min_rows = max(int(loaded.privacy.min_value_count), MIN_INDEPENDENT_ROWS)
    privacy = scorer.score_privacy(
        real,
        synth,
        loaded,
        manifest,
        holdout=holdout,
        sample_seed=sample_seed,
        seed=seed,
        chance=chance,
        min_independent_rows=min_rows,
    )
    if not isinstance(privacy.passed, bool):
        raise RuntimeError("privacy scorer did not produce a pass/fail result")
    ceiling = scorer._ceiling(dataset, tier, structure_seed=0)
    relative = float(reconstruction.relative_to(ceiling))
    return FidelityScores(
        reconstruction=float(reconstruction.score),
        structure=None,
        privacy_passed=bool(privacy.passed),
        privacy_status=str(privacy.status),
        ceiling=float(ceiling.score),
        relative_reconstruction=relative,
        privacy_unmeasured=[c.name for c in privacy.checks if c.detail.get("unmeasured", 0.0) > 0.0],
        detail={
            "reconstruction": reconstruction.model_dump(mode="json"),
            "structure": {"status": "not_applicable", "reason": "real dataset: ground_truth none"},
            "privacy": privacy.model_dump(mode="json"),
            "ceiling": ceiling.model_dump(mode="json"),
            "relative_to_ceiling": relative,
            "seeds": {"sampler": seed},
            "parts": {
                "profiled": "profiled",
                "privacy_holdout": "holdout rows not in profiled",
                "copy_check": "membership on non-key columns: profiled vs equal-size independent holdout rows",
                "key_only_copy_control": "profiled with each column shuffled, seeded by the sampler seed",
                "ceiling": "ceiling vs profiled",
            },
            "privacy_min_independent_rows": min_rows,
            "holdout_overlap": overlap,
            "split": SPLIT,
        },
        metrics_config_hash=str(reconstruction.config_hash),
    )


MIN_INDEPENDENT_ROWS = 100


def _real_parts(scorer: _RealScorer, dataset: str, tier: str, spark: SparkSession) -> _RealParts:
    """Returns the profiled part, its independent holdout rows, and the holdout's overlap with it per table.

    Independent rows are the holdout rows equal to no profiled row; they are cached on the scorer per cell.
    """
    from profiler_bench.catalog.real import load_part  # noqa: PLC0415
    from profiler_bench.fidelity.privacy import independent_rows  # noqa: PLC0415

    key = (dataset, tier)
    if key not in scorer._real_parts:
        spec = REAL_DATASETS[dataset]
        real = load_part(spec, tier, "profiled", spark, scorer.real_root)
        holdout = load_part(spec, tier, "holdout", spark, scorer.real_root)
        independent = {t: independent_rows(frame, real[t]).cache() for t, frame in holdout.items()}
        overlap: dict[str, dict[str, float]] = {}
        for table in real:
            rows, kept = holdout[table].count(), independent[table].count()
            overlap[table] = {
                "holdout_rows": float(rows),
                "holdout_independent_rows": float(kept),
                "holdout_overlap_share": (rows - kept) / rows if rows else 0.0,
            }
        scorer._real_parts[key] = (real, independent, overlap)
    return scorer._real_parts[key]
