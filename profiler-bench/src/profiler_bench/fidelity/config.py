"""Versioned, frozen configuration of the fidelity metrics.

The configuration holds every sample size, threshold, tolerance, and weight the metrics use. It is frozen per
experiment, and its hash is stamped into every score so a result can be traced to the exact settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_CONFIG = Path(__file__).with_name("metrics_config.json")
RECONSTRUCTION_GROUPS = (
    "schema",
    "marginal",
    "null_fraction",
    "distinct",
    "dependence_declared",
    "dependence_random",
    "constraints",
    "keys",
    "foreign_keys",
    "discriminator",
)
STRUCTURE_CATEGORIES = (
    "keys",
    "foreign_keys",
    "checks",
    "functional_dependencies",
    "dependencies",
    "patterns",
    "null_rates",
    "cardinality",
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Sampling(_Model):
    """Sample sizes.

    Attributes:
        max_rows: Rows per table and side collected to the driver for marginals and dependence.
        discriminator_rows: Rows per side used by the discriminator, half to train and half to test.
        privacy_rows: Rows per side used for distance to closest record.
        pattern_samples: Strings generated per regex when testing pattern equivalence.
        ceiling_sample_offset: Sample seed offset of the stand-in synthetic data of the ceiling.
        holdout_sample_offset: Sample seed offset of the real holdout used as the privacy baseline; must differ from
            `ceiling_sample_offset`.
    """

    max_rows: int = Field(gt=0)
    discriminator_rows: int = Field(gt=0)
    privacy_rows: int = Field(gt=0)
    pattern_samples: int = Field(gt=0)
    ceiling_sample_offset: int = Field(gt=0)
    holdout_sample_offset: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_offsets(self) -> Sampling:
        if self.ceiling_sample_offset == self.holdout_sample_offset:
            raise ValueError("the ceiling and the privacy holdout need different sample seeds")
        return self


class Marginal(_Model):
    """Marginal distance settings.

    Attributes:
        top_values: Most frequent real values kept as buckets of the total variation distance; the rest form `other`.
        categorical_max_distinct: Columns with at most this many distinct sampled real values are categorical.
    """

    top_values: int = Field(gt=0)
    categorical_max_distinct: int = Field(gt=0)


class Dependence(_Model):
    """Pairwise dependence settings.

    Attributes:
        random_pairs: Undeclared column pairs drawn per table with the score seed.
        correlation_bins: Quantile bins of a numeric conditioning column in the correlation ratio.
    """

    random_pairs: int = Field(ge=0)
    correlation_bins: int = Field(gt=1)


class Discriminator(_Model):
    """Discriminator settings.

    Attributes:
        max_iter: Boosting iterations of the gradient-boosted classifier.
    """

    max_iter: int = Field(gt=0)


class StructureTolerances(_Model):
    """Tolerances for counting a planted structure as recovered.

    Attributes:
        dependency: Absolute tolerance on a dependency strength, on top of the sampling error.
        distinct_log_ratio: Largest |log(profile distinct / planted distinct)|.
        range_relative: Largest bound error relative to the planted range span.
        pattern_equivalence: Smallest two-way match rate for a regex to count as equivalent.
    """

    dependency: float = Field(ge=0.0)
    distinct_log_ratio: float = Field(ge=0.0)
    range_relative: float = Field(ge=0.0)
    pattern_equivalence: float = Field(ge=0.0, le=1.0)


class PrivacyThresholds(_Model):
    """Pass/fail thresholds of the privacy guard.

    Attributes:
        max_copy_rate_excess: Largest exact-row copy rate above the real holdout's.
        min_dcr_ratio: Smallest median distance to closest record of synth over that of the real holdout.
        max_profile_leaks: Most sensitive real values allowed in the profile JSON.
        max_synth_leak_rate_excess: Largest share of distinct sensitive real values reproduced by the synth above the
            holdout's share.
        max_qi_match_rate_excess: Largest share of rows matching a real-unique quasi-identifier combination above the
            holdout's share.
        max_k_violations: Most literal values allowed below the profile's `min_value_count`.
        noise_z: Standard errors of sampling noise added to every rate-excess threshold.
        min_regex_language: A profile regex that fully matches a real value leaks it, and is held to the k-threshold,
            when it matches fewer than max(k, this) strings of length at most min(len(value), `max_regex_length`).
        min_literal_len: Shortest literal that every match of a profile regex contains and that is checked against
            real values (case-insensitively when folded from a single-letter case class).
        max_regex_length: Cap on the real value length used when counting a regex's language for a matched value, and
            the length used for the pathological-pattern budget probe. Default 1000.
        max_dfa_states: Scales the shared work budget for regex language counting (NFA build, epsilon-closure, DFA
            subset construction, and the length loop). A regex that exhausts the budget fails the privacy guard with
            reason `regex_budget_exceeded`.
        partial_copy_pairs: Most column pairs per table checked for row-aligned partial copying.
    """

    max_copy_rate_excess: float = Field(ge=0.0)
    min_dcr_ratio: float = Field(ge=0.0)
    max_profile_leaks: int = Field(ge=0)
    max_synth_leak_rate_excess: float = Field(ge=0.0)
    max_qi_match_rate_excess: float = Field(ge=0.0)
    max_k_violations: int = Field(ge=0)
    noise_z: float = Field(ge=0.0)
    min_regex_language: float = Field(gt=0)
    min_literal_len: int = Field(gt=0)
    max_regex_length: int = Field(gt=0)
    max_dfa_states: int = Field(gt=0)
    partial_copy_pairs: int = Field(gt=0)


class MetricsConfig(_Model):
    """Configuration of every fidelity metric.

    Attributes:
        version: Semantic version; bump it whenever a value changes.
        sampling: Sample sizes.
        marginal: Marginal distance settings.
        dependence: Pairwise dependence settings.
        discriminator: Discriminator settings.
        reconstruction_weights: Weight of each reconstruction group in a table score.
        structure_weights: Weight of each structure category in a table score.
        structure_tolerances: Tolerances of structure recovery.
        privacy: Privacy guard thresholds.
    """

    version: str
    sampling: Sampling
    marginal: Marginal
    dependence: Dependence
    discriminator: Discriminator
    reconstruction_weights: dict[str, float]
    structure_weights: dict[str, float]
    structure_tolerances: StructureTolerances
    privacy: PrivacyThresholds

    @model_validator(mode="after")
    def _check_weights(self) -> MetricsConfig:
        for label, weights, names in (
            ("reconstruction_weights", self.reconstruction_weights, RECONSTRUCTION_GROUPS),
            ("structure_weights", self.structure_weights, STRUCTURE_CATEGORIES),
        ):
            if sorted(weights) != sorted(names):
                raise ValueError(f"{label} must have exactly {sorted(names)}")
            if any(w < 0 for w in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-9:
                raise ValueError(f"{label} must be non-negative and sum to 1")
        return self

    def hash(self) -> str:
        """Returns the SHA-256 of the canonical JSON of the configuration."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def load_config(path: Path | None = None) -> MetricsConfig:
    """Reads a metrics configuration.

    Args:
        path: JSON file; defaults to the `metrics_config.json` shipped with the package.
    """
    return MetricsConfig.model_validate_json((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
