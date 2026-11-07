"""JSON-serializable score objects and the weighted aggregation shared by every metric family.

Every score is in [0, 1] with 1 meaning perfect. Each object keeps the per-metric detail it was aggregated from, so a
change in a top-level score can be traced to the column, pair, check, or key that moved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict


Score = float


class _Score(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ColumnScore(_Score):
    """Scores of one leaf column.

    Attributes:
        path: Leaf path, see `frames.Leaf`.
        kind: `numeric`, `categorical`, `string`, `length`, or `missing` when the synth lacks the column or its type.
        metrics: Score per metric: `ks`, `wasserstein`, `tvd`, `format`, `length_ks`, `length_wasserstein`,
            `special_values`, `missing`, `null_fraction`, `distinct`.
        raw: Raw distance or error behind each metric, and the real and synth values it compares.
    """

    path: str
    kind: str
    metrics: dict[str, Score]
    raw: dict[str, float] = {}


class PairScore(_Score):
    """Dependence error of one column pair.

    Attributes:
        columns: The two columns; for `correlation_ratio` the first is the conditioning column.
        measure: `pearson`, `spearman`, `cramers_v`, `correlation_ratio`, or `value_agreement` (two high-cardinality
            string columns; `real` and `synth` are their exact-equality rates).
        declared: Whether the manifest declares the pair; otherwise it was drawn at random.
        real: Measure on the real sample.
        synth: Measure on the synth sample, or None when the synth lacks a column.
        score: 1 - min(1, |real - synth|); for `value_agreement`, also the shared-prefix length distance.
    """

    columns: tuple[str, str]
    measure: str
    declared: bool
    real: float
    synth: float | None
    score: Score


class CheckScore(_Score):
    """Satisfaction of one check constraint or functional dependency on the synth.

    Attributes:
        label: The constraint.
        kind: `check` or `functional_dependency`.
        score: Fraction of synth rows satisfying a check, or the largest fraction of rows that can be kept so a
            functional dependency holds.
    """

    label: str
    kind: Literal["check", "functional_dependency"]
    score: Score


class KeyScore(_Score):
    """Validity of one key on the synth.

    Attributes:
        columns: Key columns.
        kind: `primary` or `unique`.
        score: Distinct non-null keys over rows (primary) or over rows with a non-null key (unique).
    """

    columns: tuple[str, ...]
    kind: Literal["primary", "unique"]
    score: Score


class ForeignKeyScore(_Score):
    """Referential integrity and fan-out of one foreign key.

    Attributes:
        label: The foreign key.
        real_coverage: Fraction of non-null real child keys with a parent.
        synth_coverage: Fraction of non-null synth child keys with a parent.
        fanout_ks: Kolmogorov-Smirnov statistic of child rows per parent key, real vs synth.
        fanout_wasserstein: Normalized Wasserstein-1 distance of the same.
        score: Mean of 1 - |coverage error|, 1 - fanout_ks, and 1 - fanout_wasserstein.
    """

    label: str
    real_coverage: float
    synth_coverage: float
    fanout_ks: float
    fanout_wasserstein: float
    score: Score


class DiscriminatorScore(_Score):
    """Two-sample classifier score.

    Attributes:
        auc: Held-out AUC of a gradient-boosted classifier telling synth from real rows.
        rows: Rows per side, half used to train and half to test.
        score: 1 - |auc - 0.5| * 2.
    """

    auc: float
    rows: int
    score: Score


class TableReconstruction(_Score):
    """Reconstruction scores of one table.

    Attributes:
        name: Table name.
        real_rows: Real row count.
        synth_rows: Synth row count, or 0 when the synth lacks the table.
        schema_match: Columns with the same name and type on both sides over all column names.
        columns: Per-leaf scores.
        pairs: Pairwise dependence errors.
        checks: Constraint satisfaction on the synth.
        keys: Key validity on the synth.
        foreign_keys: Foreign keys whose child is this table.
        discriminator: Two-sample classifier score, when both sides have enough rows.
        groups: Score per reconstruction group present in this table.
        weights: Effective weight of each group in `score`: the configured weights renormalized over the groups present.
            Tables with different groups weight the same group differently, so compare table scores group by group.
        score: Weighted mean of `groups`.
    """

    name: str
    real_rows: int
    synth_rows: int
    schema_match: Score
    columns: list[ColumnScore] = []
    pairs: list[PairScore] = []
    checks: list[CheckScore] = []
    keys: list[KeyScore] = []
    foreign_keys: list[ForeignKeyScore] = []
    discriminator: DiscriminatorScore | None = None
    groups: dict[str, Score]
    weights: dict[str, float]
    score: Score

    def relative_to(self, ceiling: TableReconstruction) -> float:
        """Returns the weighted mean of each group as a fraction of the ceiling's same group, each capped at 1.

        A group the ceiling lacks, or scores 0, counts as is. Capping per group keeps a group above its ceiling from
        masking one below it.
        """
        relative = {
            name: min(1.0, value / ceiling.groups[name]) if ceiling.groups.get(name, 0.0) > 0 else value
            for name, value in self.groups.items()
        }
        return clamp(sum(self.weights[name] * value for name, value in relative.items()))


class ReconstructionScore(_Score):
    """Reconstruction score of one dataset: synthetic data against the real data it imitates.

    Attributes:
        dataset: Catalog dataset name.
        tier: Scale tier.
        seed: Seed of the metric's sampling.
        config_version: Metrics configuration version.
        config_hash: Metrics configuration hash.
        tables: Per-table scores.
        score: Mean of the table scores.
    """

    dataset: str
    tier: str
    seed: int
    config_version: str
    config_hash: str
    tables: list[TableReconstruction]
    score: Score

    def relative_to(self, ceiling: ReconstructionScore) -> float:
        """Returns the mean over tables of each table's score relative to the real-vs-real `ceiling`.

        See `TableReconstruction.relative_to`; a table missing from the ceiling counts with its own score.
        """
        ceilings = {t.name: t for t in ceiling.tables}
        values = [t.relative_to(ceilings[t.name]) if t.name in ceilings else t.score for t in self.tables]
        return mean(values) or 0.0


class StructureItem(_Score):
    """One planted structure and whether the profile recovered it.

    Attributes:
        target: The planted structure.
        status: `recovered`, `partial`, or `not recovered`.
        credit: Recovery credit in [0, 1].
        note: Why the item was not fully recovered, when known.
    """

    target: str
    status: Literal["recovered", "partial", "not recovered"]
    credit: Score
    note: str | None = None


class StructureCategory(_Score):
    """Recovery of one category of planted structure in one table.

    Attributes:
        recall: Mean credit over planted items.
        precision: Mean credit over the profile's claims that target this category's planted columns, or None when
            the profile makes no such claims or the category has no notion of a claim.
        score: F1 of recall and precision, or the recall when precision is None.
        items: Planted items.
        claims: Number of claims counted in the precision.
        verified_true: Claims about unplanted structure that hold on the real data (left out of precision).
        verified_false: Claims about unplanted structure that do not hold (counted as 0 in precision). Only these
            counts are kept: neither the claimed nor the real values are recorded.
    """

    recall: Score
    precision: Score | None
    score: Score
    items: list[StructureItem]
    claims: int = 0
    verified_true: int = 0
    verified_false: int = 0


class TableStructure(_Score):
    """Structure recovery of one table.

    Attributes:
        name: Table name.
        categories: Categories with planted items.
        weights: Effective weight of each category in `score`, renormalized over the categories present.
        score: Weighted mean of the category scores.
    """

    name: str
    categories: dict[str, StructureCategory]
    weights: dict[str, float]
    score: Score


class StructureScore(_Score):
    """Structure recovery of one dataset: how much of the planted structure the profile states.

    Attributes:
        dataset: Catalog dataset name.
        config_version: Metrics configuration version.
        config_hash: Metrics configuration hash.
        tables: Per-table recovery.
        score: Mean of the table scores.
    """

    dataset: str
    config_version: str
    config_hash: str
    tables: list[TableStructure]
    score: Score


class GuardCheck(_Score):
    """One privacy guard check.

    Attributes:
        name: Check name.
        value: Measured value.
        threshold: Threshold from the configuration.
        passed: Whether the value is on the safe side of the threshold.
        detail: Measured values behind `value`, e.g. the synth and holdout rates.
    """

    name: str
    value: float
    threshold: float
    passed: bool
    detail: dict[str, float] = {}


class PrivacyScore(_Score):
    """Privacy guard of one dataset.

    Attributes:
        dataset: Catalog dataset name.
        config_version: Metrics configuration version.
        config_hash: Metrics configuration hash.
        checks: Every guard check, named `<table>.<check>` or `profile.<check>`.
        passed: Whether every check passed.
    """

    dataset: str
    config_version: str
    config_hash: str
    checks: list[GuardCheck]
    passed: bool


class OverallScore(_Score):
    """Reconstruction score across datasets.

    Attributes:
        config_hash: Metrics configuration hash shared by every input.
        datasets: Reconstruction score per dataset.
        relative: Score per dataset as a fraction of its ceiling, when a ceiling was given.
        score: Mean over datasets.
        relative_score: Mean of `relative`, when every dataset has a ceiling.
    """

    config_hash: str
    datasets: dict[str, Score]
    relative: dict[str, Score] = {}
    score: Score
    relative_score: Score | None = None


def mean(values: Sequence[float]) -> float | None:
    """Returns the mean of `values`, or None when empty."""
    return sum(values) / len(values) if values else None


def weighted(
    groups: Mapping[str, float | None], weights: Mapping[str, float]
) -> tuple[dict[str, float], dict[str, float], float]:
    """Returns the present groups, their effective weights, and their weighted mean.

    Weights are renormalized over the present groups, so a table without, say, foreign keys spreads that weight over
    the rest; the mean is 1 when no group is present.
    """
    present = {name: value for name, value in groups.items() if value is not None}
    total = sum(weights[name] for name in present)
    if total == 0:
        return present, {}, 1.0
    effective = {name: weights[name] / total for name in present}
    return present, effective, clamp(sum(effective[name] * value for name, value in present.items()))


def clamp(value: float) -> float:
    """Returns `value` clamped to [0, 1]."""
    return min(1.0, max(0.0, value))


def overall(
    scores: Sequence[ReconstructionScore], ceilings: Mapping[str, ReconstructionScore] | None = None
) -> OverallScore:
    """Returns the mean reconstruction score across datasets, with each as a fraction of its ceiling when given.

    Raises:
        ValueError: If the scores or ceilings were computed with different metric configurations.
    """
    ceilings = ceilings or {}
    hashes = {s.config_hash for s in [*scores, *ceilings.values()]}
    if len(hashes) != 1:
        raise ValueError("scores use different metric configurations")
    datasets = {s.dataset: s.score for s in scores}
    relative = {s.dataset: s.relative_to(ceilings[s.dataset]) for s in scores if s.dataset in ceilings}
    return OverallScore(
        config_hash=hashes.pop(),
        datasets=datasets,
        relative=relative,
        score=mean(list(datasets.values())) or 0.0,
        relative_score=mean(list(relative.values())) if len(relative) == len(datasets) else None,
    )
