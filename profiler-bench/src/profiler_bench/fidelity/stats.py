"""Bounded two-sample distances, dependence measures, and the gradient-boosted discriminator.

Every function here works on driver-side numpy or pandas samples and is deterministic for a given input.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from profiler_bench.catalog.verification import _cramers_v


def ks_wasserstein(
    real: np.ndarray, synth: np.ndarray, real_weights: np.ndarray | None = None, synth_weights: np.ndarray | None = None
) -> tuple[float, float]:
    """Returns the Kolmogorov-Smirnov statistic and the normalized Wasserstein-1 distance of two samples.

    The Wasserstein distance is computed over values clipped to the real 1st-99th percentile range widened by that
    range on each side, divided by the range, and capped at 1. When the range is zero the real min-max span is used,
    and when that is zero too the distance is 0 for an identical constant and 1 otherwise.

    Args:
        real: Finite real values.
        synth: Finite synthetic values.
        real_weights: Optional weight per real value, e.g. a frequency.
        synth_weights: Optional weight per synthetic value.
    """
    wr = np.ones(len(real)) if real_weights is None else np.asarray(real_weights, dtype=np.float64)
    ws = np.ones(len(synth)) if synth_weights is None else np.asarray(synth_weights, dtype=np.float64)
    support, fr, fs = _ecdfs(real, wr, synth, ws)
    ks = float(np.abs(fr - fs).max())
    q01, q99 = _weighted_quantile(real, wr, 0.01), _weighted_quantile(real, wr, 0.99)
    scale = q99 - q01 or float(real.max() - real.min())
    if scale == 0 or not math.isfinite(scale):
        return ks, 0.0 if ks == 0 else 1.0
    low, high = q01 - scale, q99 + scale
    support, fr, fs = _ecdfs(np.clip(real, low, high), wr, np.clip(synth, low, high), ws)
    w1 = float((np.abs(fr - fs)[:-1] * np.diff(support)).sum())
    return ks, min(1.0, w1 / scale)


def _ecdfs(a: np.ndarray, wa: np.ndarray, b: np.ndarray, wb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support = np.union1d(a, b)
    fa = np.cumsum(np.bincount(np.searchsorted(support, a), weights=wa, minlength=len(support))) / wa.sum()
    fb = np.cumsum(np.bincount(np.searchsorted(support, b), weights=wb, minlength=len(support))) / wb.sum()
    return support, fa, fb


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][min(int(np.searchsorted(cumulative, q)), len(values) - 1)])


def top_values(real: pd.Series, k: int) -> list[Any]:
    """Returns the `k` most frequent values of `real`, ties broken by their string form."""
    counts = real.value_counts(sort=False)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
    return [value for value, _ in ranked[:k]]


def tvd(real: pd.Series, synth: pd.Series, buckets: Sequence[Any]) -> float:
    """Returns the total variation distance between the shares of `buckets` plus an `other` bucket."""
    return 0.5 * float(np.abs(bucket_shares(real, buckets) - bucket_shares(synth, buckets)).sum())


def bucket_shares(values: pd.Series, buckets: Sequence[Any]) -> np.ndarray:
    """Returns the share of `values` in each of `buckets`, then the share of the rest."""
    if len(values) == 0:
        return np.zeros(len(buckets) + 1)
    counts = values.value_counts()
    known = np.array([float(counts.get(b, 0)) for b in buckets])
    return np.append(known, len(values) - known.sum()) / len(values)


def spearman(x: pd.Series, y: pd.Series) -> float:
    """Returns the Spearman rank correlation, or 0 when either side is constant."""
    return pearson(x.rank(), y.rank())


def _scaled(values: np.ndarray) -> np.ndarray:
    """Returns `values` divided by their largest magnitude, so squares cannot overflow; scale-free measures keep."""
    largest = float(np.abs(values).max()) if len(values) else 0.0
    return values / largest if largest > 0 else values


def pearson(x: pd.Series, y: pd.Series) -> float:
    """Returns the Pearson correlation, or 0 when either side is constant or there are fewer than two rows."""
    a, b = _scaled(x.to_numpy(dtype=np.float64)), _scaled(y.to_numpy(dtype=np.float64))
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Returns Cramer's V of two categorical series, or 0 when either has a single category."""
    if x.nunique() < 2 or y.nunique() < 2:
        return 0.0
    return _cramers_v(pd.crosstab(x.astype(str), y.astype(str)).to_numpy(dtype=np.float64))


def correlation_ratio(x: pd.Series, y: pd.Series, bins: int, numeric_x: bool) -> float:
    """Returns eta of numeric `y` given `x`, with numeric `x` cut into quantile bins, or 0 when `y` is constant."""
    values = _scaled(y.to_numpy(dtype=np.float64))
    if len(values) < 2 or values.std() == 0:
        return 0.0
    groups = pd.qcut(x, bins, labels=False, duplicates="drop") if numeric_x else x.astype(str)
    target = pd.Series(values, index=x.index)
    means = target.groupby(groups).transform("mean").to_numpy()
    return math.sqrt(min(1.0, float(((means - values.mean()) ** 2).sum() / ((values - values.mean()) ** 2).sum())))


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Returns the area under the ROC curve of `scores` for binary `labels`, with ties counted as one half."""
    ranks = pd.Series(scores).rank().to_numpy()
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


class Encoder:
    """Encodes sampled rows as numeric features fitted on the real sample.

    Numeric columns become their rank in the real sample, in [0, 1], plus a null indicator. Categorical and string
    columns become a one-hot of the top real values, `other`, and null, scaled so a mismatch adds 1 to the squared
    distance; string columns also get a length rank. String values are also factorized so exact equality can count in
    distances to the closest record.

    Args:
        real: Real sample with one column per feature.
        kinds: Kind of each column: `numeric`, `categorical`, or `string`.
        top: Number of top values one-hot encoded.
    """

    def __init__(self, real: pd.DataFrame, kinds: dict[str, str], top: int) -> None:
        self.kinds = kinds
        self.sorted = {c: np.sort(finite(real[c])) for c, k in kinds.items() if k == "numeric"}
        self.lengths = {c: np.sort(_lengths(real[c].dropna())) for c, k in kinds.items() if k == "string"}
        self.buckets = {c: top_values(real[c].dropna(), top) for c, k in kinds.items() if k != "numeric"}

    def features(self, frame: pd.DataFrame) -> np.ndarray:
        """Returns the feature matrix of `frame`."""
        blocks = [np.zeros((len(frame), 0))]
        for column, kind in self.kinds.items():
            values = frame[column]
            if kind == "numeric":
                blocks.append(_rank_block(values, self.sorted[column]))
                continue
            blocks.append(_one_hot(values, self.buckets[column]))
            if kind == "string":
                blocks.append(_rank_block(values.dropna().map(len).reindex(values.index), self.lengths[column]))
        return np.hstack(blocks)

    def tree_features(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns features for a tree model and which of them are categorical.

        Numeric columns become their real rank, categorical and string columns the index of their top-value bucket
        (`other` last), and string columns also their length rank; nulls are NaN.
        """
        blocks: list[np.ndarray] = []
        categorical: list[bool] = []
        for column, kind in self.kinds.items():
            values = frame[column]
            if kind == "numeric":
                blocks.append(_ranks(values, self.sorted[column]))
                categorical.append(False)
                continue
            index = {b: i for i, b in enumerate(self.buckets[column])}
            other = float(len(index))
            blocks.append(np.array([np.nan if pd.isna(v) else index.get(v, other) for v in values], dtype=np.float64))
            categorical.append(True)
            if kind == "string":
                blocks.append(_ranks(values.dropna().map(len).reindex(values.index), self.lengths[column]))
                categorical.append(False)
        matrix = np.column_stack(blocks) if blocks else np.zeros((len(frame), 0))
        return matrix, np.array(categorical, dtype=bool)

    def codes(self, *frames: pd.DataFrame) -> list[list[np.ndarray]]:
        """Returns, for each string column, the factorized values of each frame over a shared code book."""
        columns = [c for c, k in self.kinds.items() if k == "string"]
        result = []
        for column in columns:
            joined = pd.concat([f[column].astype(object) for f in frames], ignore_index=True)
            codes, _ = pd.factorize(joined, use_na_sentinel=True)
            bounds = np.cumsum([0, *(len(f) for f in frames)])
            result.append([codes[bounds[i] : bounds[i + 1]] for i in range(len(frames))])
        return result


def finite(values: pd.Series) -> np.ndarray:
    """Returns the finite numeric values of `values`; nulls, NaN, and infinities are dropped."""
    numbers: np.ndarray = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    result: np.ndarray = numbers[np.isfinite(numbers)]
    return result


def _lengths(values: pd.Series) -> np.ndarray:
    lengths: np.ndarray = values.map(len).to_numpy(dtype=np.float64)
    return lengths


def _ranks(values: pd.Series, reference: np.ndarray) -> np.ndarray:
    numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    ranks: np.ndarray = np.searchsorted(reference, np.nan_to_num(numbers), side="right") / max(len(reference), 1)
    return np.where(np.isnan(numbers), np.nan, ranks)


def _rank_block(values: pd.Series, reference: np.ndarray) -> np.ndarray:
    numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    missing = np.isnan(numbers)
    ranks = np.searchsorted(reference, np.nan_to_num(numbers), side="right") / max(len(reference), 1)
    return np.column_stack([np.where(missing, 0.5, ranks), missing.astype(np.float64)])


def _one_hot(values: pd.Series, buckets: Sequence[Any]) -> np.ndarray:
    index = {b: i for i, b in enumerate(buckets)}
    other, null = len(buckets), len(buckets) + 1
    positions = np.array([null if pd.isna(v) else index.get(v, other) for v in values], dtype=np.int64)
    block = np.zeros((len(values), len(buckets) + 2))
    block[np.arange(len(values)), positions] = 1.0 / math.sqrt(2.0)
    return block


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Returns the squared Euclidean distance between every row of `a` and every row of `b`, rounded to 1e-9."""
    distances: np.ndarray = (a**2).sum(axis=1)[:, None] + (b**2).sum(axis=1)[None, :] - 2.0 * a @ b.T
    rounded: np.ndarray = np.round(np.maximum(distances, 0.0), 9)
    return rounded


def mismatches(codes_a: list[np.ndarray], codes_b: list[np.ndarray]) -> np.ndarray:
    """Returns the number of string columns whose values differ between every row of one side and the other."""
    total = np.zeros((len(codes_a[0]) if codes_a else 0, len(codes_b[0]) if codes_b else 0))
    for a, b in zip(codes_a, codes_b, strict=True):
        total += a[:, None] != b[None, :]
    return total


LOG_LOSS_EPS = 1e-6


@dataclass(frozen=True)
class Discrimination:
    """Held-out result of a two-sample classifier telling synthetic (positive) from real rows.

    Attributes:
        auc: Area under the ROC curve on the test half.
        log_loss: Mean binary cross-entropy on the test half, with probabilities clipped to
            [`LOG_LOSS_EPS`, 1 - `LOG_LOSS_EPS`]; ln 2 for a classifier that always says 0.5.
        n_test: Test rows per side.
        histogram: Counts of the predicted synthetic probability in equal-width bins over [0, 1], per side.
    """

    auc: float
    log_loss: float
    n_test: int
    histogram: dict[str, list[int]]


FIT, VALIDATION, TEST, DROPPED = 0, 1, 2, -1
VALIDATION_SHARE = 5


def entity_folds(entities: Sequence[pd.Series], seed: int) -> np.ndarray:
    """Returns the fold of each row (`FIT`, `VALIDATION`, `TEST`, or `DROPPED`) from seeded hashes of its entities.

    Each entity column (say a row's product and its order) hashes every value into the train or the test half. A row
    is in the test fold when all its entities are in the test half, in training when all are in the train half, and
    dropped otherwise, so no entity has rows on both sides of the split. One training row in `VALIDATION_SHARE`, by a
    second hash of the first entity column, is held out for choosing the boosting iteration. Equal entity values on
    both sides of a comparison land in the same fold, so identical samples cannot be told apart. `entities` must not
    be empty; pass the row position to split by row.
    """

    def hashed(values: pd.Series, salt: int) -> np.ndarray:
        key = f"{(seed * 2 + salt) % 10**16:016d}"
        hashes: np.ndarray = pd.util.hash_pandas_object(values.astype(str), index=False, hash_key=key).to_numpy()
        return hashes

    train = np.ones(len(entities[0]), dtype=bool)
    test = np.ones(len(entities[0]), dtype=bool)
    for values in entities:
        half = hashed(values, 0) % 2 == 0
        train &= half
        test &= ~half
    validation = train & (hashed(entities[0], 1) % VALIDATION_SHARE == 0)
    folds = np.full(len(train), DROPPED)
    folds[train] = FIT
    folds[validation] = VALIDATION
    folds[test] = TEST
    return folds


def discriminate(
    real: np.ndarray,
    synth: np.ndarray,
    categorical: np.ndarray,
    folds: tuple[np.ndarray, np.ndarray],
    seed: int,
    max_iter: int,
    min_samples_leaf: int,
    bins: int,
) -> Discrimination | None:
    """Trains a gradient-boosted classifier on the fit fold of each side and scores it on the test fold.

    Each fold keeps as many rows of each side as the smaller side has, so the classes are balanced and the trivial
    log-loss is ln 2. The boosting iteration, including none (the constant prior 0.5), is the one with the lowest
    log-loss on the validation fold. That fold holds out whole entities, so a classifier that only memorizes entity
    values (a product's sample-specific price) stops early instead of turning overconfident, and two samples of one
    distribution score a log-loss near ln 2. The classifier runs on one thread: parallel histogram building is many
    times slower beside a local Spark session.

    Args:
        real: Real feature rows.
        synth: Synthetic feature rows.
        categorical: Which feature columns hold category codes.
        folds: Fold of each real and each synthetic row, see `entity_folds`.
        seed: Seed of the classifier.
        max_iter: Most boosting iterations.
        min_samples_leaf: Fewest training rows per leaf.
        bins: Histogram bins of the predicted probabilities.

    Returns:
        None when the fit or test fold has no row of one side.
    """

    def balanced(fold: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = np.flatnonzero(folds[0] == fold), np.flatnonzero(folds[1] == fold)
        n = min(len(a), len(b))
        labels = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.int64)
        return np.vstack([real[a[:n]], synth[b[:n]]]), labels

    (x_fit, y_fit), (x_val, y_val), (x_test, y_test) = balanced(FIT), balanced(VALIDATION), balanced(TEST)
    if len(y_fit) == 0 or len(y_test) == 0:
        return None
    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        min_samples_leaf=min_samples_leaf,
        early_stopping=False,
        categorical_features=categorical,
        random_state=seed,
    )
    with threadpool_limits(1):
        model.fit(x_fit, y_fit)
        best = max_iter
        if len(y_val):
            losses = [log_loss(np.full(len(y_val), 0.5), y_val)]
            losses += [log_loss(p[:, 1], y_val) for p in model.staged_predict_proba(x_val)]
            best = int(np.argmin(losses))
        scores = np.full(len(y_test), 0.5)
        for iteration, staged in enumerate(model.staged_predict_proba(x_test), start=1):
            if iteration == best:
                scores = staged[:, 1]
                break
    edges = np.linspace(0.0, 1.0, bins + 1)
    return Discrimination(
        auc=auc(scores, y_test),
        log_loss=log_loss(scores, y_test),
        n_test=len(y_test) // 2,
        histogram={
            side: np.histogram(scores[y_test == label], bins=edges)[0].tolist()
            for side, label in (("real", 0), ("synth", 1))
        },
    )


def log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Returns the mean binary cross-entropy of `probabilities` of the positive class, clipped by `LOG_LOSS_EPS`."""
    p = np.clip(probabilities, LOG_LOSS_EPS, 1 - LOG_LOSS_EPS)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
