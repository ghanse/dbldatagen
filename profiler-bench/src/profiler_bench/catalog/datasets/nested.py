"""Nested and wide tables: struct, array, and map columns, and a 200-column table."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyarrow as pa

from profiler_bench.catalog import gen
from profiler_bench.catalog.gen import DIGITS, Rng
from profiler_bench.catalog.manifest import Column, Dependency, Distribution, Family, Format, Table
from profiler_bench.catalog.registry import Build, Dataset


ADDRESS_NULL_RATE = 0.05
POSTAL_NULL_RATE = 0.1
N_CITIES = 200
SUFFIXES = ["St", "Ave", "Rd"]
TAGS = ["new", "vip", "churn_risk", "promo", "mobile", "beta", "referral", "dormant", "b2b", "eu", "us", "apac"]
TAG_WEIGHTS = [12, 6, 3, 9, 10, 2, 4, 5, 3, 7, 8, 4]
TAGS_NULL_RATE = 0.1
TAGS_LAM = 2.0
SCORES_MAX_LENGTH = 5
ATTRIBUTE_KEYS = [f"attr_{i:02d}" for i in range(20)]
ATTRIBUTE_P = 0.15
ATTRIBUTE_LAM = 10.0
OPERATING_SYSTEMS = ["android", "ios", "web"]
OS_PROBABILITIES = [0.45, 0.4, 0.15]
MAJOR_VERSIONS = (1, 17)
MINOR_LAM = 3.0
HISTORY_LAM = 1.5
KINDS = ["view", "cart", "purchase", "refund"]
KIND_PROBABILITIES = [0.6, 0.25, 0.12, 0.03]
AMOUNT_SCALE = 20.0
WIDE_FLOATS, WIDE_INTS, WIDE_STRINGS = 120, 50, 30
WIDE_PAIRS = 10
WIDE_NULL_RATES = [0.0, 0.0, 0.01, 0.1, 0.5]


def _dist(family: Family, **params: object) -> Distribution:
    return Distribution(family=family, params=params)


def _categorical(values: Sequence[object], weights: Sequence[float]) -> Distribution:
    probabilities = (np.asarray(weights, dtype=np.float64) / sum(weights)).tolist()
    return _dist(Family.CATEGORICAL, values=values, probabilities=probabilities)


def _list(lengths: np.ndarray, values: pa.Array, mask: np.ndarray | None = None) -> pa.Array:
    return pa.ListArray.from_arrays(gen.list_offsets(lengths), values, mask=None if mask is None else pa.array(mask))


def build(n: int, rng: Rng) -> Build:
    """Generates the `profiles` table with `n` rows and the `wide` table with `max(500, n // 10)` rows."""
    profiles, profiles_truth = _profiles(n, rng)
    wide, wide_truth = _wide(max(500, n // 10), rng)
    return Build({"profiles": profiles, "wide": wide}, [profiles_truth, wide_truth], [])


def _profiles(n: int, rng: Rng) -> tuple[pa.Table, Table]:
    streets = [w.capitalize() for w in gen.words(rng("address.street.names"), 300)]
    cities = [w.capitalize() for w in gen.words(rng("address.city.names"), N_CITIES)]
    address = pa.StructArray.from_arrays(
        [
            gen.join(
                pa.array(rng("address.street.number").integers(1, 10_000, n).astype(str)),
                gen.take(streets, rng("address.street").integers(0, len(streets), n)),
                gen.take(SUFFIXES, rng("address.suffix").integers(0, len(SUFFIXES), n)),
                sep=" ",
            ),
            gen.take(cities, rng("address.city").integers(0, N_CITIES, n)),
            gen.with_nulls(
                gen.random_strings(rng("address.postal_code"), n, [DIGITS] * 5),
                gen.null_mask(rng("address.postal_code.nulls"), n, POSTAL_NULL_RATE),
            ),
        ],
        names=["street", "city", "postal_code"],
        mask=pa.array(gen.null_mask(rng("address.nulls"), n, ADDRESS_NULL_RATE)),
    )

    tags_null = gen.null_mask(rng("tags.nulls"), n, TAGS_NULL_RATE)
    tag_lengths = np.where(tags_null, 0, rng("tags.length").poisson(TAGS_LAM, n))
    tag_probabilities = np.asarray(TAG_WEIGHTS) / sum(TAG_WEIGHTS)
    tags = _list(
        tag_lengths,
        gen.take(TAGS, gen.choice(rng("tags.element"), int(tag_lengths.sum()), tag_probabilities)),
        tags_null,
    )

    score_lengths = rng("scores.length").integers(0, SCORES_MAX_LENGTH + 1, n)
    scores = _list(score_lengths, pa.array(rng("scores.element").standard_normal(int(score_lengths.sum()))))

    present = rng("attributes.keys").random((n, len(ATTRIBUTE_KEYS)), dtype=np.float32) < ATTRIBUTE_P
    rows, key_index = np.nonzero(present)
    attributes = pa.MapArray.from_arrays(
        gen.list_offsets(present.sum(axis=1)),
        gen.take(ATTRIBUTE_KEYS, key_index),
        pa.array(rng("attributes.values").poisson(ATTRIBUTE_LAM, len(rows)), pa.int64()),
    )

    device = pa.StructArray.from_arrays(
        [
            gen.take(OPERATING_SYSTEMS, gen.choice(rng("device.os"), n, OS_PROBABILITIES)),
            pa.StructArray.from_arrays(
                [
                    pa.array(rng("device.major").integers(MAJOR_VERSIONS[0], MAJOR_VERSIONS[1] + 1, n), pa.int32()),
                    pa.array(rng("device.minor").poisson(MINOR_LAM, n), pa.int32()),
                ],
                names=["major", "minor"],
            ),
        ],
        names=["os", "version"],
    )

    history_lengths = rng("history.length").poisson(HISTORY_LAM, n)
    events = int(history_lengths.sum())
    history = _list(
        history_lengths,
        pa.StructArray.from_arrays(
            [
                gen.take(KINDS, gen.choice(rng("history.kind"), events, KIND_PROBABILITIES)),
                pa.array(rng("history.amount").exponential(AMOUNT_SCALE, events)),
            ],
            names=["kind", "amount"],
        ),
    )

    table = pa.table(
        {
            "id": pa.array(np.arange(1, n + 1), pa.int64()),
            "address": address,
            "tags": tags,
            "scores": scores,
            "attributes": attributes,
            "device": device,
            "history": history,
        }
    )
    truth = Table(
        name="profiles",
        description="User profiles with struct, array, map, and array-of-struct columns.",
        primary_key=("id",),
        columns=(
            Column(name="id", unique=True, distinct=n, distribution=_dist(Family.SEQUENCE, start=1, step=1)),
            Column(
                name="address",
                null_rate=ADDRESS_NULL_RATE,
                fields=(
                    Column(
                        name="street",
                        formats=(Format(regex=r"[1-9][0-9]{0,3} [A-Z][a-z]+ (St|Ave|Rd)", share=1.0),),
                    ),
                    Column(
                        name="city",
                        max_distinct=N_CITIES,
                        formats=(Format(regex=r"[A-Z][a-z]+", share=1.0),),
                        note=f"Uniform over {N_CITIES} cities.",
                    ),
                    Column(
                        name="postal_code",
                        null_rate=POSTAL_NULL_RATE,
                        formats=(Format(regex=r"[0-9]{5}", share=1.0),),
                    ),
                ),
            ),
            Column(
                name="tags",
                null_rate=TAGS_NULL_RATE,
                length=_dist(Family.POISSON, lam=TAGS_LAM),
                element=Column(name="element", max_distinct=len(TAGS), distribution=_categorical(TAGS, TAG_WEIGHTS)),
            ),
            Column(
                name="scores",
                length=_dist(Family.DISCRETE_UNIFORM, low=0, high=SCORES_MAX_LENGTH),
                element=Column(name="element", distribution=_dist(Family.NORMAL, mean=0.0, std=1.0)),
            ),
            Column(
                name="attributes",
                length=_dist(Family.BINOMIAL, n=len(ATTRIBUTE_KEYS), p=ATTRIBUTE_P),
                key=Column(
                    name="key",
                    max_distinct=len(ATTRIBUTE_KEYS),
                    formats=(Format(regex=r"attr_[0-9]{2}", share=1.0),),
                    note=f"Each key is present independently with probability {ATTRIBUTE_P}.",
                ),
                value=Column(name="value", distribution=_dist(Family.POISSON, lam=ATTRIBUTE_LAM)),
            ),
            Column(
                name="device",
                fields=(
                    Column(name="os", distribution=_categorical(OPERATING_SYSTEMS, OS_PROBABILITIES)),
                    Column(
                        name="version",
                        fields=(
                            Column(
                                name="major",
                                distribution=_dist(
                                    Family.DISCRETE_UNIFORM, low=MAJOR_VERSIONS[0], high=MAJOR_VERSIONS[1]
                                ),
                            ),
                            Column(name="minor", distribution=_dist(Family.POISSON, lam=MINOR_LAM)),
                        ),
                    ),
                ),
            ),
            Column(
                name="history",
                length=_dist(Family.POISSON, lam=HISTORY_LAM),
                element=Column(
                    name="element",
                    fields=(
                        Column(name="kind", distribution=_categorical(KINDS, KIND_PROBABILITIES)),
                        Column(name="amount", distribution=_dist(Family.EXPONENTIAL, scale=AMOUNT_SCALE)),
                    ),
                ),
            ),
        ),
    )
    return table, truth


def _wide(n: int, rng: Rng) -> tuple[pa.Table, Table]:
    params = rng("wide.params")
    names = [f"c{i:03d}" for i in range(WIDE_FLOATS + WIDE_INTS + WIDE_STRINGS)]
    null_rates = [float(r) for r in params.choice(WIDE_NULL_RATES, len(names))]
    data: dict[str, pa.Array] = {}
    columns: list[Column] = []
    dependencies: list[Dependency] = []
    base = rng("wide.base").standard_normal((n, WIDE_FLOATS))
    for i in range(WIDE_FLOATS):
        mean, std = round(float(params.uniform(-100, 100)), 4), round(float(params.uniform(0.5, 20)), 4)
        z = base[:, i]
        if i % 2 == 1 and i < 2 * WIDE_PAIRS:
            rho = round(float(params.uniform(-0.9, 0.9)), 4)
            z = rho * base[:, i - 1] + np.sqrt(1 - rho**2) * z
            dependencies.append(
                Dependency(columns=(names[i - 1], names[i]), measure="pearson", value=rho, mechanism="linear Gaussian")
            )
        data[names[i]] = pa.array(mean + std * z, mask=gen.null_mask(rng(f"wide.{names[i]}.nulls"), n, null_rates[i]))
        columns.append(
            Column(name=names[i], null_rate=null_rates[i], distribution=_dist(Family.NORMAL, mean=mean, std=std))
        )
    for i in range(WIDE_FLOATS, WIDE_FLOATS + WIDE_INTS):
        lam = round(float(params.uniform(0.5, 50)), 4)
        values = rng(f"wide.{names[i]}").poisson(lam, n)
        data[names[i]] = pa.array(
            values, pa.int32(), mask=gen.null_mask(rng(f"wide.{names[i]}.nulls"), n, null_rates[i])
        )
        columns.append(Column(name=names[i], null_rate=null_rates[i], distribution=_dist(Family.POISSON, lam=lam)))
    for i in range(WIDE_FLOATS + WIDE_INTS, len(names)):
        k = int(params.integers(2, 21))
        probabilities = np.round(params.dirichlet(np.full(k, 2.0)), 6)
        probabilities[-1] = round(1.0 - float(probabilities[:-1].sum()), 6)
        labels = [f"{names[i]}_{j}" for j in range(k)]
        array = gen.take(labels, gen.choice(rng(f"wide.{names[i]}"), n, probabilities / probabilities.sum()))
        data[names[i]] = gen.with_nulls(array, gen.null_mask(rng(f"wide.{names[i]}.nulls"), n, null_rates[i]))
        distribution = _dist(Family.CATEGORICAL, values=labels, probabilities=probabilities.tolist())
        columns.append(Column(name=names[i], null_rate=null_rates[i], max_distinct=k, distribution=distribution))
    truth = Table(
        name="wide",
        description="200 columns: normal floats (ten correlated pairs), Poisson ints, and categorical strings.",
        columns=tuple(columns),
        dependencies=tuple(dependencies),
    )
    return pa.table(data), truth


DATASET = Dataset(
    name="nested",
    description="Struct, array, map, and array-of-struct columns with length distributions, and a 200-column table.",
    build=build,
)
