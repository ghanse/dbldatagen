"""String-heavy table: formatted identifiers, mixed formats, free text, and categoricals."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from profiler_bench.catalog import gen
from profiler_bench.catalog.gen import DIGITS, HEX, UPPER, Rng
from profiler_bench.catalog.manifest import (
    Column,
    Dependency,
    Distribution,
    EnumerationCheck,
    Family,
    Format,
    FunctionalDependency,
    QuasiIdentifierSet,
    Table,
)
from profiler_bench.catalog.registry import Build, Dataset


DOMAINS = ["example.com", "mail.test"]
PHONE_US = r"\([0-9]{3}\) [0-9]{3}-[0-9]{4}"
PHONE_INTL = r"\+1-[0-9]{3}-[0-9]{3}-[0-9]{4}"
PHONE_US_SHARE = 0.7
PHONE_NULL_RATE = 0.1
ZIP5_SHARE = 0.8
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
SKU = r"[A-Z]{3}-[0-9]{4}"
COMMENT_WORDS_LAM = 8.0
COMMENT_NULL_RATE = 0.2
VOCABULARY_SIZE = 2_000
N_CITIES = 5_000
N_STATES = 50
CITY_ZIPF_S = 1.05
TIERS = ["bronze", "silver", "gold", "platinum"]
TIER_PROBABILITIES = [0.5, 0.3, 0.15, 0.05]
CHANNELS = ["web", "store", "phone"]
CHANNEL_GIVEN_TIER = [[0.7, 0.2, 0.1], [0.5, 0.35, 0.15], [0.3, 0.5, 0.2], [0.1, 0.4, 0.5]]
N_HOME_ZIPS = 100
BIRTH = ("1940-01-01", "2005-12-31")
GENDERS = ["F", "M"]


def qi_unique_fraction(n: int) -> float:
    """Returns the expected share of rows whose (home_zip, birth_date, gender) occurs once among `n` rows."""
    combinations = N_HOME_ZIPS * (gen.days(BIRTH[1]) - gen.days(BIRTH[0]) + 1) * len(GENDERS)
    return float((1 - 1 / combinations) ** (n - 1))


def sizes(n: int) -> dict[str, float]:
    """Returns the row count of the `contacts` table for base row count `n`."""
    return {"contacts": n}


def cramers_v(marginal: list[float], conditional: list[list[float]]) -> float:
    """Returns Cramér's V of the joint distribution `marginal[i] * conditional[i][j]`."""
    joint = np.asarray(marginal)[:, None] * np.asarray(conditional)
    independent = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
    phi2 = float(((joint - independent) ** 2 / independent).sum())
    return math.sqrt(phi2 / (min(joint.shape) - 1))


def build(n: int, rng: Rng, structure: Rng) -> Build:
    """Generates the `contacts` table with `n` rows."""
    names = gen.words(structure("names"), 300)
    vocabulary = gen.words(structure("vocabulary"), VOCABULARY_SIZE, 1, 3)
    cities = [w.capitalize() for w in gen.words(structure("cities"), N_CITIES, 2, 4)]
    states = gen.indexed_strings(structure("states").choice(26**2, N_STATES, replace=False), [UPPER, UPPER]).to_pylist()
    city_state = structure("city_state").integers(0, N_STATES, N_CITIES)
    home_zips = gen.indexed_strings(structure("home_zips").choice(10**5, N_HOME_ZIPS, replace=False), [DIGITS] * 5)

    phone_us = rng("phone.format").random(n) < PHONE_US_SHARE
    digits = [DIGITS] * 3
    phone = pc.if_else(
        pa.array(phone_us),
        gen.random_strings(rng("phone.us"), n, ["(", *digits, ")", " ", *digits, "-", *digits, DIGITS]),
        gen.random_strings(rng("phone.intl"), n, ["+", "1", "-", *digits, "-", *digits, "-", *digits, DIGITS]),
    )
    zip5 = gen.random_strings(rng("postal.zip5"), n, [DIGITS] * 5)
    plus4 = gen.random_strings(rng("postal.plus4"), n, ["-"] + [DIGITS] * 4)
    postal = pc.if_else(pa.array(rng("postal.format").random(n) < ZIP5_SHARE), zip5, gen.join(zip5, plus4))
    uuid = gen.random_strings(
        rng("uuid"),
        n,
        [HEX] * 8 + ["-"] + [HEX] * 4 + ["-", "4"] + [HEX] * 3 + ["-", "89ab"] + [HEX] * 3 + ["-"] + [HEX] * 12,
    )
    comment_lengths = 1 + rng("comment.length").poisson(COMMENT_WORDS_LAM, n)
    city = gen.zipf_choice(rng("city"), n, N_CITIES, CITY_ZIPF_S, structure("city.order"))
    tier = gen.choice(rng("tier"), n, TIER_PROBABILITIES)
    channel_cdf = np.cumsum(np.asarray(CHANNEL_GIVEN_TIER), axis=1)[tier]
    channel = np.minimum((rng("channel").random(n)[:, None] > channel_cdf).sum(axis=1), len(CHANNELS) - 1)

    table = pa.table(
        {
            "id": pa.array(gen.hashed_ids(np.arange(n), int(rng("id").integers(0, 2**62))), pa.int64()),
            "email": gen.join(
                gen.take(names, rng("email.first").integers(0, len(names), n)),
                gen.take([".", "_"], rng("email.sep").integers(0, 2, n)),
                gen.take(names, rng("email.last").integers(0, len(names), n)),
                gen.random_strings(rng("email.digits"), n, [DIGITS, DIGITS]),
                "@",
                gen.take(DOMAINS, rng("email.domain").integers(0, len(DOMAINS), n)),
            ),
            "phone": gen.with_nulls(phone, gen.null_mask(rng("phone.nulls"), n, PHONE_NULL_RATE)),
            "sku": gen.random_strings(rng("sku"), n, [UPPER] * 3 + ["-"] + [DIGITS] * 4),
            "uuid": uuid,
            "postal_code": postal,
            "comment": gen.with_nulls(
                gen.sentences(rng("comment"), vocabulary, comment_lengths),
                gen.null_mask(rng("comment.nulls"), n, COMMENT_NULL_RATE),
            ),
            "city": gen.take(cities, city),
            "state": gen.take(states, city_state[city]),
            "tier": gen.take(TIERS, tier),
            "channel": gen.take(CHANNELS, channel),
            "account_number": gen.random_strings(rng("account_number"), n, [DIGITS] * 10),
            "session_token": gen.random_strings(rng("session_token"), n, [HEX] * 32),
            "home_zip": gen.take(home_zips, rng("home_zip").integers(0, N_HOME_ZIPS, n)),
            "birth_date": gen.dates(rng("birth_date").integers(gen.days(BIRTH[0]), gen.days(BIRTH[1]) + 1, n)),
            "gender": gen.take(GENDERS, rng("gender").integers(0, len(GENDERS), n)),
        }
    )
    return Build({"contacts": gen.shuffle(table, rng("row_order"))}, [_truth(n)], [])


def _truth(n: int) -> Table:
    channel_probabilities = (np.asarray(TIER_PROBABILITIES) @ np.asarray(CHANNEL_GIVEN_TIER)).round(12).tolist()
    return Table(
        name="contacts",
        row_order="shuffled",
        description="One row per contact; string columns with known formats, shares, and cardinalities.",
        primary_key=("id",),
        unique_keys=(("uuid",), ("session_token",)),
        columns=(
            Column(
                name="id",
                unique=True,
                distinct=n,
                note="Hashed: a keyed 63-bit bijection of 0..n-1, so ids are distinct, non-negative, and unordered.",
            ),
            Column(
                name="email",
                privacy="direct_identifier",
                formats=(Format(regex=r"[a-z]+[._][a-z]+[0-9]{2}@(example\.com|mail\.test)", share=1.0),),
                note="Near-unique; collisions are possible and the values must not leak.",
            ),
            Column(
                name="phone",
                null_rate=PHONE_NULL_RATE,
                privacy="direct_identifier",
                formats=(
                    Format(regex=PHONE_US, share=PHONE_US_SHARE),
                    Format(regex=PHONE_INTL, share=1 - PHONE_US_SHARE),
                ),
            ),
            Column(name="sku", max_distinct=26**3 * 10**4, formats=(Format(regex=SKU, share=1.0),)),
            Column(
                name="uuid",
                unique=True,
                privacy="direct_identifier",
                formats=(Format(regex=UUID, share=1.0),),
                note="Random UUID v4; a pseudonymous record id that links rows to a person.",
            ),
            Column(
                name="postal_code",
                formats=(
                    Format(regex=r"[0-9]{5}", share=ZIP5_SHARE),
                    Format(regex=r"[0-9]{5}-[0-9]{4}", share=1 - ZIP5_SHARE),
                ),
            ),
            Column(
                name="comment",
                null_rate=COMMENT_NULL_RATE,
                privacy="free_text",
                formats=(Format(regex=r"[a-z]+( [a-z]+)*", share=1.0),),
                note=f"1 + Poisson({COMMENT_WORDS_LAM}) words drawn uniformly from {VOCABULARY_SIZE} words.",
            ),
            Column(
                name="city",
                max_distinct=N_CITIES,
                formats=(Format(regex=r"[A-Z][a-z]+", share=1.0),),
                distribution=Distribution(family=Family.ZIPF, params={"s": CITY_ZIPF_S, "n_values": N_CITIES}),
            ),
            Column(name="state", max_distinct=N_STATES, formats=(Format(regex=r"[A-Z]{2}", share=1.0),)),
            Column(
                name="tier",
                distribution=Distribution(
                    family=Family.CATEGORICAL, params={"values": TIERS, "probabilities": TIER_PROBABILITIES}
                ),
            ),
            Column(
                name="channel",
                distribution=Distribution(
                    family=Family.CATEGORICAL, params={"values": CHANNELS, "probabilities": channel_probabilities}
                ),
            ),
            Column(
                name="account_number",
                privacy="direct_identifier",
                max_distinct=10**10,
                formats=(Format(regex=r"[0-9]{10}", share=1.0),),
                note="Near-unique random digits.",
            ),
            Column(
                name="session_token",
                unique=True,
                privacy="secret",
                formats=(Format(regex=r"[0-9a-f]{32}", share=1.0),),
            ),
            Column(
                name="home_zip",
                privacy="quasi_identifier",
                max_distinct=N_HOME_ZIPS,
                formats=(Format(regex=r"[0-9]{5}", share=1.0),),
                note=f"Uniform over {N_HOME_ZIPS} zip codes.",
            ),
            Column(
                name="birth_date",
                privacy="quasi_identifier",
                distribution=Distribution(family=Family.UNIFORM, params={"low": BIRTH[0], "high": BIRTH[1]}),
            ),
            Column(
                name="gender",
                privacy="quasi_identifier",
                distribution=Distribution(
                    family=Family.CATEGORICAL, params={"values": GENDERS, "probabilities": [0.5, 0.5]}
                ),
            ),
        ),
        quasi_identifiers=(
            QuasiIdentifierSet(
                columns=("home_zip", "birth_date", "gender"),
                unique_fraction=qi_unique_fraction(n),
                mechanism="independent uniform zip, birth date, and gender",
            ),
        ),
        checks=(
            EnumerationCheck(column="tier", values=tuple(TIERS)),
            EnumerationCheck(column="channel", values=tuple(CHANNELS)),
        ),
        functional_dependencies=(FunctionalDependency(determinant=("city",), dependent=("state",)),),
        dependencies=(
            Dependency(
                columns=("tier", "channel"),
                measure="cramers_v",
                value=cramers_v(TIER_PROBABILITIES, CHANNEL_GIVEN_TIER),
                mechanism="channel drawn from a per-tier conditional distribution",
            ),
        ),
    )


DATASET = Dataset(
    name="strings",
    description="Emails, phones, SKUs, UUIDs, free text, mixed-format columns, categoricals, and near-unique ids.",
    build=build,
    sizes=sizes,
)
