"""Temporal table: seasonal timestamps, monotonic ids, inter-arrival times, and both timestamp kinds."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from profiler_bench.catalog import gen
from profiler_bench.catalog.gen import Rng
from profiler_bench.catalog.manifest import (
    Column,
    ComparisonCheck,
    Distribution,
    ExpressionCheck,
    Family,
    FunctionalDependency,
    RangeCheck,
    Table,
)
from profiler_bench.catalog.registry import Build, Dataset


START = "2024-01-01T00:00:00"
WEEKS = 52
HOUR_WEIGHTS = [1, 1, 1, 1, 1, 2, 4, 7, 9, 8, 7, 8, 10, 9, 7, 7, 8, 9, 10, 9, 6, 4, 2, 1]
WEEKDAY_WEIGHTS = [1.2, 1.1, 1.1, 1.0, 1.3, 0.7, 0.6]
FIRST_EVENT_ID = 1_000
ID_GAP_LAM = 0.5
ARRIVAL_SCALE_SECONDS = 30.0
LOCAL_OFFSET_HOURS = 5
DUE_DAYS = 30
CREATED = ("2020-01-01T00:00:00", "2024-12-31T00:00:00")
CREATED_NULL_RATE = 0.03
BIRTH = ("1940-01-01", "2005-12-31")
MICROS_PER_HOUR = 3_600_000_000
MICROS_PER_DAY = 24 * MICROS_PER_HOUR


def sizes(n: int) -> dict[str, float]:
    """Returns the row count of each table for base row count `n`."""
    return {"events": n}


def build(n: int, rng: Rng) -> Build:
    """Generates the `events` table with `n` rows in timestamp order."""
    start = gen.micros(START)
    day_weights = np.tile(WEEKDAY_WEIGHTS, WEEKS)
    day = gen.choice(rng("event_ts.day"), n, day_weights / day_weights.sum())
    hour = gen.choice(rng("event_ts.hour"), n, np.asarray(HOUR_WEIGHTS) / sum(HOUR_WEIGHTS))
    offset = rng("event_ts.offset").integers(0, MICROS_PER_HOUR, n)
    event_ts = np.sort(start + day * MICROS_PER_DAY + hour * MICROS_PER_HOUR + offset)
    event_days = event_ts // MICROS_PER_DAY
    gaps = np.maximum(1, np.round(rng("arrival_ts").exponential(ARRIVAL_SCALE_SECONDS * 1e6, n))).astype(np.int64)
    arrival = start + np.cumsum(gaps)
    created_low, created_high = gen.micros(CREATED[0]), gen.micros(CREATED[1])
    table = pa.table(
        {
            "event_id": pa.array(
                FIRST_EVENT_ID + np.cumsum(1 + rng("event_id").poisson(ID_GAP_LAM, n)) - 1, pa.int64()
            ),
            "event_ts": gen.timestamps(event_ts, utc=True),
            "local_ts": gen.timestamps(event_ts - LOCAL_OFFSET_HOURS * MICROS_PER_HOUR, utc=False),
            "event_date": gen.dates(event_days),
            "due_date": gen.dates(event_days + rng("due_date").integers(0, DUE_DAYS + 1, n)),
            "arrival_ts": gen.timestamps(arrival, utc=False),
            "created_ntz": gen.timestamps(
                rng("created_ntz").integers(created_low, created_high + 1, n),
                utc=False,
                mask=gen.null_mask(rng("created_ntz.nulls"), n, CREATED_NULL_RATE),
            ),
            "birth_date": gen.dates(rng("birth_date").integers(gen.days(BIRTH[0]), gen.days(BIRTH[1]) + 1, n)),
        }
    )
    return Build({"events": table}, [_truth(n)], [])


def _truth(n: int) -> Table:
    end = str(np.datetime64(START) + np.timedelta64(WEEKS * 7, "D"))
    return Table(
        name="events",
        description="Event log sorted by event time; ids and arrival times increase monotonically.",
        primary_key=("event_id",),
        columns=(
            Column(
                name="event_id",
                unique=True,
                distinct=n,
                monotonic="strictly_increasing",
                increments=Distribution(family=Family.POISSON, params={"lam": ID_GAP_LAM, "shift": 1}),
            ),
            Column(
                name="event_ts",
                monotonic="increasing",
                distribution=Distribution(
                    family=Family.SEASONAL,
                    params={
                        "start": START + "Z",
                        "end": end + "Z",
                        "hour_weights": HOUR_WEIGHTS,
                        "weekday_weights": WEEKDAY_WEIGHTS,
                    },
                ),
                note="Daily and weekly seasonality in UTC; minutes and seconds uniform within the hour.",
            ),
            Column(
                name="local_ts", monotonic="increasing", note=f"event_ts as UTC-{LOCAL_OFFSET_HOURS:02d}:00 wall clock."
            ),
            Column(name="event_date", monotonic="increasing", note="UTC date of event_ts."),
            Column(name="due_date", note=f"event_date plus 0..{DUE_DAYS} days, uniform."),
            Column(
                name="arrival_ts",
                unique=True,
                monotonic="strictly_increasing",
                increments=Distribution(family=Family.EXPONENTIAL, params={"scale": ARRIVAL_SCALE_SECONDS}),
                note="Poisson process: exponential inter-arrival times in seconds.",
            ),
            Column(
                name="created_ntz",
                null_rate=CREATED_NULL_RATE,
                distribution=Distribution(family=Family.UNIFORM, params={"low": CREATED[0], "high": CREATED[1]}),
            ),
            Column(
                name="birth_date",
                distribution=Distribution(family=Family.UNIFORM, params={"low": BIRTH[0], "high": BIRTH[1]}),
            ),
        ),
        checks=(
            ComparisonCheck(left="event_date", op="<=", right="due_date"),
            ExpressionCheck(sql=f"datediff(due_date, event_date) <= {DUE_DAYS}", columns=("due_date", "event_date")),
            ExpressionCheck(sql="event_date = to_date(event_ts)", columns=("event_date", "event_ts")),
            ExpressionCheck(
                sql=f"local_ts = CAST(event_ts AS TIMESTAMP_NTZ) - INTERVAL {LOCAL_OFFSET_HOURS} HOURS",
                columns=("local_ts", "event_ts"),
            ),
            RangeCheck(column="birth_date", min=BIRTH[0], max=BIRTH[1]),
        ),
        functional_dependencies=(
            FunctionalDependency(determinant=("event_ts",), dependent=("event_date",)),
            FunctionalDependency(determinant=("event_ts",), dependent=("local_ts",)),
        ),
    )


DATASET = Dataset(
    name="temporal",
    description="Seasonal timestamps, monotonic ids, exponential inter-arrivals, timestamp vs timestamp_ntz, dates.",
    build=build,
    sizes=sizes,
)
