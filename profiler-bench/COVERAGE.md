# Catalog coverage

`python -m profiler_bench.catalog list` is the authoritative dataset inventory. There are **6 datasets** with
**11 tables** and **288 top-level columns**, each at three tiers (`xs`, `s`, `m`) and any seed. Every claim in a
manifest is checked against the data by `profiler_bench.catalog.verify`; the test suite runs it on every dataset at
`xs`.

| Dataset | Tables | Structures planted |
|---|---|---|
| `retail` | `customers`, `products`, `orders`, `order_items` | Single and composite primary keys (`order_items(order_id, line_no)`); unique keys (`email`, `sku`); three foreign keys with Zipf fan-out (s=1.1 customers, s=1.3 products, parents ranked in random order, some with no children) and shifted-Poisson fan-out (1 + Poisson(3) items per order); `order_date <= ship_date`; `line_total = unit_price * quantity` (exact decimals); status, segment, country, and category enumerations; `ship_date` null exactly for unshipped statuses; functional dependencies `country -> region`, `category -> department`, `product_id -> unit_price`; lognormal prices; uniform dates; sensitive unique email |
| `numeric` | `measurements` | Normal (5% nulls), lognormal, two-component normal mixture, Pareto (alpha=1.5, infinite variance), Poisson and overdispersed negative binomial counts (2% nulls), low-cardinality integer categorical, float32 uniform in [0, 1]; Gaussian copula over normal, lognormal, and cubed-normal marginals with declared Spearman values; linear Pearson pair (0.8); nonlinear `y = x**2 + noise` with Pearson 0 and correlation ratio 0.98; `range_low <= range_high` |
| `strings` | `contacts` | Emails, US phone numbers in two formats (70/30, 10% nulls), SKUs `[A-Z]{3}-[0-9]{4}`, UUID v4, ZIP and ZIP+4 (80/20), free text (1 + Poisson(8) words from 2,000), Zipf city over 5,000 values with `city -> state`, low-cardinality tier and channel with Cramér's V from a conditional table; near-unique sensitive ids (`email`, `account_number`, `session_token`) that must not leak |
| `temporal` | `events` | UTC `timestamp` with hour-of-day and day-of-week seasonality, matching `timestamp_ntz` wall clock at UTC-05:00, UTC `date`; strictly increasing ids with 1 + Poisson(0.5) gaps; Poisson-process `timestamp_ntz` arrivals with exponential(30 s) inter-arrival times; `event_date <= due_date <= event_date + 30`; uniform `timestamp_ntz` with nulls; uniform birth dates |
| `nested` | `profiles`, `wide` | Struct with nullable fields (5% null struct, 10% null field), two-level nested struct, `array<string>` with Poisson length and categorical elements (10% null arrays), `array<double>` with uniform length 0..5, `map<string,bigint>` with Binomial(20, 0.15) length and Poisson values, `array<struct>` with Poisson length; `wide`: 200 columns (120 normal floats with ten declared Pearson pairs, 50 Poisson ints, 30 categoricals with 2..20 values), null rates from 0 to 50% |
| `edge` | `edge_cases`, `empty` | All-null string, int, and double columns; constant string and int; `decimal(38,10)` and `decimal(38,0)` with planted ± maximum values; uniform int64 with planted min, max, and zero; full-range int8; NaN, +inf, -inf shares with 5% nulls; doubles spanning 1e-300..1e300 with planted ±max and the smallest subnormal; empty strings vs nulls; leading and trailing spaces; accents, CJK, Cyrillic, Arabic, emoji, combining marks, zero-width space; nullable boolean; dates 0001-01-01..9999-12-31; binary with uniform length; a zero-row table |

## What verification proves

Structural claims hold exactly: row counts, Spark types, primary and unique keys, foreign-key coverage, check
constraints (Spark SQL), functional dependencies, exact distinct counts and bounds, planted extremes, and null rates
of 0 or 1. Statistical claims hold within about five standard errors for the sample size: distribution parameters,
format and category shares, fan-out, null rates, special-value shares, and dependency strengths. Negative tests
tamper with a manifest or the data and check that verification reports it.

Declared values are the generating parameters, not statistics of the sample. A profiler is scored on how well it
recovers what was planted, and the sample's own noise is part of what it sees.

## Not covered yet

- Real public datasets. The download-and-checksum hook exists; no dataset is registered.
- Cross-table aggregate constraints, e.g. an order total equal to the sum of its lines.
- Composite foreign keys, self-references (hierarchies), many-to-many bridge tables, and orphaned child keys
  (coverage below 1).
- Conditional and multivariate structure beyond pairs: conditional null patterns other than `ship_date`, Simpson
  style confounding, three-way interactions, and categorical-to-numeric conditional distributions.
- Temporal: trends and change points, holidays, wall-clock data from named time zones with DST, out-of-order
  arrival, late data, and timestamps before 1900 or after 2100.
- Strings: JSON and other semi-structured text, mixed-case and locale-specific formats, very long strings, and values
  with embedded newlines or delimiters.
- Types: `interval`, `time`, `variant`, unsigned integers, `float16`, and maps with null values or struct keys.
- Skew at the partition level: files of uneven size, many small files, and sorted vs unsorted storage.
- Duplicated rows and near-duplicates, and tables wider than 200 columns or deeper than two nesting levels.
- Privacy ground truth beyond the `sensitive` flag: quasi-identifier combinations and rare categories that make a
  row unique.

These gaps are explicit; they are not counted as covered structures.
