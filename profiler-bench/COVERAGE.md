# Catalog coverage

`python -m profiler_bench.catalog list` is the authoritative dataset inventory. There are **6 datasets** with
**11 tables** and **291 top-level columns**, each at three tiers (`xs`, `s`, `m`) and any seed. Every claim in a
manifest is checked against the data by `profiler_bench.catalog.verify`; the test suite runs it on every dataset at
`xs`. Tiers `s` and `m` write every table as 4 and 16 parquet part files. The structure seed fixes each
distribution and its manifest; the sample seed draws the rows, and the tests verify several sample seeds.

| Dataset | Tables | Structures planted |
|---|---|---|
| `retail` | `customers`, `products`, `orders`, `order_items` | Single and composite primary keys (`order_items(order_id, line_no)`); unique keys (`email`, `sku`); three foreign keys with Zipf fan-out (s=1.1 customers, s=1.3 products, parents ranked in random order, some with no children) and fixed-total fan-out (1 + Binomial(3·orders, 1/orders) items per order, about 1 + Poisson(3), so the table size is exact); `order_date <= ship_date`; `line_total = unit_price * quantity` (exact decimals); status, segment, country, and category enumerations; `ship_date` null exactly for unshipped statuses; functional dependencies `country -> region`, `category -> department`, `product_id -> unit_price`; lognormal prices; uniform dates; unique email (`direct_identifier`); hashed 63-bit `customer_id` (unique, non-sequential); every table stored in shuffled row order |
| `numeric` | `measurements` | Normal (5% nulls), lognormal, two-component normal mixture, Pareto (alpha=1.5, infinite variance), Poisson and overdispersed negative binomial counts (2% nulls), low-cardinality integer categorical, float32 uniform in [0, 1]; Gaussian copula over normal, lognormal, and cubed-normal marginals with declared Spearman values; linear Pearson pair (0.8); nonlinear `y = x**2 + noise` with Pearson 0 and correlation ratio 0.98; `range_low <= range_high` |
| `strings` | `contacts` | Emails, US phone numbers in two formats (70/30, 10% nulls), SKUs `[A-Z]{3}-[0-9]{4}`, UUID v4, ZIP and ZIP+4 (80/20), free text (1 + Poisson(8) words from 2,000), Zipf city over 5,000 values with `city -> state`, low-cardinality tier and channel with Cramér's V from a conditional table; privacy classes: `direct_identifier` (`email`, `phone`, `uuid`, `account_number`), `secret` (`session_token`), `free_text` (`comment`); a planted quasi-identifier set `(home_zip, birth_date, gender)` (100 zips × 24,107 days × 2) whose declared unique-row share is `(1 - 1/K)**(n-1)` (0.999 at `xs`, 0.979 at `s`, 0.66 at `m`); hashed `id` and shuffled row order |
| `temporal` | `events` | UTC `timestamp` with hour-of-day and day-of-week seasonality, matching `timestamp_ntz` wall clock at UTC-05:00, UTC `date`; strictly increasing ids with 1 + Poisson(0.5) gaps; Poisson-process `timestamp_ntz` arrivals with exponential(30 s) inter-arrival times; `event_date <= due_date <= event_date + 30`; uniform `timestamp_ntz` with nulls; uniform birth dates |
| `nested` | `profiles`, `wide` | Struct with nullable fields (5% null struct, 10% null field), two-level nested struct, `array<string>` with Poisson length and categorical elements (10% null arrays), `array<double>` with uniform length 0..5, `map<string,bigint>` with Binomial(20, 0.15) length and Poisson values, `array<struct>` with Poisson length; `wide`: 200 columns (120 normal floats with ten declared Pearson pairs, 50 Poisson ints, 30 categoricals with 2..20 values), null rates from 0 to 50% |
| `edge` | `edge_cases`, `empty` | All-null string, int, and double columns; constant string and int; `decimal(38,10)` and `decimal(38,0)` with planted ± maximum values; uniform int64 with planted min, max, and zero; full-range int8; NaN, +inf, -inf shares with 5% nulls; doubles spanning 1e-300..1e300 with planted ±max and the smallest subnormal; empty strings vs nulls; leading and trailing spaces; accents, CJK, Cyrillic, Arabic, emoji, combining marks, zero-width space; nullable boolean; dates 0001-01-01..9999-12-31; binary with uniform length; a zero-row table |

## What verification proves

Structural claims hold exactly: row counts, Spark types, primary and unique keys, foreign-key coverage, check
constraints (Spark SQL), functional dependencies, exact distinct counts and bounds, planted extremes, null rates of
0 or 1, absence of undeclared NaN or infinite values, and storage order (part files in path order, rows in file
order). Statistical claims are tested at `ALPHA = 1e-5` (`Z` = 4.42):

| Claim | Test |
|---|---|
| Continuous distribution (normal, lognormal, mixture, uniform, exponential, Pareto) | one-sample KS against the declared CDF, plus bounds and moments |
| Discrete distribution (Poisson, binomial, negative binomial, discrete uniform, categorical, lengths, fan-out) | chi-square on binned counts (bins pooled to an expected count of at least 5), plus bounds and moments |
| Zipf (column values and fan-out) | top-three rank shares, plus a chi-square test of the occupancy profile (how many categories hold 0, 1, 2, ... draws) |
| Seasonal timestamps | chi-square on the hour-of-day and day-of-week profiles |
| Pearson, Spearman | Fisher z; Spearman with the 1.06 / (n - 3) variance correction |
| Cramér's V, correlation ratio | seeded bootstrap standard error (200 replicates) |
| Null rates, format, category and special-value shares, quasi-identifier unique share | binomial standard error |

Negative tests tamper with a manifest or the data and check that verification reports it. They include wrong
families with matched mean and variance (uniform, mixture, and lognormal declared as normal; a discrete uniform
declared negative binomial; a Poisson declared binomial; a wrong Zipf exponent) and misdeclared strengths for every
dependency measure and the quasi-identifier set.

Declared values are the generating parameters, not statistics of the sample. A profiler is scored on how well it
recovers what was planted, and the sample's own noise is part of what it sees. `realized(manifest, frames)` gives the
sample statistics for metrics that need them.

## Real datasets

`python -m profiler_bench.catalog real` exports four real datasets (27 tables) from a Databricks SQL warehouse
into the cache, at tiers `full` (the export) and `s` (a deterministic primary-key-hash subset of at most 100,000
rows per table, closed under the declared foreign keys). Their manifests are `ground_truth: none`: they claim the
schema, row counts, keys, foreign keys, and privacy classes, and nothing about distributions. Data is never
committed.

| Dataset | Licence | What it adds over the synthetic catalog |
|---|---|---|
| `tpcds_sf1` | internal use, no stated licence; TPC caveat: not publishable as TPC-DS results | Star schema with real fan-out skew (store_sales rows per item p99 347, per customer p99 79), composite keys, `decimal(7,2)` measures, about 4.5% null foreign keys in `store_sales`, customer quasi-identifiers (birth date and country, address, zip) |
| `wanderbricks` | internal use, no stated licence | 16 tables with `struct` and `array<struct>` columns, emails, phones, and free text, skewed hosts-to-properties fan-out (p99 34, max 119), duplicated "id" columns, and orphaned child keys |
| `nyctaxi_trips` | internal use, no stated licence | Heavy-tailed trip distances and fares, negative fares, zip quasi-identifiers |
| `simulated_retail` | CC BY 4.0, Marketplace consumer terms; attribution "Simulated Retail Customer Data" by Databricks | Change-data-capture customer versions (`customer_id` repeats; the key is `(customer_id, valid_from)`), JSON held in strings (`ordered_products`, `clicked_items`, `promo_info`, `product`), tax ids and addresses |

Privacy classes: `direct_identifier` for names, emails, phones, logins, tax ids, full addresses, and record ids
(`c_customer_id`, `ca_address_id`); `quasi_identifier` for birth dates and countries, street parts, cities, zips,
coordinates, `company_name`, and `users.country`; `free_text` for `reviews.comment`, `properties.description`, and
`customer_support_logs.messages[].message`.

Documented keys that do **not** hold, so they are not declared (measured on the full sources):

| Table | Column | Finding |
|---|---|---|
| `wanderbricks.page_views` | `view_id` | 5,000 distinct values over 500,000 rows |
| `wanderbricks.reviews` | `review_id` | 1,000 distinct over 99,793 rows |
| `wanderbricks.payments` | `payment_id` | 45,173 distinct over 49,638 rows; 8 `booking_id` values have no booking |
| `wanderbricks.booking_updates` | `booking_update_id` | 75,125 distinct over 83,068 rows; 47 `booking_id` values have no booking |
| `simulated_retail.customers` | `customer_id` | 28,670 distinct over 28,813 rows (CDC versions), so `sales.customer_id` and `sales_orders.customer_id` reference no key |
| `simulated_retail.sales_orders` | `order_number` | 4,000 distinct over 4,074 rows |

Each real dataset has one sample, so reconstruction and privacy have no independent holdout or real-vs-real ceiling,
and structure scoring has no planted truth to compare against. Wiring real datasets into the evaluator (skip
structure when `ground_truth` is `none`; split a holdout) is not done yet.

## Not covered yet

- Real datasets in the evaluator: structure scoring does not yet skip `ground_truth: none`, and privacy has no
  holdout for a real dataset.
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
- Skew at the partition level: files of uneven size, many small files, and partitioned (Hive-style) layouts.
- Duplicated rows and near-duplicates, and tables wider than 200 columns or deeper than two nesting levels.
- Privacy ground truth beyond the one quasi-identifier set: rare categories that make a row unique, quasi-identifiers
  across tables, and sensitive values embedded in free text.
- Goodness of fit for the copula itself (only its pairwise Spearman values are tested) and for timestamps within an
  hour of the seasonal profile.

These gaps are explicit; they are not counted as covered structures.
