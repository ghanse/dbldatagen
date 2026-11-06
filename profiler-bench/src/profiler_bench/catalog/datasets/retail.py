"""Retail relational model: customers, products, orders, and order items."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from profiler_bench.catalog import gen
from profiler_bench.catalog.gen import DIGITS, UPPER, Rng
from profiler_bench.catalog.manifest import (
    Column,
    ComparisonCheck,
    Distribution,
    EnumerationCheck,
    ExpressionCheck,
    Family,
    ForeignKey,
    Format,
    FunctionalDependency,
    RangeCheck,
    Table,
)
from profiler_bench.catalog.registry import Build, Dataset


CATEGORIES = ["apparel", "books", "electronics", "garden", "grocery", "kitchen", "sports", "toys"]
CATEGORY_PROBABILITIES = [0.2, 0.1, 0.15, 0.05, 0.2, 0.1, 0.1, 0.1]
DEPARTMENTS = {
    "apparel": "fashion",
    "books": "media",
    "electronics": "media",
    "garden": "home",
    "grocery": "food",
    "kitchen": "home",
    "sports": "outdoor",
    "toys": "outdoor",
}
SEGMENTS = ["consumer", "small_business", "enterprise"]
SEGMENT_PROBABILITIES = [0.7, 0.22, 0.08]
COUNTRIES = ["US", "CA", "GB", "DE", "FR", "JP", "BR"]
COUNTRY_PROBABILITIES = [0.5, 0.1, 0.1, 0.1, 0.08, 0.07, 0.05]
REGIONS = {"US": "AMER", "CA": "AMER", "BR": "AMER", "GB": "EMEA", "DE": "EMEA", "FR": "EMEA", "JP": "APAC"}
DOMAINS = ["example.com", "mail.test", "corp.example"]
STATUSES = ["delivered", "shipped", "pending", "cancelled", "returned"]
STATUS_PROBABILITIES = [0.55, 0.2, 0.1, 0.05, 0.1]
UNSHIPPED = ("pending", "cancelled")
SKU = [UPPER] * 3 + ["-"] + [DIGITS] * 4
PRICE_MU, PRICE_SIGMA = 3.0, 0.8
CUSTOMER_ZIPF_S = 1.1
PRODUCT_ZIPF_S = 1.3
ITEMS_LAM = 3.0
QUANTITY_LAM = 1.0
SHIP_LAG_LAM = 2.0
SIGNUP = ("2018-01-01", "2024-12-31")
ORDERED = ("2023-01-01", "2024-12-31")


def _categorical(values: list[str], probabilities: list[float]) -> Distribution:
    return Distribution(family=Family.CATEGORICAL, params={"values": values, "probabilities": probabilities})


def _uniform_dates(rng: np.random.Generator, n: int, bounds: tuple[str, str]) -> np.ndarray:
    return rng.integers(gen.days(bounds[0]), gen.days(bounds[1]) + 1, n)


def _counts(n: int) -> tuple[int, int, int]:
    return max(100, n // 200), max(200, n // 20), n // 4


def _extra_items(n_orders: int) -> int:
    return round(ITEMS_LAM * n_orders)


def sizes(n: int) -> dict[str, float]:
    """Returns the expected row count of each retail table for base row count `n`."""
    n_products, n_customers, n_orders = _counts(n)
    return {
        "customers": n_customers,
        "products": n_products,
        "orders": n_orders,
        "order_items": n_orders + _extra_items(n_orders),
    }


def build(n: int, rng: Rng, structure: Rng) -> Build:
    """Generates the retail tables for base row count `n` (about `n` order items), each in shuffled row order."""
    n_products, n_customers, n_orders = _counts(n)

    category = gen.choice(rng("products.category"), n_products, CATEGORY_PROBABILITIES)
    price_cents = np.maximum(
        1, np.round(np.exp(rng("products.unit_price").normal(PRICE_MU, PRICE_SIGMA, n_products)) * 100)
    ).astype(np.int64)
    products = pa.table(
        {
            "product_id": pa.array(np.arange(1, n_products + 1), pa.int64()),
            "sku": gen.indexed_strings(rng("products.sku").choice(26**3 * 10**4, n_products, replace=False), SKU),
            "category": gen.take(CATEGORIES, category),
            "department": gen.take([DEPARTMENTS[c] for c in CATEGORIES], category),
            "unit_price": gen.cents_to_decimal(price_cents, 10),
        }
    )

    country = gen.choice(rng("customers.country"), n_customers, COUNTRY_PROBABILITIES)
    customer_number = np.arange(1, n_customers + 1)
    customer_id = gen.hashed_ids(customer_number, int(rng("customers.customer_id").integers(0, 2**62)))
    names = gen.words(structure("customers.names"), 400)
    customers = pa.table(
        {
            "customer_id": pa.array(customer_id, pa.int64()),
            "email": gen.join(
                gen.take(names, rng("customers.first").integers(0, 200, n_customers)),
                ".",
                gen.take(names, rng("customers.last").integers(200, 400, n_customers)),
                pa.array(customer_number.astype(str)),
                "@",
                gen.take(DOMAINS, rng("customers.domain").integers(0, len(DOMAINS), n_customers)),
            ),
            "segment": gen.take(SEGMENTS, gen.choice(rng("customers.segment"), n_customers, SEGMENT_PROBABILITIES)),
            "country": gen.take(COUNTRIES, country),
            "region": gen.take([REGIONS[c] for c in COUNTRIES], country),
            "signup_date": gen.dates(_uniform_dates(rng("customers.signup_date"), n_customers, SIGNUP)),
        }
    )

    status = gen.choice(rng("orders.status"), n_orders, STATUS_PROBABILITIES)
    order_date = _uniform_dates(rng("orders.order_date"), n_orders, ORDERED)
    unshipped = np.isin(status, [STATUSES.index(s) for s in UNSHIPPED])
    ship_date = order_date + rng("orders.ship_date").poisson(SHIP_LAG_LAM, n_orders)
    orders = pa.table(
        {
            "order_id": pa.array(np.arange(1, n_orders + 1), pa.int64()),
            "customer_id": pa.array(
                customer_id[gen.zipf_choice(rng("orders.customer_id"), n_orders, n_customers, CUSTOMER_ZIPF_S)],
                pa.int64(),
            ),
            "order_date": gen.dates(order_date),
            "ship_date": gen.dates(ship_date, mask=unshipped),
            "status": gen.take(STATUSES, status),
        }
    )

    extra = _extra_items(n_orders)
    items_per_order = 1 + rng("order_items.count").multinomial(extra, np.full(n_orders, 1 / n_orders))
    n_items = int(items_per_order.sum())
    starts = np.repeat(np.cumsum(items_per_order) - items_per_order, items_per_order)
    product = gen.zipf_choice(rng("order_items.product_id"), n_items, n_products, PRODUCT_ZIPF_S)
    quantity = 1 + rng("order_items.quantity").poisson(QUANTITY_LAM, n_items)
    order_items = pa.table(
        {
            "order_id": pa.array(np.repeat(np.arange(1, n_orders + 1), items_per_order), pa.int64()),
            "line_no": pa.array(np.arange(n_items) - starts + 1, pa.int32()),
            "product_id": pa.array(product + 1, pa.int64()),
            "quantity": pa.array(quantity, pa.int32()),
            "unit_price": gen.cents_to_decimal(price_cents[product], 10),
            "line_total": gen.cents_to_decimal(price_cents[product] * quantity, 12),
        }
    )

    ordered = {"customers": customers, "products": products, "orders": orders, "order_items": order_items}
    tables = {name: gen.shuffle(table, rng(f"{name}.row_order")) for name, table in ordered.items()}
    return Build(tables, _truth(n_products, n_customers, n_orders), _foreign_keys(n_customers, n_products, n_orders))


def _truth(n_products: int, n_customers: int, n_orders: int) -> list[Table]:
    sequence = Distribution(family=Family.SEQUENCE, params={"start": 1, "step": 1})
    unshipped_rate = sum(STATUS_PROBABILITIES[STATUSES.index(s)] for s in UNSHIPPED)
    unshipped_sql = ", ".join(f"'{s}'" for s in UNSHIPPED)
    customers = Table(
        name="customers",
        row_order="shuffled",
        description="Customers with a unique email, a country, and the region the country belongs to.",
        primary_key=("customer_id",),
        unique_keys=(("email",),),
        columns=(
            Column(
                name="customer_id",
                unique=True,
                distinct=n_customers,
                note="Hashed: a keyed 63-bit bijection of 1..n, so ids are distinct, non-negative, and unordered.",
            ),
            Column(
                name="email",
                unique=True,
                privacy="direct_identifier",
                formats=(Format(regex=r"[a-z]+\.[a-z]+[0-9]+@(example\.com|mail\.test|corp\.example)", share=1.0),),
                note="first.last plus the customer number, so it is unique and must not leak.",
            ),
            Column(name="segment", distinct=len(SEGMENTS), distribution=_categorical(SEGMENTS, SEGMENT_PROBABILITIES)),
            Column(
                name="country", max_distinct=len(COUNTRIES), distribution=_categorical(COUNTRIES, COUNTRY_PROBABILITIES)
            ),
            Column(name="region", max_distinct=len(set(REGIONS.values()))),
            Column(
                name="signup_date",
                distribution=Distribution(family=Family.UNIFORM, params={"low": SIGNUP[0], "high": SIGNUP[1]}),
            ),
        ),
        checks=(
            RangeCheck(column="customer_id", min=0),
            EnumerationCheck(column="segment", values=tuple(SEGMENTS)),
            EnumerationCheck(column="country", values=tuple(COUNTRIES)),
            RangeCheck(column="signup_date", min=SIGNUP[0], max=SIGNUP[1]),
        ),
        functional_dependencies=(FunctionalDependency(determinant=("country",), dependent=("region",)),),
    )
    products = Table(
        name="products",
        row_order="shuffled",
        description="Product catalog; every category belongs to one department.",
        primary_key=("product_id",),
        unique_keys=(("sku",),),
        columns=(
            Column(name="product_id", unique=True, distinct=n_products, distribution=sequence),
            Column(name="sku", unique=True, formats=(Format(regex=r"[A-Z]{3}-[0-9]{4}", share=1.0),)),
            Column(
                name="category",
                max_distinct=len(CATEGORIES),
                distribution=_categorical(CATEGORIES, CATEGORY_PROBABILITIES),
            ),
            Column(name="department", max_distinct=len(set(DEPARTMENTS.values()))),
            Column(
                name="unit_price",
                distribution=Distribution(family=Family.LOGNORMAL, params={"mu": PRICE_MU, "sigma": PRICE_SIGMA}),
                note="Rounded to cents, at least 0.01.",
            ),
        ),
        checks=(
            EnumerationCheck(column="category", values=tuple(CATEGORIES)),
            RangeCheck(column="unit_price", min="0.01"),
        ),
        functional_dependencies=(FunctionalDependency(determinant=("category",), dependent=("department",)),),
    )
    orders = Table(
        name="orders",
        row_order="shuffled",
        description="Orders; customers are chosen by a Zipf law, and only shipped orders have a ship date.",
        primary_key=("order_id",),
        columns=(
            Column(name="order_id", unique=True, distinct=n_orders, distribution=sequence),
            Column(name="customer_id", max_distinct=n_customers),
            Column(
                name="order_date",
                distribution=Distribution(family=Family.UNIFORM, params={"low": ORDERED[0], "high": ORDERED[1]}),
            ),
            Column(
                name="ship_date",
                null_rate=unshipped_rate,
                note=f"order_date plus Poisson({SHIP_LAG_LAM}) days; null exactly when status is {list(UNSHIPPED)}.",
            ),
            Column(
                name="status", max_distinct=len(STATUSES), distribution=_categorical(STATUSES, STATUS_PROBABILITIES)
            ),
        ),
        checks=(
            EnumerationCheck(column="status", values=tuple(STATUSES)),
            ComparisonCheck(left="order_date", op="<=", right="ship_date"),
            ExpressionCheck(
                sql=f"(ship_date IS NULL) = (status IN ({unshipped_sql}))", columns=("ship_date", "status")
            ),
        ),
    )
    order_items = Table(
        name="order_items",
        row_order="shuffled",
        description="Order lines; products are chosen by a Zipf law and the line total is price times quantity.",
        primary_key=("order_id", "line_no"),
        columns=(
            Column(name="order_id", max_distinct=n_orders),
            Column(name="line_no", note="1..k within each order."),
            Column(name="product_id", max_distinct=n_products),
            Column(
                name="quantity",
                distribution=Distribution(family=Family.POISSON, params={"lam": QUANTITY_LAM, "shift": 1}),
            ),
            Column(name="unit_price", note="Copied from products.unit_price."),
            Column(name="line_total", note="unit_price * quantity, exact."),
        ),
        checks=(
            RangeCheck(column="quantity", min=1),
            RangeCheck(column="line_no", min=1),
            ExpressionCheck(sql="line_total = unit_price * quantity", columns=("line_total", "unit_price", "quantity")),
        ),
        functional_dependencies=(FunctionalDependency(determinant=("product_id",), dependent=("unit_price",)),),
    )
    return [customers, products, orders, order_items]


def _foreign_keys(n_customers: int, n_products: int, n_orders: int) -> list[ForeignKey]:
    return [
        ForeignKey(
            child_table="orders",
            child_columns=("customer_id",),
            parent_table="customers",
            parent_columns=("customer_id",),
            fanout=Distribution(family=Family.ZIPF, params={"s": CUSTOMER_ZIPF_S, "n_values": n_customers}),
        ),
        ForeignKey(
            child_table="order_items",
            child_columns=("order_id",),
            parent_table="orders",
            parent_columns=("order_id",),
            fanout=Distribution(
                family=Family.BINOMIAL,
                params={"n": _extra_items(n_orders), "p": 1 / n_orders, "shift": 1},
            ),
        ),
        ForeignKey(
            child_table="order_items",
            child_columns=("product_id",),
            parent_table="products",
            parent_columns=("product_id",),
            fanout=Distribution(family=Family.ZIPF, params={"s": PRODUCT_ZIPF_S, "n_values": n_products}),
        ),
    ]


DATASET = Dataset(
    name="retail",
    description="Relational retail model with PK/FKs, Zipf fan-out, date ordering, a status enum, and line totals.",
    build=build,
    sizes=sizes,
)
