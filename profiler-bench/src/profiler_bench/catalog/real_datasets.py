"""Real datasets in the catalog, with their sources, licences, keys, privacy classes, and pinned hashes.

Row counts, keys, and foreign keys were measured on the sources in November 2026. A documented key or foreign
key is declared only when it holds in the data; the ones that do not are listed in COVERAGE.md.
"""

from __future__ import annotations

from profiler_bench.catalog.manifest import ForeignKey
from profiler_bench.catalog.real import RealDataset, RealTable


SAMPLES_LICENSE = "internal use, no stated licence (Databricks `samples` catalog)"
SAMPLES_TERMS = "Bundled with Databricks workspaces for demos and testing; no licence is stated, so use it internally."
RETAIL_CATALOG = "simulated_retail_customer_data"


def _fk(child: str, columns: str, parent: str, parent_columns: str) -> ForeignKey:
    return ForeignKey(
        child_table=child,
        child_columns=tuple(columns.split(",")),
        parent_table=parent,
        parent_columns=tuple(parent_columns.split(",")),
    )


TPCDS = RealDataset(
    name="tpcds_sf1",
    description=(
        "TPC-DS scale factor 1 from samples.tpcds_sf1: a retail star schema with real fan-out skew, decimals, "
        "nullable foreign keys, and customer quasi-identifiers."
    ),
    license=SAMPLES_LICENSE,
    terms=(
        f"{SAMPLES_TERMS} The data comes from the TPC-DS generator; results measured on it are not TPC-DS "
        "benchmark results and must not be published as such (TPC fair-use policy)."
    ),
    tables=(
        RealTable(
            name="customer",
            source="samples.tpcds_sf1.customer",
            rows=100_000,
            columns=(
                "c_customer_sk int, c_customer_id string, c_current_cdemo_sk int, c_current_hdemo_sk int, "
                "c_current_addr_sk int, c_first_shipto_date_sk int, c_first_sales_date_sk int, c_salutation string, "
                "c_first_name string, c_last_name string, c_preferred_cust_flag string, c_birth_day int, "
                "c_birth_month int, c_birth_year int, c_birth_country string, c_login string, "
                "c_email_address string, c_last_review_date_sk int, c_last_review_date int"
            ),
            primary_key=("c_customer_sk",),
            privacy={
                "c_customer_id": "direct_identifier",
                "c_first_name": "direct_identifier",
                "c_last_name": "direct_identifier",
                "c_login": "direct_identifier",
                "c_email_address": "direct_identifier",
                "c_birth_day": "quasi_identifier",
                "c_birth_month": "quasi_identifier",
                "c_birth_year": "quasi_identifier",
                "c_birth_country": "quasi_identifier",
            },
        ),
        RealTable(
            name="customer_address",
            source="samples.tpcds_sf1.customer_address",
            rows=50_000,
            columns=(
                "ca_address_sk int, ca_address_id string, ca_street_number string, ca_street_name string, "
                "ca_street_type string, ca_suite_number string, ca_city string, ca_county string, ca_state string, "
                "ca_zip string, ca_country string, ca_gmt_offset decimal(5,2), ca_location_type string"
            ),
            primary_key=("ca_address_sk",),
            privacy={
                "ca_address_id": "direct_identifier",
                "ca_street_number": "quasi_identifier",
                "ca_street_name": "quasi_identifier",
                "ca_suite_number": "quasi_identifier",
                "ca_city": "quasi_identifier",
                "ca_zip": "quasi_identifier",
            },
        ),
        RealTable(
            name="date_dim",
            source="samples.tpcds_sf1.date_dim",
            rows=73_049,
            columns=(
                "d_date_sk int, d_date_id string, d_date date, d_month_seq int, d_week_seq int, d_quarter_seq int, "
                "d_year int, d_dow int, d_moy int, d_dom int, d_qoy int, d_fy_year int, d_fy_quarter_seq int, "
                "d_fy_week_seq int, d_day_name string, d_quarter_name string, d_holiday string, d_weekend string, "
                "d_following_holiday string, d_first_dom int, d_last_dom int, d_same_day_ly int, "
                "d_same_day_lq int, d_current_day string, d_current_week string, d_current_month string, "
                "d_current_quarter string, d_current_year string"
            ),
            primary_key=("d_date_sk",),
        ),
        RealTable(
            name="item",
            source="samples.tpcds_sf1.item",
            rows=18_000,
            columns=(
                "i_item_sk int, i_item_id string, i_rec_start_date date, i_rec_end_date date, i_item_desc string, "
                "i_current_price decimal(7,2), i_wholesale_cost decimal(7,2), i_brand_id int, i_brand string, "
                "i_class_id int, i_class string, i_category_id int, i_category string, i_manufact_id int, "
                "i_manufact string, i_size string, i_formulation string, i_color string, i_units string, "
                "i_container string, i_manager_id int, i_product_name string"
            ),
            primary_key=("i_item_sk",),
        ),
        RealTable(
            name="store_sales",
            source="samples.tpcds_sf1.store_sales",
            rows=2_879_789,
            columns=(
                "ss_sold_date_sk int, ss_sold_time_sk int, ss_item_sk int, ss_customer_sk int, ss_cdemo_sk int, "
                "ss_hdemo_sk int, ss_addr_sk int, ss_store_sk int, ss_promo_sk int, ss_ticket_number bigint, "
                "ss_quantity int, ss_wholesale_cost decimal(7,2), ss_list_price decimal(7,2), "
                "ss_sales_price decimal(7,2), ss_ext_discount_amt decimal(7,2), ss_ext_sales_price decimal(7,2), "
                "ss_ext_wholesale_cost decimal(7,2), ss_ext_list_price decimal(7,2), ss_ext_tax decimal(7,2), "
                "ss_coupon_amt decimal(7,2), ss_net_paid decimal(7,2), ss_net_paid_inc_tax decimal(7,2), "
                "ss_net_profit decimal(7,2)"
            ),
            primary_key=("ss_item_sk", "ss_ticket_number"),
        ),
        RealTable(
            name="web_sales",
            source="samples.tpcds_sf1.web_sales",
            rows=718_931,
            columns=(
                "ws_sold_date_sk int, ws_sold_time_sk int, ws_ship_date_sk int, ws_item_sk int, "
                "ws_bill_customer_sk int, ws_bill_cdemo_sk int, ws_bill_hdemo_sk int, ws_bill_addr_sk int, "
                "ws_ship_customer_sk int, ws_ship_cdemo_sk int, ws_ship_hdemo_sk int, ws_ship_addr_sk int, "
                "ws_web_page_sk int, ws_web_site_sk int, ws_ship_mode_sk int, ws_warehouse_sk int, ws_promo_sk int, "
                "ws_order_number bigint, ws_quantity int, ws_wholesale_cost decimal(7,2), "
                "ws_list_price decimal(7,2), ws_sales_price decimal(7,2), ws_ext_discount_amt decimal(7,2), "
                "ws_ext_sales_price decimal(7,2), ws_ext_wholesale_cost decimal(7,2), "
                "ws_ext_list_price decimal(7,2), ws_ext_tax decimal(7,2), ws_coupon_amt decimal(7,2), "
                "ws_ext_ship_cost decimal(7,2), ws_net_paid decimal(7,2), ws_net_paid_inc_tax decimal(7,2), "
                "ws_net_paid_inc_ship decimal(7,2), ws_net_paid_inc_ship_tax decimal(7,2), ws_net_profit decimal(7,2)"
            ),
            primary_key=("ws_item_sk", "ws_order_number"),
        ),
    ),
    foreign_keys=(
        _fk("customer", "c_current_addr_sk", "customer_address", "ca_address_sk"),
        _fk("customer", "c_first_shipto_date_sk", "date_dim", "d_date_sk"),
        _fk("customer", "c_first_sales_date_sk", "date_dim", "d_date_sk"),
        _fk("customer", "c_last_review_date_sk", "date_dim", "d_date_sk"),
        _fk("store_sales", "ss_customer_sk", "customer", "c_customer_sk"),
        _fk("store_sales", "ss_item_sk", "item", "i_item_sk"),
        _fk("store_sales", "ss_sold_date_sk", "date_dim", "d_date_sk"),
        _fk("store_sales", "ss_addr_sk", "customer_address", "ca_address_sk"),
        _fk("web_sales", "ws_bill_customer_sk", "customer", "c_customer_sk"),
        _fk("web_sales", "ws_ship_customer_sk", "customer", "c_customer_sk"),
        _fk("web_sales", "ws_item_sk", "item", "i_item_sk"),
        _fk("web_sales", "ws_sold_date_sk", "date_dim", "d_date_sk"),
        _fk("web_sales", "ws_ship_date_sk", "date_dim", "d_date_sk"),
        _fk("web_sales", "ws_bill_addr_sk", "customer_address", "ca_address_sk"),
        _fk("web_sales", "ws_ship_addr_sk", "customer_address", "ca_address_sk"),
    ),
    hashes={
        "customer": "e0db6def9be90f3f6f456e70d4e405fe6bb3e48113b8a60bc2805aba600c7db9",
        "customer_address": "3ee273674ebad03077e1094899847b7eaf9d244d9e2a5023e140c4f3808de6bd",
        "date_dim": "d3b9ea922fdfc4c35749dc25c0a1f15d5b0e4adaa520688594ae595592baca74",
        "item": "e9604a05a9666fc824ae5d70a16452d0fe67d25a2f7b5d57e81f246f4d78b63a",
        "store_sales": "b5f72b617e325ffeb6a80eac1ab521f8feb6273268acd9166194dae4167c81cc",
        "web_sales": "7a32b4c9558c27ae68d0dc6aecd4b4f1c9674d92946dcda6f37953cf5d92ab18",
    },
    split_roots=("customer",),
)

_W = "samples.wanderbricks"

WANDERBRICKS = RealDataset(
    name="wanderbricks",
    description=(
        "Wanderbricks travel bookings from samples.wanderbricks: 16 tables with struct and array columns, "
        "emails and phone numbers, free text, skewed hosts-to-properties fan-out, and duplicated event ids."
    ),
    license=SAMPLES_LICENSE,
    terms=SAMPLES_TERMS,
    tables=(
        RealTable(
            "amenities",
            f"{_W}.amenities",
            38,
            "amenity_id bigint, name string, category string, icon string",
            ("amenity_id",),
        ),
        RealTable(
            "booking_updates",
            f"{_W}.booking_updates",
            83_068,
            "booking_id bigint, booking_update_id bigint, check_in date, check_out date, created_at timestamp, "
            "guests_count int, property_id bigint, status string, total_amount float, updated_at timestamp, "
            "user_id bigint",
        ),
        RealTable(
            "bookings",
            f"{_W}.bookings",
            72_247,
            "booking_id bigint, user_id bigint, property_id bigint, check_in date, check_out date, guests_count int, "
            "total_amount float, status string, created_at timestamp, updated_at timestamp",
            ("booking_id",),
        ),
        RealTable(
            "clickstream",
            f"{_W}.clickstream",
            100_000,
            "event string, metadata struct<device:string,referrer:string>, property_id bigint, timestamp timestamp, "
            "user_id bigint",
        ),
        RealTable(
            "countries",
            f"{_W}.countries",
            168,
            "country string, country_code string, continent string",
            ("country_code",),
            unique_keys=(("country",),),
        ),
        RealTable(
            "customer_support_logs",
            f"{_W}.customer_support_logs",
            1_900,
            "created_at string, messages array<struct<message:string,sender:string,sentiment:string,timestamp:string>>, "
            "support_agent_id string, ticket_id string, user_id bigint",
            ("ticket_id",),
            privacy={"messages.element.message": "free_text"},
        ),
        RealTable(
            "destinations",
            f"{_W}.destinations",
            42,
            "destination_id bigint, destination string, country string, state_or_province string, "
            "state_or_province_code string, description string",
            ("destination_id",),
        ),
        RealTable(
            "employees",
            f"{_W}.employees",
            73_006,
            "employee_id bigint, host_id bigint, name string, role string, email string, phone string, country string, "
            "joined_at date, end_service_date date, is_currently_employed boolean",
            ("employee_id",),
            privacy={"name": "direct_identifier", "email": "direct_identifier", "phone": "direct_identifier"},
        ),
        RealTable(
            "hosts",
            f"{_W}.hosts",
            19_384,
            "host_id bigint, name string, email string, phone string, is_verified boolean, is_active boolean, "
            "rating float, country string, joined_at date",
            ("host_id",),
            privacy={"name": "direct_identifier", "email": "direct_identifier", "phone": "direct_identifier"},
        ),
        RealTable(
            "page_views",
            f"{_W}.page_views",
            500_000,
            "device_type string, page_url string, property_id bigint, referrer string, timestamp timestamp, "
            "user_id bigint, view_id bigint",
        ),
        RealTable(
            "payments",
            f"{_W}.payments",
            49_638,
            "payment_id bigint, booking_id bigint, amount decimal(15,4), payment_method string, status string, "
            "payment_date timestamp",
        ),
        RealTable(
            "properties",
            f"{_W}.properties",
            18_163,
            "property_id bigint, host_id bigint, destination_id bigint, title string, description string, "
            "base_price float, property_type string, max_guests int, bedrooms int, bathrooms int, "
            "property_latitude float, property_longitude float, created_at date",
            ("property_id",),
            privacy={"description": "free_text"},
        ),
        RealTable(
            "property_amenities",
            f"{_W}.property_amenities",
            118_108,
            "property_id bigint, amenity_id bigint",
            ("property_id", "amenity_id"),
        ),
        RealTable(
            "property_images",
            f"{_W}.property_images",
            54_186,
            "image_id bigint, property_id bigint, url string, sequence int, is_primary boolean, uploaded_at date",
            ("image_id",),
        ),
        RealTable(
            "reviews",
            f"{_W}.reviews",
            99_793,
            "booking_id bigint, comment string, created_at timestamp, is_deleted boolean, property_id bigint, "
            "rating double, review_id bigint, updated_at timestamp, user_id bigint",
            privacy={"comment": "free_text"},
        ),
        RealTable(
            "users",
            f"{_W}.users",
            124_509,
            "user_id bigint, email string, name string, country string, user_type string, created_at timestamp, "
            "is_business boolean, company_name string",
            ("user_id",),
            privacy={
                "email": "direct_identifier",
                "name": "direct_identifier",
                "company_name": "quasi_identifier",
                "country": "quasi_identifier",
            },
        ),
    ),
    foreign_keys=(
        _fk("bookings", "user_id", "users", "user_id"),
        _fk("bookings", "property_id", "properties", "property_id"),
        _fk("booking_updates", "user_id", "users", "user_id"),
        _fk("booking_updates", "property_id", "properties", "property_id"),
        _fk("clickstream", "user_id", "users", "user_id"),
        _fk("clickstream", "property_id", "properties", "property_id"),
        _fk("customer_support_logs", "user_id", "users", "user_id"),
        _fk("employees", "host_id", "hosts", "host_id"),
        _fk("page_views", "user_id", "users", "user_id"),
        _fk("page_views", "property_id", "properties", "property_id"),
        _fk("properties", "host_id", "hosts", "host_id"),
        _fk("properties", "destination_id", "destinations", "destination_id"),
        _fk("property_amenities", "property_id", "properties", "property_id"),
        _fk("property_amenities", "amenity_id", "amenities", "amenity_id"),
        _fk("property_images", "property_id", "properties", "property_id"),
        _fk("reviews", "booking_id", "bookings", "booking_id"),
        _fk("reviews", "property_id", "properties", "property_id"),
        _fk("reviews", "user_id", "users", "user_id"),
        _fk("users", "country", "countries", "country"),
        _fk("hosts", "country", "countries", "country"),
        _fk("destinations", "country", "countries", "country"),
    ),
    hashes={
        "amenities": "6bae9946c624582e5ebd23474be0bc3f7ff9cf3a05c5d4046358d6fdb54ec787",
        "booking_updates": "8dc95f1798bbb6a72a0571250361a5a53ae538b9e49165af335ac0d3b63ce8a9",
        "bookings": "a5f8ffd000da8bb49cf4ff731c9358a93fd7c6957627d7ab321b5adc64da55f0",
        "clickstream": "ba9e029a8aec285a4223435bc21670235e3200fbf9e5d5f7c4664dd7dc43cf4d",
        "countries": "40a8a5e26a7dc789217792f6e54a4ccbbe5d366534aef485dbbdcc838738953d",
        "customer_support_logs": "9b69ececd7117b422d1c5d99c583ef64e5c8e386d8a1586ad342b87582480df9",
        "destinations": "e1caa9e01c382f105a61b2a0e2062d3ae38a4bc6b1da8c853ca479ee46d7ed36",
        "employees": "92857a1ac4c1d0c66b27c89eb5fa4f90d0f328122a77227dfda54fdbdabc6ac9",
        "hosts": "79b65dc2e68b8a97ecedbb08b38ef95463be4d6f6d47e0f2cc53bbf3b1273870",
        "page_views": "05a8e7f04f14ef8a721946bdf3e2fc1b44127b3ee7c1655e2501ec09f2745949",
        "payments": "be2fcf38edc5c59b265f6f05aacbf32e6294ecd28d684b262d5f96a37fe8e30b",
        "properties": "f0df10e5b84d3969afae22ee9757d9452d6b17bdcb889aa4f601a305d5249395",
        "property_amenities": "91610f9a4c09d0ec84935926b3661910b0403f5db2154a7fefdb7b1f80a90f17",
        "property_images": "9433a6d28179f226697ad36433318c7d5e0a778a2025cc943369cc963cfe4b69",
        "reviews": "0e361be7797d7b09b3e38e96c677e669ecd5b2396145226e7372fbd21731be36",
        "users": "25081d5491fb4f3c04ca68015acbf46a1c7a4a87847eeb5c74ba6338fe708492",
    },
    split_roots=("users", "hosts"),
)

NYCTAXI = RealDataset(
    name="nyctaxi_trips",
    description="NYC yellow-cab trips from samples.nyctaxi.trips: heavy-tailed distances and fares, negative fares.",
    license=SAMPLES_LICENSE,
    terms=SAMPLES_TERMS,
    tables=(
        RealTable(
            "trips",
            "samples.nyctaxi.trips",
            21_932,
            "tpep_pickup_datetime timestamp, tpep_dropoff_datetime timestamp, trip_distance double, "
            "fare_amount double, pickup_zip int, dropoff_zip int",
            privacy={"pickup_zip": "quasi_identifier", "dropoff_zip": "quasi_identifier"},
        ),
    ),
    hashes={
        "trips": "1d45b0d901ac12d86fe0785e636cfc249a941b87bb833ff0d9e6366c1631283b",
    },
)

_R = f"{RETAIL_CATALOG}.v01"

SIMULATED_RETAIL = RealDataset(
    name="simulated_retail",
    description=(
        "Simulated Retail Customer Data (Databricks Marketplace, provider Databricks), schema v01: customers with "
        "change-data-capture versions, sales, and orders holding JSON strings."
    ),
    license="CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
    terms=(
        "Databricks Marketplace consumer terms "
        "(https://cms.databricks.com/sites/default/files/2023-01/marketplace-consumer-terms_0.pdf). Attribution: "
        "'Simulated Retail Customer Data' by Databricks, Databricks Marketplace, CC BY 4.0. Install the listing as "
        f"catalog {RETAIL_CATALOG} in the workspace you export from."
    ),
    tables=(
        RealTable(
            "customers",
            f"{_R}.customers",
            28_813,
            "customer_id bigint, tax_id string, tax_code string, customer_name string, state string, city string, "
            "postcode string, street string, number string, unit string, region string, district string, lon double, "
            "lat double, ship_to_address string, valid_from bigint, valid_to string, units_purchased bigint, "
            "loyalty_segment bigint",
            ("customer_id", "valid_from"),
            privacy={
                "tax_id": "direct_identifier",
                "customer_name": "direct_identifier",
                "ship_to_address": "direct_identifier",
                "street": "quasi_identifier",
                "number": "quasi_identifier",
                "unit": "quasi_identifier",
                "postcode": "quasi_identifier",
                "city": "quasi_identifier",
                "lon": "quasi_identifier",
                "lat": "quasi_identifier",
            },
        ),
        RealTable(
            "sales",
            f"{_R}.sales",
            360,
            "customer_id bigint, customer_name string, product_name string, order_date date, product_category string, "
            "product string, total_price bigint",
            privacy={"customer_name": "direct_identifier"},
        ),
        RealTable(
            "sales_orders",
            f"{_R}.sales_orders",
            4_074,
            "clicked_items string, customer_id bigint, customer_name string, number_of_line_items bigint, "
            "order_datetime bigint, order_number bigint, ordered_products string, promo_info string",
            privacy={"customer_name": "direct_identifier"},
        ),
    ),
    hashes={
        "customers": "fa60a06a702be4b5d3c1aa1dbd3d339c60bc5c4e668e2028aa46789aca34e4cd",
        "sales": "60a8c8d5a258f8fc5a844d8fdcac833bdb44f387539433f3a48d08cafa822666",
        "sales_orders": "b745f0f38c29ad8eb0c9075f6351bc3031f3156b9e723cb80c12e67055170c3f",
    },
)

REAL_DATASETS: dict[str, RealDataset] = {d.name: d for d in (TPCDS, WANDERBRICKS, NYCTAXI, SIMULATED_RETAIL)}
