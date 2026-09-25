#!/bin/bash
# Retail data warehouse (synthetic), loaded into its own schema, retail, from
# "data/structured/Retail Data Warehouse/*.csv" (mounted at /data in the container):
#   categories, suppliers, products, stores, employees, customers, promotions,
#   orders, order_items, payments, shipments, returns
# - The values are generated independently: order_items.price is not products.price,
#   payments.amount is not the sum of the order's lines, and refunds can exceed the line total.
#   The comments below say so, so the model picks the right column for each question.
# - The agent's read-only role (APP_DB_USER, from 02-roles.sh) gets SELECT on the schema.
# - About 1.6M rows: loads in under a minute.
# Runs automatically on first start with an empty data volume. It is idempotent (drops and
# reloads the tables), so on an existing database re-run it with:
#   docker compose exec postgres bash /docker-entrypoint-initdb.d/06-retail.sh
# Re-run `uv run rag-sql-index` afterwards so the agent sees schema changes.

: "${APP_DB_USER:?APP_DB_USER is not set}"

psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -v app_user="$APP_DB_USER" \
    -v owner="$POSTGRES_USER" <<'SQL'
\set data_dir '/data/structured/Retail Data Warehouse'

BEGIN;

CREATE SCHEMA IF NOT EXISTS retail;
COMMENT ON SCHEMA retail IS 'Retail data warehouse (synthetic): customers, stores, products, orders with their lines, payments, shipments and returns, 2020-2023.';

DROP TABLE IF EXISTS
    retail.returns, retail.shipments, retail.payments, retail.order_items, retail.orders,
    retail.promotions, retail.customers, retail.employees, retail.stores, retail.products,
    retail.suppliers, retail.categories;

CREATE TABLE retail.categories (
    category_id   smallint PRIMARY KEY,
    category_name text     NOT NULL UNIQUE
);

CREATE TABLE retail.suppliers (
    supplier_id smallint PRIMARY KEY,
    country     text     NOT NULL
);

CREATE TABLE retail.products (
    product_id  integer  PRIMARY KEY,
    category_id smallint NOT NULL REFERENCES retail.categories,
    supplier_id smallint NOT NULL REFERENCES retail.suppliers,
    price       integer  NOT NULL CHECK (price > 0)
);

CREATE TABLE retail.stores (
    store_id smallint PRIMARY KEY,
    city     text     NOT NULL
);

CREATE TABLE retail.employees (
    employee_id integer  PRIMARY KEY,
    store_id    smallint NOT NULL REFERENCES retail.stores,
    salary      integer  NOT NULL CHECK (salary > 0)
);

CREATE TABLE retail.customers (
    customer_id integer PRIMARY KEY,
    city        text    NOT NULL,
    signup_date date    NOT NULL
);

CREATE TABLE retail.promotions (
    promotion_id smallint PRIMARY KEY,
    discount     smallint NOT NULL CHECK (discount BETWEEN 0 AND 100)
);

CREATE TABLE retail.orders (
    order_id     integer  PRIMARY KEY,
    customer_id  integer  NOT NULL REFERENCES retail.customers,
    store_id     smallint NOT NULL REFERENCES retail.stores,
    order_date   date     NOT NULL,
    promotion_id smallint NOT NULL REFERENCES retail.promotions
);

CREATE TABLE retail.order_items (
    order_item_id integer  PRIMARY KEY,
    order_id      integer  NOT NULL REFERENCES retail.orders,
    product_id    integer  NOT NULL REFERENCES retail.products,
    qty           smallint NOT NULL CHECK (qty > 0),
    price         integer  NOT NULL CHECK (price > 0)
);

CREATE TABLE retail.payments (
    payment_id integer PRIMARY KEY,
    order_id   integer NOT NULL UNIQUE REFERENCES retail.orders,
    amount     integer NOT NULL CHECK (amount > 0)
);

CREATE TABLE retail.shipments (
    shipment_id integer PRIMARY KEY,
    order_id    integer NOT NULL UNIQUE REFERENCES retail.orders,
    status      text    NOT NULL CHECK (status IN ('shipped', 'delivered', 'late'))
);

CREATE TABLE retail.returns (
    return_id     integer PRIMARY KEY,
    order_item_id integer NOT NULL REFERENCES retail.order_items,
    refund        integer NOT NULL CHECK (refund > 0)
);

COMMENT ON TABLE retail.categories IS 'Product categories (30 rows). Names are placeholders: Cat_1 to Cat_30.';
COMMENT ON COLUMN retail.categories.category_id IS 'Category primary key.';
COMMENT ON COLUMN retail.categories.category_name IS 'Category name, Cat_<n> (e.g. Cat_7); carries no meaning beyond the id.';

COMMENT ON TABLE retail.suppliers IS 'Product suppliers (200 rows) and their country.';
COMMENT ON COLUMN retail.suppliers.supplier_id IS 'Supplier primary key.';
COMMENT ON COLUMN retail.suppliers.country IS 'Supplier country: China, India or USA.';

COMMENT ON TABLE retail.products IS 'Products (10k rows), with category, supplier and list price. Products have no names: refer to them by product_id.';
COMMENT ON COLUMN retail.products.product_id IS 'Product primary key.';
COMMENT ON COLUMN retail.products.category_id IS 'Category of the product.';
COMMENT ON COLUMN retail.products.supplier_id IS 'Supplier of the product.';
COMMENT ON COLUMN retail.products.price IS 'List price, whole currency units (100-4999; currency not stated). Not what customers paid: that is order_items.price.';

COMMENT ON TABLE retail.stores IS 'Stores (100 rows) and their city.';
COMMENT ON COLUMN retail.stores.store_id IS 'Store primary key.';
COMMENT ON COLUMN retail.stores.city IS 'Store city: Bangalore, Delhi, Mumbai or Pune.';

COMMENT ON TABLE retail.employees IS 'Store employees (1000 rows): the store each works at and their salary. Not the same as public.employees.';
COMMENT ON COLUMN retail.employees.employee_id IS 'Employee primary key.';
COMMENT ON COLUMN retail.employees.store_id IS 'Store the employee works at.';
COMMENT ON COLUMN retail.employees.salary IS 'Salary, whole currency units (20046-79949; period and currency not stated).';

COMMENT ON TABLE retail.customers IS 'Customers (50k rows): home city and signup date.';
COMMENT ON COLUMN retail.customers.customer_id IS 'Customer primary key.';
COMMENT ON COLUMN retail.customers.city IS 'Customer home city: Bangalore, Delhi, Mumbai or Pune. Can differ from the city of the store they ordered at (stores.city).';
COMMENT ON COLUMN retail.customers.signup_date IS 'Date the customer signed up, 2019-01-01 to 2024-01-01. Can be later than some of their orders.';

COMMENT ON TABLE retail.promotions IS 'Promotions (50 rows). Every order has one.';
COMMENT ON COLUMN retail.promotions.promotion_id IS 'Promotion primary key.';
COMMENT ON COLUMN retail.promotions.discount IS 'Discount of the promotion, most likely a percentage (5-39). Not applied to any stored amount.';

COMMENT ON TABLE retail.orders IS 'Customer orders (300k rows), 2020-01-01 to 2024-01-01: who ordered, at which store, when, with which promotion. Each order has exactly one payment and one shipment, and zero or more order_items (about 41k orders have none).';
COMMENT ON COLUMN retail.orders.order_id IS 'Order primary key.';
COMMENT ON COLUMN retail.orders.customer_id IS 'Customer who placed the order.';
COMMENT ON COLUMN retail.orders.store_id IS 'Store the order was placed at.';
COMMENT ON COLUMN retail.orders.order_date IS 'Date the order was placed.';
COMMENT ON COLUMN retail.orders.promotion_id IS 'Promotion applied to the order (never NULL).';

COMMENT ON TABLE retail.order_items IS 'Order lines (600k rows): product, quantity and unit price paid. Line total = qty * price. Use this for sales by product, category or supplier. A product can appear on more than one line of the same order.';
COMMENT ON COLUMN retail.order_items.order_item_id IS 'Order line primary key.';
COMMENT ON COLUMN retail.order_items.order_id IS 'Order the line belongs to.';
COMMENT ON COLUMN retail.order_items.product_id IS 'Product sold.';
COMMENT ON COLUMN retail.order_items.qty IS 'Quantity, 1-4.';
COMMENT ON COLUMN retail.order_items.price IS 'Unit price paid, whole currency units (100-4999). Differs from the product''s list price (products.price).';

COMMENT ON TABLE retail.payments IS 'Payments (300k rows): exactly one per order. Use amount for revenue by order, customer, store or date. It does not equal the sum of the order''s lines.';
COMMENT ON COLUMN retail.payments.payment_id IS 'Payment primary key.';
COMMENT ON COLUMN retail.payments.order_id IS 'Order paid for (unique: one payment per order).';
COMMENT ON COLUMN retail.payments.amount IS 'Amount paid for the order, whole currency units (100-19999).';

COMMENT ON TABLE retail.shipments IS 'Shipments (300k rows): exactly one per order, with its delivery status.';
COMMENT ON COLUMN retail.shipments.shipment_id IS 'Shipment primary key.';
COMMENT ON COLUMN retail.shipments.order_id IS 'Order shipped (unique: one shipment per order).';
COMMENT ON COLUMN retail.shipments.status IS 'Shipment status, lowercase: shipped (in transit), delivered, or late.';

COMMENT ON TABLE retail.returns IS 'Returns (30k rows): refunds on order lines. An order line can be returned more than once (up to 3 returns).';
COMMENT ON COLUMN retail.returns.return_id IS 'Return primary key.';
COMMENT ON COLUMN retail.returns.order_item_id IS 'Order line returned; join order_items for the order and product.';
COMMENT ON COLUMN retail.returns.refund IS 'Amount refunded, whole currency units (50-4999). Can exceed the line total (qty * price).';

-- Parents first, so the foreign keys can be checked as rows arrive.
\set src :data_dir '/categories.csv'
COPY retail.categories FROM :'src' WITH (FORMAT csv, HEADER true);
\set src :data_dir '/suppliers.csv'
COPY retail.suppliers FROM :'src' WITH (FORMAT csv, HEADER true);
\set src :data_dir '/products.csv'
COPY retail.products FROM :'src' WITH (FORMAT csv, HEADER true);
\set src :data_dir '/stores.csv'
COPY retail.stores FROM :'src' WITH (FORMAT csv, HEADER true);
\set src :data_dir '/employees.csv'
COPY retail.employees FROM :'src' WITH (FORMAT csv, HEADER true);
\set src :data_dir '/customers.csv'
COPY retail.customers FROM :'src' WITH (FORMAT csv, HEADER true);
\set src :data_dir '/promotions.csv'
COPY retail.promotions FROM :'src' WITH (FORMAT csv, HEADER true);
\set src :data_dir '/orders.csv'
COPY retail.orders FROM :'src' WITH (FORMAT csv, HEADER true);
\set src :data_dir '/order_items.csv'
COPY retail.order_items FROM :'src' WITH (FORMAT csv, HEADER true);
\set src :data_dir '/payments.csv'
COPY retail.payments FROM :'src' WITH (FORMAT csv, HEADER true);
\set src :data_dir '/shipments.csv'
COPY retail.shipments FROM :'src' WITH (FORMAT csv, HEADER true);
\set src :data_dir '/returns.csv'
COPY retail.returns FROM :'src' WITH (FORMAT csv, HEADER true);

CREATE INDEX ON retail.products (category_id);
CREATE INDEX ON retail.products (supplier_id);
CREATE INDEX ON retail.employees (store_id);
CREATE INDEX ON retail.orders (customer_id);
CREATE INDEX ON retail.orders (store_id);
CREATE INDEX ON retail.orders (order_date);
CREATE INDEX ON retail.order_items (order_id);
CREATE INDEX ON retail.order_items (product_id);
CREATE INDEX ON retail.returns (order_item_id);

GRANT USAGE ON SCHEMA retail TO :"app_user";
GRANT SELECT ON ALL TABLES IN SCHEMA retail TO :"app_user";
ALTER DEFAULT PRIVILEGES FOR ROLE :"owner" IN SCHEMA retail GRANT SELECT ON TABLES TO :"app_user";

COMMIT;

ANALYZE retail.categories, retail.suppliers, retail.products, retail.stores, retail.employees,
    retail.customers, retail.promotions, retail.orders, retail.order_items, retail.payments,
    retail.shipments, retail.returns;
SQL
