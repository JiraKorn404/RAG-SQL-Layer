#!/bin/bash
# Instacart Market Basket Analysis dataset, loaded into its own schema, imba, from
# data/structured/Instacart Market Basket Analysis/*.csv (mounted at /data in the container):
#   departments, aisles, products, orders, order_products
# - order_products__prior.csv and order_products__train.csv are loaded into one table,
#   order_products; orders.eval_set tells which set an order belongs to.
# - The agent's read-only role (APP_DB_USER, from 02-roles.sh) gets SELECT on the schema.
# - About 37M rows: the first load takes a few minutes.
# Runs automatically on first start with an empty data volume. It is idempotent (drops and
# reloads the tables), so on an existing database re-run it with:
#   docker compose exec postgres bash /docker-entrypoint-initdb.d/05-imba.sh
# Re-run `uv run rag-sql-index` afterwards so the agent sees schema changes.

: "${APP_DB_USER:?APP_DB_USER is not set}"

psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -v app_user="$APP_DB_USER" \
    -v owner="$POSTGRES_USER" <<'SQL'
\set data_dir '/data/structured/Instacart Market Basket Analysis'

BEGIN;

SET LOCAL maintenance_work_mem = '512MB';

CREATE SCHEMA IF NOT EXISTS imba;
COMMENT ON SCHEMA imba IS 'Instacart Market Basket Analysis: grocery orders, their products, aisles and departments.';

DROP TABLE IF EXISTS imba.order_products, imba.orders, imba.products, imba.aisles, imba.departments;

CREATE TABLE imba.departments (
    department_id smallint PRIMARY KEY,
    department    text     NOT NULL
);

CREATE TABLE imba.aisles (
    aisle_id smallint PRIMARY KEY,
    aisle    text     NOT NULL
);

CREATE TABLE imba.products (
    product_id    integer  PRIMARY KEY,
    product_name  text     NOT NULL,
    aisle_id      smallint NOT NULL REFERENCES imba.aisles,
    department_id smallint NOT NULL REFERENCES imba.departments
);

CREATE TABLE imba.orders (
    order_id               integer  PRIMARY KEY,
    user_id                integer  NOT NULL,
    eval_set               text     NOT NULL CHECK (eval_set IN ('prior', 'train', 'test')),
    order_number           smallint NOT NULL CHECK (order_number >= 1),
    order_dow              smallint NOT NULL CHECK (order_dow BETWEEN 0 AND 6),
    order_hour_of_day      smallint NOT NULL CHECK (order_hour_of_day BETWEEN 0 AND 23),
    days_since_prior_order smallint CHECK (days_since_prior_order >= 0)
);

-- Keys and indexes are added after the load: much faster than checking 33M rows one by one.
CREATE TABLE imba.order_products (
    order_id          integer  NOT NULL,
    product_id        integer  NOT NULL,
    add_to_cart_order smallint NOT NULL CHECK (add_to_cart_order >= 1),
    reordered         boolean  NOT NULL
);

COMMENT ON TABLE imba.departments IS 'Store departments (21 rows), e.g. produce, dairy eggs, frozen.';
COMMENT ON COLUMN imba.departments.department_id IS 'Department primary key.';
COMMENT ON COLUMN imba.departments.department IS 'Department name, lowercase (e.g. produce, dairy eggs, snacks).';

COMMENT ON TABLE imba.aisles IS 'Store aisles (134 rows); each product sits in one aisle.';
COMMENT ON COLUMN imba.aisles.aisle_id IS 'Aisle primary key.';
COMMENT ON COLUMN imba.aisles.aisle IS 'Aisle name, lowercase (e.g. fresh fruits, yogurt, packaged cheese).';

COMMENT ON TABLE imba.products IS 'Product catalog (about 50k products), with the aisle and department of each.';
COMMENT ON COLUMN imba.products.product_id IS 'Product primary key.';
COMMENT ON COLUMN imba.products.product_name IS 'Product name as sold (e.g. Banana, Organic Strawberries).';
COMMENT ON COLUMN imba.products.aisle_id IS 'Aisle of the product.';
COMMENT ON COLUMN imba.products.department_id IS 'Department of the product.';

COMMENT ON TABLE imba.orders IS 'One row per customer order (about 3.4M orders from about 206k users). Order dates are not known, only the day of week, hour, and days since the previous order.';
COMMENT ON COLUMN imba.orders.order_id IS 'Order primary key.';
COMMENT ON COLUMN imba.orders.user_id IS 'Customer who placed the order (anonymized; there is no users table).';
COMMENT ON COLUMN imba.orders.eval_set IS 'Dataset split: prior (history), train (last order, products known) or test (last order, products not included, so it has no order_products rows).';
COMMENT ON COLUMN imba.orders.order_number IS 'Sequence number of this order for the user: 1 = first order.';
COMMENT ON COLUMN imba.orders.order_dow IS 'Day of week the order was placed, 0-6 (the day each number stands for is not documented; 0 and 1 are the busiest).';
COMMENT ON COLUMN imba.orders.order_hour_of_day IS 'Hour of day the order was placed, 0-23.';
COMMENT ON COLUMN imba.orders.days_since_prior_order IS 'Days since the user''s previous order, capped at 30. NULL for the first order (order_number = 1).';

COMMENT ON TABLE imba.order_products IS 'Products in each order (about 33M rows): one row per order and product. Covers prior and train orders.';
COMMENT ON COLUMN imba.order_products.order_id IS 'Order the product was bought in.';
COMMENT ON COLUMN imba.order_products.product_id IS 'Product bought.';
COMMENT ON COLUMN imba.order_products.add_to_cart_order IS 'Position the product was added to the cart in: 1 = first.';
COMMENT ON COLUMN imba.order_products.reordered IS 'TRUE if the user had bought this product in an earlier order, FALSE if it is the first time.';

\set src :data_dir '/departments.csv'
COPY imba.departments FROM :'src' WITH (FORMAT csv, HEADER true);
\set src :data_dir '/aisles.csv'
COPY imba.aisles FROM :'src' WITH (FORMAT csv, HEADER true);
\set src :data_dir '/products.csv'
COPY imba.products FROM :'src' WITH (FORMAT csv, HEADER true);

-- days_since_prior_order is written as a float ("15.0") and blank for first orders.
CREATE TEMP TABLE orders_raw (
    order_id               integer,
    user_id                integer,
    eval_set               text,
    order_number           smallint,
    order_dow              smallint,
    order_hour_of_day      smallint,
    days_since_prior_order numeric
) ON COMMIT DROP;
\set src :data_dir '/orders.csv'
COPY orders_raw FROM :'src' WITH (FORMAT csv, HEADER true);
INSERT INTO imba.orders
SELECT order_id, user_id, eval_set, order_number, order_dow, order_hour_of_day,
       days_since_prior_order::smallint
FROM orders_raw
ORDER BY order_id;

-- reordered is 0/1, which COPY reads as boolean.
\set src :data_dir '/order_products__prior.csv'
COPY imba.order_products FROM :'src' WITH (FORMAT csv, HEADER true, FREEZE true);
\set src :data_dir '/order_products__train.csv'
COPY imba.order_products FROM :'src' WITH (FORMAT csv, HEADER true, FREEZE true);

ALTER TABLE imba.order_products
    ADD PRIMARY KEY (order_id, product_id),
    ADD FOREIGN KEY (order_id) REFERENCES imba.orders,
    ADD FOREIGN KEY (product_id) REFERENCES imba.products;

CREATE INDEX ON imba.order_products (product_id);
CREATE INDEX ON imba.orders (user_id, order_number);
CREATE INDEX ON imba.products (aisle_id);
CREATE INDEX ON imba.products (department_id);

GRANT USAGE ON SCHEMA imba TO :"app_user";
GRANT SELECT ON ALL TABLES IN SCHEMA imba TO :"app_user";
ALTER DEFAULT PRIVILEGES FOR ROLE :"owner" IN SCHEMA imba GRANT SELECT ON TABLES TO :"app_user";

COMMIT;

ANALYZE imba.departments, imba.aisles, imba.products, imba.orders, imba.order_products;
SQL
