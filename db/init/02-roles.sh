#!/bin/bash
# Creates the read-only role the agent connects as (APP_DB_USER).
# Runs automatically on first start with an empty data volume. It is idempotent, so on an
# existing database re-run it with:
#   docker compose exec postgres bash /docker-entrypoint-initdb.d/02-roles.sh

: "${APP_DB_USER:?APP_DB_USER is not set}"
: "${APP_DB_PASSWORD:?APP_DB_PASSWORD is not set}"

psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -v app_user="$APP_DB_USER" \
    -v app_password="$APP_DB_PASSWORD" \
    -v db_name="$POSTGRES_DB" \
    -v owner="$POSTGRES_USER" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN', :'app_user')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'app_user') \gexec

ALTER ROLE :"app_user" WITH LOGIN PASSWORD :'app_password';
-- Defense in depth: every transaction for this role is read-only, even if the SQL guard is bypassed.
ALTER ROLE :"app_user" SET default_transaction_read_only = on;

GRANT CONNECT ON DATABASE :"db_name" TO :"app_user";
GRANT USAGE ON SCHEMA public TO :"app_user";
GRANT SELECT ON ALL TABLES IN SCHEMA public TO :"app_user";
-- Tables the admin user creates later are readable too.
ALTER DEFAULT PRIVILEGES FOR ROLE :"owner" IN SCHEMA public GRANT SELECT ON TABLES TO :"app_user";
SQL
