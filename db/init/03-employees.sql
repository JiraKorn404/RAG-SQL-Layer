-- Employees table, loaded and cleaned from data/structured/employees.csv
-- (mounted at /data in the container). Runs automatically on first start with an empty data
-- volume. It is idempotent (reloads the table), so on an existing database re-run it with:
--   docker compose exec postgres psql -v ON_ERROR_STOP=1 -U postgres -d <POSTGRES_DB> -f /docker-entrypoint-initdb.d/03-employees.sql
-- Re-run `uv run rag-sql-index` afterwards so the agent sees schema changes.

BEGIN;

CREATE TABLE IF NOT EXISTS employees (
    emp_id             integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    emp_name           text          NOT NULL UNIQUE,
    department         text,
    join_date          timestamp     NOT NULL,
    salary             numeric(12, 2) CHECK (salary >= 0),
    age                smallint      CHECK (age BETWEEN 16 AND 100),
    performance_rating smallint      CHECK (performance_rating BETWEEN 1 AND 5),
    city               text          NOT NULL,
    remote_work        boolean
);

COMMENT ON TABLE employees IS 'One row per employee: department, pay, age, performance and work location.';
COMMENT ON COLUMN employees.emp_id IS 'Surrogate primary key.';
COMMENT ON COLUMN employees.emp_name IS 'Employee name, unique (e.g. Employee_1).';
COMMENT ON COLUMN employees.department IS 'One of Engineering, Finance, HR, Marketing, Sales. NULL when unknown.';
COMMENT ON COLUMN employees.join_date IS 'Date and time the employee joined the company.';
COMMENT ON COLUMN employees.salary IS 'Annual salary. NULL when unknown.';
COMMENT ON COLUMN employees.age IS 'Age in years. NULL when unknown.';
COMMENT ON COLUMN employees.performance_rating IS 'Performance rating from 1 (lowest) to 5 (highest). NULL when unrated.';
COMMENT ON COLUMN employees.city IS 'Office city: Bangalore, Chennai, Delhi, Mumbai or Pune.';
COMMENT ON COLUMN employees.remote_work IS 'TRUE if the employee works remotely, FALSE if not, NULL when unknown.';

-- Load the raw file as text so nothing is rejected before cleaning.
CREATE TEMP TABLE employees_raw (
    emp_name           text,
    department         text,
    join_date          text,
    salary             text,
    age                text,
    performance_rating text,
    city               text,
    remote_work        text
) ON COMMIT DROP;

COPY employees_raw FROM '/data/structured/employees.csv' WITH (FORMAT csv, HEADER true);

TRUNCATE employees RESTART IDENTITY;

-- Cleaning: drop exact duplicate rows, blanks and 'nan' become NULL, department casing is
-- normalized, and yes/no variants become booleans.
INSERT INTO employees (
    emp_name, department, join_date, salary, age, performance_rating, city, remote_work
)
SELECT
    trim(emp_name),
    CASE upper(nullif(trim(department), ''))
        WHEN 'HR' THEN 'HR'
        ELSE initcap(nullif(trim(department), ''))
    END,
    trim(join_date)::timestamp,
    nullif(trim(salary), '')::numeric(12, 2),
    nullif(trim(age), '')::numeric::smallint,
    nullif(trim(performance_rating), '')::numeric::smallint,
    initcap(trim(city)),
    CASE lower(trim(remote_work))
        WHEN 'yes' THEN true
        WHEN 'no' THEN false
    END
FROM (SELECT DISTINCT * FROM employees_raw) AS deduped
ORDER BY trim(join_date)::timestamp;

COMMIT;
