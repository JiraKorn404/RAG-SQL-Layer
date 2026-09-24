#!/bin/bash
# Creates the chat history table and the role that reads and writes it (CHAT_DB_USER).
# - History lives in its own schema, chat_memory. The agent's read-only role gets no access to
#   it, so generated SQL can never read past conversations, and it is not indexed for retrieval.
# - The chat role can only SELECT and INSERT there: history is append-only and kept forever.
# Runs automatically on first start with an empty data volume. It is idempotent, so on an
# existing database re-run it with:
#   docker compose exec postgres bash /docker-entrypoint-initdb.d/04-chat-memory.sh

: "${CHAT_DB_USER:?CHAT_DB_USER is not set}"
: "${CHAT_DB_PASSWORD:?CHAT_DB_PASSWORD is not set}"

psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -v chat_user="$CHAT_DB_USER" \
    -v chat_password="$CHAT_DB_PASSWORD" \
    -v db_name="$POSTGRES_DB" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN', :'chat_user')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'chat_user') \gexec

ALTER ROLE :"chat_user" WITH LOGIN PASSWORD :'chat_password';

CREATE SCHEMA IF NOT EXISTS chat_memory;

-- One row per question asked; a conversation is all rows with the same thread_id.
CREATE TABLE IF NOT EXISTS chat_memory.chat_turns (
    id          bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    thread_id   text        NOT NULL,
    question    text        NOT NULL,
    standalone  text        NOT NULL,
    sql         text,
    row_count   integer,
    answer      text        NOT NULL,
    error       text,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chat_turns_thread_idx ON chat_memory.chat_turns (thread_id, id);

COMMENT ON TABLE chat_memory.chat_turns IS 'Agent chat history, one row per question. Append-only.';
COMMENT ON COLUMN chat_memory.chat_turns.standalone IS 'Question rewritten to stand on its own, using earlier turns.';
COMMENT ON COLUMN chat_memory.chat_turns.sql IS 'SQL that produced the answer; NULL when every attempt failed.';
COMMENT ON COLUMN chat_memory.chat_turns.error IS 'Last validation or execution error when the turn failed.';

GRANT CONNECT ON DATABASE :"db_name" TO :"chat_user";
GRANT USAGE ON SCHEMA chat_memory TO :"chat_user";
GRANT SELECT, INSERT ON chat_memory.chat_turns TO :"chat_user";
SQL
