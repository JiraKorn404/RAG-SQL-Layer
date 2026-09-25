#!/bin/bash
# Creates the chat history tables and the role that reads and writes them (CHAT_DB_USER):
#   chat_turns      every question and answer, per conversation (thread_id)
#   query_examples  successful question + SQL pairs with embeddings, retrieved as SQL examples
#   turn_metrics    time and tokens of each node of a turn
# - All live in their own schema, chat_memory. The agent's read-only role gets no access to it,
#   so generated SQL can never read past conversations, and it is not indexed by rag-sql-index.
# - The chat role can only SELECT and INSERT there: history is append-only and kept forever.
#   Examples are disabled/enabled by the admin (`rag-sql-history disable|enable`).
# - Requires the pgvector extension (01-extensions.sql).
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
    created_at  timestamptz NOT NULL DEFAULT now(),
    sql_reasoning    text,
    answer_reasoning text,
    model            text
);
-- Databases created before these columns existed get them here.
ALTER TABLE chat_memory.chat_turns
    ADD COLUMN IF NOT EXISTS sql_reasoning    text,
    ADD COLUMN IF NOT EXISTS answer_reasoning text,
    ADD COLUMN IF NOT EXISTS model            text;
CREATE INDEX IF NOT EXISTS chat_turns_thread_idx ON chat_memory.chat_turns (thread_id, id);

COMMENT ON TABLE chat_memory.chat_turns IS 'Agent chat history, one row per question. Append-only.';
COMMENT ON COLUMN chat_memory.chat_turns.standalone IS 'Question rewritten to stand on its own, using earlier turns.';
COMMENT ON COLUMN chat_memory.chat_turns.sql IS 'SQL that produced the answer; NULL when every attempt failed.';
COMMENT ON COLUMN chat_memory.chat_turns.error IS 'Last validation or execution error when the turn failed.';
COMMENT ON COLUMN chat_memory.chat_turns.sql_reasoning IS 'Model thinking behind the last SQL attempt; NULL when the model gave none.';
COMMENT ON COLUMN chat_memory.chat_turns.answer_reasoning IS 'Model thinking behind the answer; NULL when the model gave none.';
COMMENT ON COLUMN chat_memory.chat_turns.model IS 'Chat model that answered, e.g. qwen3.5:9b; NULL for turns saved before it was recorded.';

-- Successful queries, searched by question similarity before the agent writes new SQL.
-- `embedding` has no fixed dimension; searches only compare rows of the current embed_model.
CREATE TABLE IF NOT EXISTS chat_memory.query_examples (
    id           bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    turn_id      bigint      REFERENCES chat_memory.chat_turns (id) ON DELETE SET NULL,
    question     text        NOT NULL,
    sql          text        NOT NULL,
    row_count    integer     NOT NULL,
    embedding    vector      NOT NULL,
    embed_model  text        NOT NULL,
    enabled      boolean     NOT NULL DEFAULT true,
    created_at   timestamptz NOT NULL DEFAULT now()
);
-- md5: a btree entry can't hold arbitrarily long SQL text.
CREATE UNIQUE INDEX IF NOT EXISTS query_examples_unique_idx
    ON chat_memory.query_examples (embed_model, md5(question), md5(sql));

COMMENT ON TABLE chat_memory.query_examples IS 'Successful question + SQL pairs, retrieved by vector similarity as SQL examples.';
COMMENT ON COLUMN chat_memory.query_examples.question IS 'Standalone question (follow-ups rewritten using the conversation).';
COMMENT ON COLUMN chat_memory.query_examples.embedding IS 'Embedding of question, made with embed_model.';
COMMENT ON COLUMN chat_memory.query_examples.enabled IS 'false = excluded from retrieval and never re-added.';

-- Time and tokens of each node of a turn (src/rag_sql/metrics.py), one row per node run in
-- run order; a turn's totals are sums over its rows. Rows go when their turn is deleted (admin).
CREATE TABLE IF NOT EXISTS chat_memory.turn_metrics (
    turn_id       bigint   NOT NULL REFERENCES chat_memory.chat_turns (id) ON DELETE CASCADE,
    seq           smallint NOT NULL,
    node          text     NOT NULL,
    attempt       smallint NOT NULL,
    ms            integer  NOT NULL,
    llm_ms        integer,
    load_ms       integer,
    input_tokens  integer,
    output_tokens integer,
    PRIMARY KEY (turn_id, seq)
);

COMMENT ON TABLE chat_memory.turn_metrics IS 'Time and tokens of each node run of a chat turn. Append-only.';
COMMENT ON COLUMN chat_memory.turn_metrics.seq IS 'Position of the node run within the turn, from 0.';
COMMENT ON COLUMN chat_memory.turn_metrics.attempt IS 'SQL attempt the node ran in; 0 before the first SQL generation.';
COMMENT ON COLUMN chat_memory.turn_metrics.ms IS 'Wall time of the node, including model and database calls.';
COMMENT ON COLUMN chat_memory.turn_metrics.llm_ms IS 'Model time reported by Ollama; NULL for nodes without a model call.';
COMMENT ON COLUMN chat_memory.turn_metrics.load_ms IS 'Part of llm_ms spent loading the model into memory (cold start).';

GRANT CONNECT ON DATABASE :"db_name" TO :"chat_user";
GRANT USAGE ON SCHEMA chat_memory TO :"chat_user";
GRANT SELECT, INSERT ON chat_memory.chat_turns TO :"chat_user";
GRANT SELECT, INSERT ON chat_memory.query_examples TO :"chat_user";
GRANT SELECT, INSERT ON chat_memory.turn_metrics TO :"chat_user";
SQL
