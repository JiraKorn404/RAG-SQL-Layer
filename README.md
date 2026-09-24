# RAG-SQL-Layer

RAG text-to-SQL agent. Ask a question in natural language in a Jupyter notebook or a Streamlit web chat. The agent retrieves relevant schema context from pgvector, writes SQL with a self-hosted Ollama model, runs it read-only against PostgreSQL, and answers from the result. It shows its thinking, the SQL, and the query output along the way.

Stack: PostgreSQL 17 + pgvector (Docker) · Ollama over Tailscale (`langchain_ollama`) · LangGraph · Jupyter · Streamlit (Docker).

See [CLAUDE.md](CLAUDE.md) for architecture, module layout and conventions.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (manages Python 3.14 and dependencies)
- Docker with Compose
- An Ollama server reachable over Tailscale, with your chat and embedding models pulled

## Setup

### 1. Configure

```sh
cp .env.example .env
```

Edit `.env`. At minimum, set `POSTGRES_PASSWORD`, `APP_DB_PASSWORD`, `CHAT_DB_PASSWORD` and `OLLAMA_BASE_URL`.

### 2. Start PostgreSQL

```sh
docker compose up -d
docker compose ps          # wait for "healthy"
```

On first start (empty volume), the scripts in `db/init/` enable pgvector, create the read-only role `APP_DB_USER`, and create the chat history table (`chat_memory.chat_turns`) with its own role `CHAT_DB_USER`. The agent's SQL only ever runs as the read-only role, which can't see chat history.

If the database volume already existed before these scripts were added, run them manually (both are safe to re-run):

```sh
docker compose up -d       # recreates the container if .env gained new keys
docker compose exec postgres bash /docker-entrypoint-initdb.d/02-roles.sh
docker compose exec postgres bash /docker-entrypoint-initdb.d/04-chat-memory.sh
```

> Git Bash on Windows rewrites the container path. Prefix the command with `MSYS_NO_PATHCONV=1`, or run it from PowerShell.

### 3. Prepare Ollama (on the Ollama machine)

```sh
# Listen on all interfaces, including Tailscale
OLLAMA_HOST=0.0.0.0 ollama serve

ollama pull qwen3:14b            # OLLAMA_CHAT_MODEL
ollama pull nomic-embed-text     # OLLAMA_EMBED_MODEL
```

Check it from this machine:

```sh
curl http://<tailscale-host>:11434/api/tags
```

### 4. Install and index

```sh
uv sync                    # creates .venv, installs the package + dev tools
uv run rag-sql-index       # embed the schema + few-shot examples into pgvector
```

### 5. Run

```sh
uv run jupyter lab notebooks/demo.ipynb
```

```python
from rag_sql.display import run_and_display

run_and_display("Which department has the highest average salary?")
```

Follow-up questions go in a `Chat`, which keeps one conversation. Every question and answer is saved in Postgres and kept.

```python
from rag_sql.display import Chat, show_threads

chat = Chat()
chat.ask("Which department has the highest average salary?")
chat.ask("And the lowest?")  # rewritten to a standalone question using the history

chat.show_history()  # this conversation's saved turns
show_threads()  # all saved conversations
chat = Chat("<thread id>")  # continue one later, even after a restart
```

### Web UI

`docker compose up -d` also starts a Streamlit chat at **http://localhost:8501**. It's reachable from this device only. Pick the chat model in the sidebar: it lists the chat models installed on your Ollama server (embedding models are left out) and starts on `OLLAMA_CHAT_MODEL`. Each reply names the model that answered, and it's saved with the conversation. The model's thinking streams in live, followed by the SQL, the result table and the answer. The sidebar lists saved conversations (the same ones as `show_threads()`), and **New chat** starts another. The open conversation is in the URL, so a page refresh keeps it.

```sh
docker compose up -d --build     # rebuild the web image after code changes
docker compose logs -f web       # app logs
```

The web container uses only the read-only and chat roles. It never gets the admin password, and `.env` isn't copied into the image. To use another host port, set `WEB_PORT` in `.env`.

To run it without Docker (from `.env`, with `POSTGRES_HOST=localhost`):

```sh
uv sync --extra web
uv run streamlit run src/rag_sql/web.py
```

### Query history

Every query that runs and returns rows is saved with an embedding of its question. For each new question, the agent retrieves up to `QUERY_HISTORY_K` (default 5) similar past queries from pgvector and shows them to the model as extra examples. They appear in the notebook as "Similar past queries". Only matches with a similarity of at least `QUERY_HISTORY_MIN_SIMILARITY` (default 0.75) are used, so a new kind of question can get none.

```sh
uv run rag-sql-history backfill          # add successful questions already in chat history
uv run rag-sql-history list --sql        # see what is stored, with ids
uv run rag-sql-history disable 12 15     # a query ran but was wrong: stop using it (enable to undo)
```

Run `backfill` again after changing `OLLAMA_EMBED_MODEL`.

## Development

```sh
uv run pytest                      # unit tests (no network)
uv run pytest -m integration       # needs Postgres + Ollama
uv run ruff check . && uv run ruff format .
```
