# RAG-SQL-Layer

RAG text-to-SQL agent. Ask a question in natural language in a Jupyter notebook. The agent retrieves relevant schema context from pgvector, writes SQL with a self-hosted Ollama model, runs it read-only against PostgreSQL, and answers from the result. It shows its thinking, the SQL, and the query output along the way.

Stack: PostgreSQL 17 + pgvector (Docker) · Ollama over Tailscale (`langchain_ollama`) · LangGraph · Jupyter.

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

Edit `.env`. At minimum, set `POSTGRES_PASSWORD`, `APP_DB_PASSWORD` and `OLLAMA_BASE_URL`.

### 2. Start PostgreSQL

```sh
docker compose up -d
docker compose ps          # wait for "healthy"
```

On first start (empty volume), the scripts in `db/init/` enable pgvector and create the read-only role `APP_DB_USER`. The agent only ever connects as that role.

If the database volume already existed before the role script was added, create the role manually (safe to re-run):

```sh
docker compose exec postgres bash /docker-entrypoint-initdb.d/02-roles.sh
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
run_and_display("Top 5 customers by revenue last quarter?")
```

## Development

```sh
uv run pytest                      # unit tests (no network)
uv run pytest -m integration       # needs Postgres + Ollama
uv run ruff check . && uv run ruff format .
```
