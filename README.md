# RAG Chatbot

Retrieval-augmented generation chatbot running on local infrastructure using **Chainlit**, **Haystack**, **Ollama**, and **Qdrant**.

Answers domain questions grounded in static company documents (txt / md / pdf) with citation-style source references. Includes a typed tool call for vacation-days lookup backed by a simple username/password simulation.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12 | Pinned in `.python-version` |
| [uv](https://docs.astral.sh/uv/) | latest | Sole dependency/env manager — no pip/poetry |
| [Ollama](https://ollama.com) | latest | Runs models locally |
| [Docker](https://www.docker.com) | latest | Used to run Qdrant |

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd chatbot
uv sync
```

`uv sync` creates the virtual environment (`.venv/`), pins Python 3.12, and installs all dependencies from `uv.lock`.

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env as needed — defaults work for a standard local setup
```

Key variables (all have sensible defaults):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `CHAT_MODEL` | `llama3.2` | Generation model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `QDRANT_HOST` | `localhost` | Qdrant hostname |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | `chatbot` | Collection name |
| `CORPUS_PATH` | `corpus` | Source document directory |
| `RETRIEVAL_TOP_K` | `5` | Chunks returned per query |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.5` | Minimum similarity score |
| `LOG_FORMAT` | `console` | `console` (dev) or `json` (CI/prod) |

### 3. Pull required Ollama models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 4. Start Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

---

## Running the App

```bash
uv run chainlit run src/ui/app.py
```

Opens a chat interface at `http://localhost:8000` by default.

---

## Ingestion

Place source documents (`.txt`, `.md`, `.pdf`) in the directory configured by `CORPUS_PATH` (default: `corpus/`).

```bash
# Index all documents in CORPUS_PATH
uv run python -m src.ingestion.cli ingest

# Wipe the collection and re-index from scratch
uv run python -m src.ingestion.cli reindex

# Drop the collection without re-indexing
uv run python -m src.ingestion.cli reset
```

---

## Running the Test Suite

```bash
uv run pytest
```

Integration tests require Qdrant and Ollama to be running. Unit tests are self-contained.

---

## Linting and Type Checks

```bash
uv run ruff check .          # lint
uv run ruff format .         # format (in-place)
uv run ruff format --check . # format check only (CI mode)
uv run pyright               # strict type check
```

All four commands must pass cleanly before merging any branch. CI enforces this automatically on every pull request.

---

## Evaluation

```bash
uv run python -m src.evaluation.cli run
```

Runs the benchmark dataset against the live system and prints correctness, citation relevance, and unsupported-claim metrics. CI enforces MVP thresholds:

- Correctness ≥ 80 %
- Citation relevance ≥ 90 % (for answerable questions)
- Unsupported claim rate ≤ 10 %

---

## Project Structure

```
src/
  config/       Typed pydantic-settings configuration (env vars)
  ui/           Chainlit UI layer and session lifecycle
  app/          Chat orchestrator — coordinates retrieval, tool calls, generation
  retrieval/    Query embedding, Qdrant search, reranking, source packaging
  ingestion/    File discovery, extraction, chunking, embedding, indexing
  tools/        Typed tool schemas and external-service adapters
tests/
  unit/         Fast, self-contained tests for business logic
  integration/  End-to-end tests requiring live services
doc/            Architecture and requirement specifications
corpus/         (git-ignored) Source documents for ingestion
```

Module boundaries use `Protocol`-based interfaces — orchestration code never imports infrastructure (Qdrant client, HTTP clients) directly.

---

## Known Limitations

- **Single-user only**: Chainlit's WebSocket-based session model is inherently stateful and not designed for horizontal scaling. See `doc/05-delivery-plan.md` → *Multi-User and Stateless Architecture* for the production path.
- **Local model quality variance**: Answer quality depends on the Ollama model chosen. Model pinning and benchmark-based acceptance gates mitigate regressions.
- **PDF extraction quality**: Complex layouts (tables, columns) may degrade extraction fidelity, which directly affects retrieval quality.
- **No production IAM**: Auth is simulated via username/password at tool-call time. Session-scoped only; no OAuth2/OIDC integration.
- **Prompt injection**: Partially mitigated via source-grounded answering and instruction policy. Full red-teaming is post-MVP.
- **No dynamic document upload**: Corpus changes require a CLI re-index. End-user upload is out of MVP scope.
