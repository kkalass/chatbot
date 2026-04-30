# RAG Chatbot

Retrieval-augmented generation chatbot running on local infrastructure using **Chainlit**, **Haystack**, **Ollama**, and **Qdrant**.

Answers domain questions grounded in static company documents (txt / md) with citation-style source references. Includes a typed tool call for vacation-days lookup backed by a simple username/password simulation.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12 | Pinned in `.python-version` |
| [uv](https://docs.astral.sh/uv/) | latest | Sole dependency/env manager — no pip/poetry |
| [Ollama](https://ollama.com) | latest | Runs models locally |
| Docker-compatible runtime | latest | Used to run Qdrant (Docker Desktop, Colima+Docker CLI, Podman, Rancher Desktop) |

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
| `EMBEDDING_DIM` | `768` | Embedding vector dimension — must match `EMBEDDING_MODEL` output |
| `SPLIT_LENGTH` | `200` | Words per ingestion chunk |
| `SPLIT_OVERLAP` | `20` | Word overlap between adjacent chunks |
| `LOG_FORMAT` | `console` | `console` (dev) or `json` (CI/prod) |
| `OTEL_ENABLED` | `false` | Enables OpenTelemetry tracing export |
| `OTEL_SERVICE_NAME` | `chatbot` | Trace resource service name |
| `OTEL_PHOENIX_OTLP_ENDPOINT` | `http://localhost:6006/v1/traces` | Phoenix OTLP/HTTP trace endpoint |
| `OTEL_EXPORT_PHOENIX` | `true` | Enables Phoenix export when `OTEL_ENABLED=true` |
| `OTEL_EXPORT_JAEGER` | `true` | Enables Jaeger export when `OTEL_ENABLED=true` |
| `OTEL_JAEGER_OTLP_ENDPOINT` | `http://localhost:4318/v1/traces` | Jaeger OTLP endpoint (HTTP default; non-`/v1/traces` endpoints use gRPC mode) |
| `OTEL_SAMPLE_RATE` | `1.0` | Root trace sampling ratio (`0.0` to `1.0`) |
| `OTEL_CONSOLE_EXPORT` | `false` | Also print spans to stdout |

### 3. Pull required Ollama models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 4. Start Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

Qdrant dashboard: <http://localhost:6333/dashboard>

macOS quick option (recommended): Colima + Docker CLI

```bash
brew install colima docker
colima start
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

Docker Desktop is not always required. For many local-dev setups, Colima is the simplest free alternative.

### Qdrant Best Practices (Short)

- Keep one embedding model (and vector dimension) per collection.
- Use deterministic point IDs (`doc_id + chunk_index`) for idempotent re-indexing.
- Persist data with a volume in real usage (not only ephemeral container state).
- Treat retrieval params (`top_k`, score threshold, chunk size) as evaluation-controlled settings.

---

## Running the App

```bash
uv run chainlit run src/chatbot/ui/app.py
```

Opens a chat interface at `http://localhost:8000` by default.

---

## Local Tracing (OpenTelemetry + Arize Phoenix + Jaeger)

This project can emit OpenTelemetry traces for the full chat pipeline (UI turn, orchestrator rounds, model call, retrieval tool, citation tool, and Qdrant retrieval).

### 1. Start local Phoenix

```bash
uv tool run --from arize-phoenix python -m phoenix.server.main serve
```

Phoenix UI: <http://localhost:6006>

### 2. Start local Jaeger

```bash
docker run --rm --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -e COLLECTOR_OTLP_GRPC_HOST_PORT=0.0.0.0:4317 \
  -e COLLECTOR_OTLP_HTTP_HOST_PORT=0.0.0.0:4318 \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/all-in-one:1.59
```

Jaeger UI: <http://localhost:16686>

### 3. Enable tracing in `.env`

For a standard local setup, only this is required:

```bash
OTEL_ENABLED=true
```

All other tracing variables already have sensible defaults. In this project, Jaeger defaults to OTLP/HTTP (`/v1/traces`). Set them only when you want to override targets or behavior.

Optional overrides:

```bash
OTEL_SERVICE_NAME=chatbot
OTEL_PHOENIX_OTLP_ENDPOINT=http://localhost:6006/v1/traces
OTEL_EXPORT_PHOENIX=true
OTEL_EXPORT_JAEGER=true
OTEL_JAEGER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
OTEL_SAMPLE_RATE=1.0
OTEL_CONSOLE_EXPORT=false
```

Optional Jaeger OTLP gRPC mode:

```bash
OTEL_JAEGER_OTLP_ENDPOINT=localhost:4317
```

Use gRPC mode only when port `4317` is reachable from the app process.

With `OTEL_ENABLED=true`, both exports are enabled by default. You can disable either backend independently:

```bash
# Phoenix only
OTEL_EXPORT_PHOENIX=true
OTEL_EXPORT_JAEGER=false

# Jaeger only
OTEL_EXPORT_PHOENIX=false
OTEL_EXPORT_JAEGER=true
```

### 4. Start chatbot and generate traffic

```bash
./chatbot.sh
```

Ask one or two questions in Chainlit, then open <http://localhost:6006> and inspect traces for service `chatbot`.

You can inspect the same traces in Phoenix (<http://localhost:6006>) and Jaeger (<http://localhost:16686>) when both exporters are enabled.

### 5. What you should see in traces

- Root UI span for each message (`chat.ui.on_message`)
- Orchestrator spans (`chat.orchestrator.process_message`, `chat.orchestrator.round`)
- Model call span (`chat.model.ollama.stream`) with request/response previews
- Tool spans (`chat.tool.search_documents`, `chat.tool.cite_sources`)
- Retriever span (`chat.retriever.qdrant.retrieve`) with top-k result preview

### 5.1 Tracing Schema (Ownership Rules)

Tracing follows a strict ownership model so each span level contributes one readable view only:

- UI span (`chat.ui.on_message`): user input preview + final emitted assistant response preview.
- Orchestrator spans (`chat.orchestrator.*`): control-flow only (round state, tool dispatch summaries, citation-pass diagnostics).
- Model span (`chat.model.ollama.stream`): compact request message summary + model output preview.
- Tool spans (`chat.tool.*`): tool input plus tool-specific result summary.
- Retriever span (`chat.retriever.qdrant.retrieve`): retrieval parameters + compact chunk/content previews.

Canonical span names are centralized in `src/chatbot/observability/schema.py` and should be reused instead of hard-coded string literals.

### 6. Troubleshooting

- No traces in Phoenix:
  - Verify Phoenix is running on `http://localhost:6006`.
  - Verify `OTEL_ENABLED=true`, `OTEL_EXPORT_PHOENIX=true`, and endpoint matches `http://localhost:6006/v1/traces`.
- No traces in Jaeger:
  - Verify Jaeger is running with OTLP enabled (`COLLECTOR_OTLP_ENABLED=true`).
  - Verify Jaeger OTLP receivers are bound to all interfaces (`0.0.0.0`) when running in Docker.
  - For default OTLP/HTTP mode, verify endpoint is `http://localhost:4318/v1/traces`.
  - For gRPC mode, verify port mapping includes `-p 4317:4317` and endpoint is `localhost:4317`.
  - Verify `OTEL_ENABLED=true` and `OTEL_EXPORT_JAEGER=true`.
  - Set `OTEL_CONSOLE_EXPORT=true` temporarily to confirm spans are produced.
- Too many traces:
  - Lower `OTEL_SAMPLE_RATE`, e.g. `0.2`.

---

## Ingestion

Place source documents (`.txt`, `.md`) in the directory configured by `CORPUS_PATH` (default: `corpus/`), or use the example documents already provided in `corpus/`.

```bash
# Index (or re-index) all documents in CORPUS_PATH
uv run python -m src.ingest.cli reindex

# Wipe the Qdrant collection and re-index from scratch
uv run python -m src.ingest.cli reset

# Wipe the collection only, without re-indexing
uv run python -m src.ingest.cli reset --wipe-only
```

`reindex` uses an OVERWRITE duplicate policy, so running it again on unchanged files is safe — it replaces stale vectors rather than creating duplicates.

---

## Running the Test Suite

```bash
# Unit tests (no external services needed)
uv run pytest tests/unit/

# Integration tests (requires Qdrant + Ollama running)
INTEGRATION_TESTS=1 uv run pytest tests/integration/

# All tests
uv run pytest
```

Integration tests are skipped automatically unless `INTEGRATION_TESTS=1` is set.  
To run them: start Qdrant and Ollama first, then run the command above.

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
  settings/     Shared pydantic-settings configuration (Settings + get_settings)
  chatbot/      Chatbot application
    app/          Orchestrator, protocols, prompts
    ui/           Chainlit UI layer and session lifecycle
    tools/        Typed tool schemas and external-service adapters
    infrastructure/
      chat/       Ollama chat model adapter
      embeddings_text/  Query-time text embedder
      retrieval/  Qdrant vector search adapter
    config.py     Settings → chatbot infrastructure config converters
  ingest/       Ingestion pipeline
    pipeline.py   File discovery, chunking, embedding, indexing
    cli.py        Developer CLI (reindex / reset)
    infrastructure/
      embeddings_document/  Document embedder (Ollama)
      document_store/       Qdrant document store
    config.py     Settings → ingest infrastructure config converters
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
- **Password visible in chat**: Chainlit's `AskUserMessage` only supports plain-text input (`AskSpec.type = "text"`); there is no masked/password input in the current API. The user's password is therefore visible as plain text in the chat transcript. Resolution requires either a Chainlit custom element or a future API addition.
- **Prompt injection**: Partially mitigated via source-grounded answering and instruction policy. Full red-teaming is post-MVP.
- **No dynamic document upload**: Corpus changes require a CLI re-index. End-user upload is out of MVP scope.
