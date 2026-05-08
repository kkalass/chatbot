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
| `CHAT_MODEL_PROVIDER` | `ollama` | `ollama` or `openai_compatible` |
| `CHAT_BASE_URL` | `http://localhost:11434` | Chat model endpoint — Ollama URL or OpenAI-compatible base URL (e.g. `https://api.groq.com/openai/v1`) |
| `CHAT_API_KEY` | — | API key for `openai_compatible` providers (not used for Ollama) |
| `CHAT_MODEL` | `qwen3.5:9b` | Generation model identifier — Ollama name or provider model id (e.g. `qwen3-32b`, `Qwen/Qwen3-235B-A22B`) |
| `EMBEDDING_MODEL_PROVIDER` | `ollama` | Currently only `ollama` is supported |
| `EMBEDDING_BASE_URL` | `http://localhost:11434` | Ollama server URL for embeddings |
| `EMBEDDING_MODEL` | `bge-m3` | Embedding model |
| `QDRANT_HOST` | `localhost` | Qdrant hostname |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | `chatbot` | Collection name |
| `CORPUS_PATH` | `corpus` | Source document directory |
| `RETRIEVAL_TOP_K` | `5` | Chunks returned per query |
| `RETRIEVAL_LLM_TOP_K` | — | Max documents passed to LLM after RRF fusion; default uses `RETRIEVAL_TOP_K` |
| `EMBEDDING_DIM` | `1024` | Embedding vector dimension — must match `EMBEDDING_MODEL` output |
| `SPLIT_LENGTH` | `200` | Words per ingestion chunk |
| `SPLIT_OVERLAP` | `20` | Word overlap between adjacent chunks |
| `VISION_INGESTION_ENABLED` | `true` | Ingest images via a vision model; set `false` to skip all image content |
| `VISION_MODEL` | `qwen2.5vl:7b` | Vision model for image description generation |
| `VISION_PROVIDER` | `ollama` | Vision model backend — currently only `ollama` |
| `VISION_BASE_URL` | `http://localhost:11434` | Ollama server URL for vision-model calls |
| `IMAGE_CACHE_DIR` | `.cache/image_descriptions` | Cache directory for vision-model descriptions (keyed by content hash) |
| `EXTRACTED_IMAGE_DIR` | `.cache/extracted_images` | Directory for PDF-extracted images (surfaced as citation images) |
| `IMAGE_MIN_DIMENSION` | `64` | Drop images whose width or height is below this value (pixels) |
| `IMAGE_MIN_DESCRIPTION_LENGTH` | `40` | Drop descriptions shorter than this (chars) — filters decorative images |
| `LOG_FORMAT` | `console` | `console` (dev) or `json` (CI/prod) |
| `OTEL_ENABLED` | `false` | Enables OpenTelemetry tracing export |
| `OTEL_SERVICE_NAME` | `chatbot` | Trace resource service name |
| `PHOENIX_PROJECT_NAME` | `chatbot-local` | Phoenix project name shown in the UI |
| `OTEL_DEPLOYMENT_ENVIRONMENT` | `development` | OTel deployment environment resource attribute |
| `OTEL_PHOENIX_OTLP_ENDPOINT` | `http://localhost:6006/v1/traces` | Phoenix OTLP/HTTP trace endpoint |
| `OTEL_EXPORT_PHOENIX` | `true` | Enables Phoenix export when `OTEL_ENABLED=true` |
| `OTEL_EXPORT_JAEGER` | `true` | Enables Jaeger export when `OTEL_ENABLED=true` |
| `OTEL_JAEGER_OTLP_ENDPOINT` | `http://localhost:4318/v1/traces` | Jaeger OTLP endpoint (HTTP default; non-`/v1/traces` endpoints use gRPC mode) |
| `OTEL_SAMPLE_RATE` | `1.0` | Root trace sampling ratio (`0.0` to `1.0`) |
| `OTEL_CONSOLE_EXPORT` | `false` | Also print spans to stdout |
| `OTEL_AUTO_INSTRUMENT_HAYSTACK` | `true` | Enable OpenInference auto-instrumentation for Haystack |
| `MODEL_TEMPERATURE` | `0.0` | Chat model temperature (0.0 = deterministic, 1.0 = creative) |
| `MODEL_SEED` | `42` | Chat model seed for reproducible generations |
| `EVAL_ENVIRONMENT` | `local` | Evaluation environment label attached to all traces |
| `EVAL_NAME` | — | Evaluation cycle name (e.g. `rag-prompt-tuning-2026-04`) |
| `EVAL_RUN_ID` | — | Run identifier; auto-generated per process if unset |
| `EVAL_CANDIDATE_ID` | — | Candidate ID for comparing prompt/model variants |
| `EVAL_PROMPT_VERSION_ANSWER` | — | Version label for the answer-generation prompt |
| `EVAL_PROMPT_VERSION_CITATION` | — | Version label for the citation-pass prompt |
| `EVAL_RETRIEVAL_VERSION` | — | Version label for retrieval config; auto-derived from `RETRIEVAL_TOP_K` if unset |
| `EVAL_CORPUS_VERSION` | — | Corpus snapshot identifier |
| `EVAL_DATASET_VERSION` | — | Dataset snapshot identifier |
| `EVAL_JUDGE_MODEL` | `llama3.1:8b` | LLM judge model for eval evaluators |
| `EVAL_JUDGE_PROVIDER` | `ollama` | LLM judge backend: `ollama` or `openai_compatible` |
| `EVAL_JUDGE_BASE_URL` | — | Judge provider URL; defaults to `http://localhost:11434` for Ollama |
| `EVAL_JUDGE_API_KEY` | — | API key for `openai_compatible` judge provider |
| `EVAL_JUDGE_INITIAL_PER_SECOND_REQUEST_RATE` | `1.5` | Initial judge request rate (lower to avoid rate-limit errors) |

### 3. Pull required Ollama models

```bash
# Runtime default chat model
ollama pull qwen3.5:9b

# Embeddings
ollama pull bge-m3

# Vision model (for multi-modal ingestion — can be skipped if VISION_INGESTION_ENABLED=false)
ollama pull qwen2.5vl:7b

# Test chat model (used by pytest via tests/conftest.py)
ollama pull llama3.2
```

The embedding and vision models always run locally via Ollama. The chat model can alternatively run via a cloud provider — see [Using a cloud/remote chat model](#using-a-cloudremote-chat-model) below.
For tests, the suite pins `CHAT_MODEL=llama3.2` to keep test environments fast and reproducible even when the runtime default model changes.

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

### Using a cloud/remote chat model

Set `CHAT_MODEL_PROVIDER=openai_compatible` to replace the local Ollama chat model with any OpenAI-API-compatible endpoint. Embeddings always remain local.

**Groq** (free tier, fast):

```bash
CHAT_MODEL_PROVIDER=openai_compatible
CHAT_BASE_URL=https://api.groq.com/openai/v1
CHAT_API_KEY=gsk_...
CHAT_MODEL=qwen3-32b
```

**Together AI** (pay-as-you-go, larger models):

```bash
CHAT_MODEL_PROVIDER=openai_compatible
CHAT_BASE_URL=https://api.together.xyz/v1
CHAT_API_KEY=...
CHAT_MODEL=Qwen/Qwen3-235B-A22B
```

Any other provider with an OpenAI-compatible `/chat/completions` endpoint works the same way.

---

## Running the App

```bash
uv run chainlit run src/chatbot/ui/app.py
```

Opens a chat interface at `http://localhost:8000` by default.

---

## Local Tracing (OpenTelemetry + Arize Phoenix + Jaeger)

This project can emit OpenTelemetry traces for the full chat pipeline (UI turn, orchestrator rounds, model call, retrieval tool, and Qdrant retrieval).

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
- Model call span (`chat.model.ollama.stream` or `chat.model.openai_compatible.stream`) with request/response previews
- Tool spans (`chat.tool.search_documents`) with tool input and result summary
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

# Integration tests (requires Qdrant + Ollama running; pytest pins CHAT_MODEL=llama3.2)
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
# Run experiment against the default dataset (eval/datasets/rag_questions.jsonl)
uv run --group eval python eval/run_experiment.py

# Custom dataset / experiment name
uv run --group eval python eval/run_experiment.py \
  --dataset-file eval/datasets/rag_questions.jsonl \
  --experiment-name "retrieval-top-k-5"

# Quick sanity check (1 example, no Phoenix upload)
uv run --group eval python eval/run_experiment.py --dry-run
```

Drives the `ChatOrchestrator` directly (bypasses Chainlit) and records results in Arize Phoenix. Requires `OTEL_ENABLED=true` and a running Phoenix instance. See `eval/README.md` for a full feature overview and dataset management guide.

Evaluation metadata (run ID, prompt versions, candidate ID, …) is controlled via `EVAL_*` env vars — see the configuration table above.

---

## Project Structure

```
src/
  shared/       Cross-cutting infrastructure shared by chatbot and ingest
    settings/     Settings + get_settings (pydantic-settings, single source of truth)
    observability/  Logging (structlog) and OpenTelemetry tracing bootstrap
    qdrant/       Shared Qdrant document-store factory and sparse embedder
  chatbot/      Chatbot application
    app/          Orchestrator, credential store, prompts
      citation/   CitationModel, citation parser, citeable-tool messages
    contracts/    Protocol interfaces: ChatModel, Retriever, CredentialStore, …
    ui/           Chainlit UI layer, session lifecycle, composition root
    infrastructure/
      chat/       Ollama and OpenAI-compatible chat model adapters
      embeddings_text/  Query-time text embedder (Ollama)
      retrieval/  Hybrid Qdrant vector+sparse retriever
      tools/
        retrieval/      Document retrieval tool (CiteableTool)
        vacation_days/  Vacation-days lookup tool
    build_from_settings.py  Settings → chatbot component factory
  ingest/       Ingestion pipeline
    app/          Ingestion pipeline logic (chunking, embedding, indexing)
    contracts/    Protocol interfaces: FormatHandler, DocumentConverter, …
    cli/          Developer CLI (reindex / reset / wipe-only)
    infrastructure/
      converters/       Format converters (PDF, image)
      embeddings_document/  Document embedder (Ollama)
      image_cache/      Content-hash cache for vision-model descriptions
      image_description.py  Image description service and filter config
      vision/           Vision model adapter (Ollama)
    build_from_settings.py  Settings → ingest component factory
eval/
  run_experiment.py  Offline evaluation runner (Phoenix Experiments)
  datasets/          Curated evaluation datasets (JSONL)
  results/           Stored experiment result files
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
- **Local model quality variance**: Answer quality depends on the model chosen. Model pinning and benchmark-based acceptance gates mitigate regressions.
- **PDF extraction quality**: Complex layouts (tables, columns) may degrade extraction fidelity, which directly affects retrieval quality.
- **No production IAM**: Auth is simulated via username/password at tool-call time. Session-scoped only; no OAuth2/OIDC integration.
- **Prompt injection**: Partially mitigated via source-grounded answering and instruction policy. Full red-teaming is post-MVP.
- **No dynamic document upload**: Corpus changes require a CLI re-index. End-user upload is out of MVP scope.
