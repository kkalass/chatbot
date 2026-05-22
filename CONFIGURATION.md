# Configuration Reference

All settings are read from environment variables (or a `.env` file in the project root). Copy `.env.example` to `.env` to get started — all variables have sensible defaults for a standard local setup.

---

## Chat Model

| Variable | Default | Description |
|---|---|---|
| `CHAT_MODEL_PROVIDER` | `ollama` | `ollama` or `openai_compatible` |
| `CHAT_BASE_URL` | `http://localhost:11434` | Chat model endpoint — Ollama URL or OpenAI-compatible base URL (e.g. `https://api.groq.com/openai/v1`) |
| `CHAT_API_KEY` | — | API key for `openai_compatible` providers (not used for Ollama) |
| `CHAT_MODEL` | `qwen3.5:9b` | Generation model identifier — Ollama name or provider model id (e.g. `qwen3-32b`, `Qwen/Qwen3-235B-A22B`) |
| `MODEL_TEMPERATURE` | `0.0` | Chat model temperature (0.0 = deterministic, 1.0 = creative) |
| `MODEL_SEED` | `42` | Chat model seed for reproducible generations |

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

## Embedding Model

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL_PROVIDER` | `ollama` | Currently only `ollama` is supported |
| `EMBEDDING_BASE_URL` | `http://localhost:11434` | Ollama server URL for embeddings |
| `EMBEDDING_MODEL` | `bge-m3` | Embedding model |
| `EMBEDDING_DIM` | `1024` | Embedding vector dimension — must match `EMBEDDING_MODEL` output |

> **Note:** The embedding model has an outsized impact on retrieval quality. Changing it requires wiping and re-indexing the entire collection.

---

## Qdrant

| Variable | Default | Description |
|---|---|---|
| `QDRANT_HOST` | `localhost` | Qdrant hostname |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | `chatbot` | Collection name |

### Best Practices

- Keep one embedding model (and vector dimension) per collection.
- Use deterministic point IDs (`doc_id + chunk_index`) for idempotent re-indexing.
- Persist data with a volume in real usage (not only ephemeral container state).
- Treat retrieval params (`top_k`, score threshold, chunk size) as evaluation-controlled settings.

---

## Ingestion and Retrieval

| Variable | Default | Description |
|---|---|---|
| `CORPUS_PATH` | `corpus` | Source document directory |
| `RETRIEVAL_TOP_K` | `5` | Chunks returned per query |
| `RETRIEVAL_LLM_TOP_K` | — | Max documents passed to LLM after RRF fusion; default uses `RETRIEVAL_TOP_K` |
| `SPLIT_LENGTH` | `200` | Words per ingestion chunk |
| `SPLIT_OVERLAP` | `20` | Word overlap between adjacent chunks |

---

## Vision / Image Ingestion

| Variable | Default | Description |
|---|---|---|
| `VISION_INGESTION_ENABLED` | `true` | Ingest images via a vision model; set `false` to skip all image content |
| `VISION_MODEL` | `qwen2.5vl:7b` | Vision model for image description generation |
| `VISION_PROVIDER` | `ollama` | Vision model backend — currently only `ollama` |
| `VISION_BASE_URL` | `http://localhost:11434` | Ollama server URL for vision-model calls |
| `IMAGE_CACHE_DIR` | `.cache/image_descriptions` | Cache directory for vision-model descriptions (keyed by content hash) |
| `EXTRACTED_IMAGE_DIR` | `.cache/extracted_images` | Directory for PDF-extracted images (surfaced as citation images) |
| `IMAGE_MIN_DIMENSION` | `64` | Drop images whose width or height is below this value (pixels) |
| `IMAGE_MIN_DESCRIPTION_LENGTH` | `40` | Drop descriptions shorter than this (chars) — filters decorative images |

---

## Logging

| Variable | Default | Description |
|---|---|---|
| `LOG_FORMAT` | `console` | `console` (dev) or `json` (CI/prod) |

---

## OpenTelemetry / Tracing

| Variable | Default | Description |
|---|---|---|
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

---

## Evaluation

| Variable | Default | Description |
|---|---|---|
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
