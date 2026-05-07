# Architecture

## High-Level Design
The application is a locally-run RAG chatbot with explicit separation of concerns. Each subsystem package (`chatbot`, `ingest`) follows a `contracts/ → app/ → infrastructure/ → ui|cli/` layering. Cross-subsystem concerns live in `src/shared/`.

## Components
- UI Layer (Chainlit)
  - Handles user interaction, streaming output, and source display.
  - Renders citations as inline numbered references (`[N]`) plus a deduplicated *Sources* section appended after the answer; hallucinated citations and unsubstantiated claims are rendered with their own visual marker.
  - Suspends the stream on `AuthRequiredEvent`, displays a login form, writes the collected credentials into the session-scoped `CredentialStore`, and resolves the future to resume the orchestrator.
- Application Layer
  - **Orchestrator** (`src/chatbot/app/orchestrator.py`): single entry point for the UI. Coordinates message history and the agentic tool-call loop. Depends only on Protocol interfaces from `chatbot/contracts/`. Retrieval is modelled as a regular `Tool` (`search_documents`); the LLM decides if and when to call it.
  - **CitationModel** (`src/chatbot/app/citation/`): wraps a `ChatModel` and implements the full citation protocol — augments the system prompt with one tool-agnostic citation instruction plus per-tool fragments contributed by registered `CiteableTool`s, parses inline citation markers from the model's text stream, and resolves each marker against a global token index built from prior tool results. The orchestrator consumes a stream of `str | list[ToolCallInfo] | Citation | HallucinatedCitation | UnsubstantiatedClaim | ThinkingContent`.
  - **Prompts**: `Prompts` lives in `chatbot/contracts/prompts.py` (it appears in `ModelProfile.adjust_prompts()`); default content lives in `src/chatbot/app/chat_prompts.py::DEFAULT_PROMPTS`. The orchestrator receives a model-specific `ModelProfile` via constructor injection and derives effective prompts by applying it to `DEFAULT_PROMPTS`.
- Retrieval Layer
  - Query embedding, hybrid dense+sparse vector search (Qdrant) with reciprocal-rank-fusion, source packaging.
- Ingestion Layer
  - File discovery, converter routing by input type (`txt`, `md`, `pdf`, image), character-aware chunking strategy selection, embedding, indexing.
  - Sidecar metadata merge (`<document>.meta.json`) at conversion time; sidecars are not treated as standalone documents.
  - Image-bearing PDFs and standalone image files run through `ImageDescriptionService` (vision LLM + content-hash cache + size/length filter) before becoming `image_description`-kind chunks.
  - Micro-batch orchestration for bounded-memory ingestion.
- Tool Integration Layer
  - **`Tool`** Protocol (`chatbot/contracts/tools.py`): `schema: ToolSchema`, `display_name: I18nMessage`, `describe_call(args) -> I18nMessage`, `async execute(args: JsonObject) -> JsonObject`. Tools are constructed once per session with all dependencies (callbacks, adapters, credential store) injected at construction time. Tools never import the orchestrator or any UI module.
  - **`CiteableTool`** Protocol (`chatbot/contracts/citation.py`): extends `Tool` with `cite_instructions() -> CiteInstructions`, `render_for_history(result) -> ToolHistoryRendering`, and `enrich(raw, unit) -> Citation`. The `render_for_history` rendering must embed each `CitableUnit.citation_token` verbatim so the model can copy it into a marker on subsequent turns.
  - **Citation wire format**: the model emits citations inline as marker blocks `<°_quote_°>{"ref":"<token>"}</°_quote_°>` or `<°_quote_°>{"kind":"unsubstantiated"}</°_quote_°>`. There is no separate `cite_sources` tool. `CitationModel` parses these markers from the streaming text, resolves `ref` against the global citation-token index, and yields typed `Citation` / `HallucinatedCitation` / `UnsubstantiatedClaim` events; plain text between markers is passed through as `str` chunks.
  - **`RetrievalTool`** (`search_documents`): `CiteableTool` implementation wrapping the `Retriever` Protocol. Returns retrieved chunks as structured JSON; renders them with embedded citation tokens; enriches resolved citations into typed `DocumentCitation`s.
  - **`VacationDaysTool`** (`get_vacation_days`): `Tool` implementation that reads credentials from the injected `CredentialStore`. Raises `AuthRequiredException` when credentials are missing or rejected by the adapter.
  - **`ToolInputModel`**: shared Pydantic base model for LLM-facing tool arguments. Structured fields (`list`/`dict`) are pre-coerced from JSON-serialized strings before validation.
- Model Runtime
  - Ollama-hosted local models for generation in MVP (`OllamaChatModel`); an OpenAI-compatible adapter is also available. Both adapters can be wrapped by `TextToolCallParsingWrapper` for models that emit tool calls as text rather than via the native tool-call protocol.
  - Per-model quirks (template overrides, prompt adjustments) are encoded in `ModelProfile` implementations under `chatbot/infrastructure/chat/_model_profile.py`.
  - Ingestion embedder is consumed through an injected boundary (`DocumentEmbedder`) so provider/runtime can be swapped without changing ingestion orchestration.
- Observability Layer
  - OpenTelemetry tracing is configured at app startup via `arize-phoenix-otel`'s `register()`, which creates a `TracerProvider` that exports spans to a Phoenix-compatible OTLP endpoint (default: `http://localhost:6006/v1/traces`).
  - Generic, subsystem-agnostic OpenInference helpers live in `src/shared/observability/openinference.py`. Chatbot-specific OpenInference payload builders (`build_llm_attributes`, `build_retriever_attributes`, `summarize_messages`, …) live in `src/chatbot/infrastructure/observability/`. Stable span operation names (`SPAN_CHAT_*`) live in `src/chatbot/contracts/observability.py` so the `app` layer can reference them without depending on `infrastructure/`.
  - Span kinds used: `CHAIN` (UI turn, orchestrator step, tool dispatch), `LLM` (model streaming), `RETRIEVER` (Qdrant query), `TOOL` (tool execution).
  - **Session propagation**: `using_session_attributes(session_id)` from `openinference.instrumentation` is used in the UI `on_message` handler to propagate session context to all child spans in a turn.
  - Optional **Haystack auto-instrumentation** is enabled by default (`OTEL_AUTO_INSTRUMENT_HAYSTACK=true`).
  - Tracing-only payload construction is kept out of business logic: request/response/error attribute assembly lives in module-level `_trace_*` helper functions, and control-flow code only invokes those helpers. Trace payload attributes are previewed with truncation to keep spans inspectable while bounding attribute size.
  - **Local Phoenix setup**: run `uv run python -m phoenix.server.main serve` to start the Phoenix UI at `http://localhost:6006`. Traces are visible under the project name `chatbot-local` (configurable via `PHOENIX_PROJECT_NAME`).

## Binding Points
- Binding points are documented at consumer boundaries (tool constructor and composition root), not via mandatory nominal inheritance on implementation classes.
- Vacation-days wiring in `on_chat_start`:
  - `SimulatedVacationDaysAdapter` is bound to `VacationDaysService`.
  - The session-scoped `CredentialStore` instance is injected into `VacationDaysTool`.
- Composition is constrained to the runtime entry layer: only `src/chatbot/ui/composition.py` and `src/ingest/cli/composition.py` instantiate infrastructure adapters and wire them into app-layer classes. Settings → config mapping happens here, not in `app/`.
- This keeps Protocol usage structural (PEP 544) while making runtime wiring explicit and discoverable.

## Data Flow
1. Ingestion flow:
   - Load corpus files (txt/md/pdf/image).
   - Route each file to a type-specific converter via the `FormatHandler` registry built in `src/ingest/cli/composition.py`.
   - Merge sidecar metadata (`<document>.meta.json`) into document metadata when present.
   - Select splitter strategy based on document type/character and create chunks; PDF-embedded images are extracted, deduplicated by content hash, described by `ImageDescriptionService`, and emitted as additional `image_description`-kind chunks.
   - Process chunks in micro-batches: embed (dense + sparse) and upsert each batch to Qdrant before loading the next.
2. Query flow:
   - User message arrives via Chainlit; `on_message` opens the UI span and propagates the session id.
   - Orchestrator wraps the configured `ChatModel` in a `CitationModel` (registering all `CiteableTool`s for prompt-fragment contribution and token-index construction), then runs the agentic loop:
     1. Append the user message to history.
     2. Stream from `CitationModel.stream(history, tools)` and yield `str` / `Citation` / `HallucinatedCitation` / `UnsubstantiatedClaim` / `ThinkingContent` to the UI as they arrive.
     3. When the stream surfaces tool calls, dispatch each one: emit `ToolCallStarted`, run `tool.execute(args)`, append the JSON result to history (rendered through `CiteableTool.render_for_history` for citeable tools so the embedded `citation_token`s are visible to the model on subsequent turns), emit `ToolCallFinished`.
     4. Loop until a step ends with no pending tool calls, or until `_MAX_TOOL_STEPS` is reached.
   - When a tool raises `AuthRequiredException`, the orchestrator emits `AuthRequiredEvent` carrying the credential key, service display name, and an `asyncio.Future`. It awaits the future; on `True` it retries the tool call (now finding credentials in the `CredentialStore`); on `False` it appends a user-safe cancellation result and continues.
   - As validated `Citation`s arrive, the orchestrator assigns stable per-session reference numbers (reused on repeat citations of the same canonical key) and yields `NumberedCitation` events; the UI renders these as `[N]` markers in the answer text and accumulates a deduplicated *Sources* section.
   - In parallel, OpenTelemetry spans record request/response previews and counters for each stage to enable deterministic debugging in Phoenix.

### Public stream type
`Orchestrator.process_message` yields `ProcessEvent` (`chatbot/contracts/process.py`):

```python
type ProcessEvent = (
    str
    | NumberedCitation
    | HallucinatedCitation
    | UnsubstantiatedClaim
    | ToolCallStarted
    | ToolCallFinished
    | AuthRequiredEvent
    | ThinkingContent
)
```

Pyright flattens the union, so `match`/`case` + `typing.assert_never` on `ProcessEvent` gives statically-checked exhaustive dispatch over all variants — the Python equivalent of Dart's sealed-class pattern.

## Authentication Flows (MVP)
- The Chainlit UI is accessible without mandatory global app-level login in MVP.
- Authentication is required only for tools backed by external services (currently `get_vacation_days`) and happens at tool-call time.
- Credentials are scoped to the active chat session (`CredentialStore` instance is created per `on_chat_start`) and never persisted as long-term application user accounts.
- Tool-call authentication failures are returned as user-safe results without exposing sensitive details.
- Optional app-level Chainlit authentication can be added later as a separate access-control concern.

## Auth-Protected Tool Sequence (MVP)
Credential collection is mediated by the orchestrator + UI via `AuthRequiredEvent`. The LLM is not involved in the auth flow.

1. LLM calls `get_vacation_days(year=...)` — no credentials in arguments.
2. `VacationDaysTool.execute` looks up its `CredentialStore` slot (`"vacation_days"`).
3. If credentials are absent, the tool raises `AuthRequiredException(credential_key=..., service_display_name=...)`.
4. The orchestrator catches the exception, emits `AuthRequiredEvent` with the key, the localizable service name, and an `asyncio.Future[bool]`, and awaits the future.
5. The UI shows a login form. On submit it writes `(username, password)` into the `CredentialStore` under the key and resolves the future to `True`. On cancel it resolves to `False`.
6. On `True` the orchestrator retries the same tool call; the tool now finds the credentials and calls the service adapter. On `False` the orchestrator appends a user-safe cancellation result and continues.
7. If the adapter rejects credentials at runtime, the tool clears the `CredentialStore` entry and raises `AuthRequiredException` again — the loop above runs once more.

Implementation rules:
- Credentials must never be sourced from model-generated tool arguments or retrieval context; they are managed exclusively by the session-scoped `CredentialStore` and collected via the UI's login form.
- Tools must not import `chainlit` or any UI module directly; the only auth touchpoint is the injected `CredentialStore` Protocol plus raising `AuthRequiredException`.
- The credential store is not a tool and is never registered in the tool list exposed to the LLM.

## Key Design Decisions
- **Retrieval as a tool**: modelled as `search_documents`, not as eager pre-retrieval — the LLM decides when to search and formulates its own queries, enabling multi-hop retrieval and query reformulation from conversation context.
- **Inline citations via markers, not a citation tool**: the model emits `<°_quote_°>{"ref":"<token>"}</°_quote_°>` blocks inline; `CitationModel` parses and resolves them against a global token index built from prior tool results. This eliminates the failure modes of a separate citation pass (tool-selection drift in weak models, isolation from full history) and makes citations stream live with the answer text.
- **`CiteableTool` Protocol as the citation contract**: tools own (a) the prompt fragment that teaches the model how to cite their results, (b) the LLM-visible rendering that embeds citation tokens, and (c) the enrichment from `RawCitation` to typed `Citation`. This keeps the citation layer tool-agnostic.
- **Hallucinated citations are surfaced, not silently dropped**: failed token resolution yields a `HallucinatedCitation` event; the model's explicit `{"kind":"unsubstantiated"}` markers yield `UnsubstantiatedClaim`. Both are rendered in the UI.
- **Session-stable citation reference numbers**: `NumberedCitation.reference_number` is assigned by the orchestrator and reused across the session for the same canonical key, so the same source keeps `[N]` even when re-cited later.
- **Auth via exception + event pause**: `AuthRequiredException` lets tools stay synchronous in the happy path; the orchestrator-level pause/resume mechanism keeps Chainlit-specific UI flow out of tool implementations.
- **`Tool.execute(args)` is single-argument**: there is no `ToolContext`. State a tool needs (history snapshot, credential store, callbacks) is injected at construction time by the composition root.
- **`ProcessEvent` as a flattened union**: a single `type` alias enumerates everything `process_message` may yield. Pyright flattens it, so `match`/`case` + `assert_never` gives exhaustive dispatch over all variants including future additions.
- **UI is Chainlit-first** for chat-focused UX and rapid conversational iteration.
- **CLI** remains available for ingestion (`uv run python -m src.ingest.cli reindex`) and evaluation workflows.
- **Multi-modal strategy** uses extraction-first: images are described into text by a vision model and indexed as `image_description` chunks alongside text chunks.
- **Ingestion architecture is hardened before PDF/image rollout**: converter routing, splitter strategies, injected embedder boundary, image cache, and micro-batching are introduced first to avoid compounding migration risk.
- **MCP** is intentionally excluded from MVP to reduce complexity.

## Subsystem Package Layering Conventions

Each subsystem package (`chatbot`, `ingest`) uses the same vertical layering. Cross-subsystem code lives in `src/shared/`.

| Layer | Purpose | Where |
|---|---|---|
| `contracts/` | Pure types and Protocols. No framework imports beyond Haystack value types (in ingest only, not in chatbot subsystem) and Pydantic v2 base classes. Imported by both `app/` and `infrastructure/`. | `src/chatbot/contracts/`, `src/ingest/contracts/` |
| `app/` | Use-case orchestration and application-level policy. Depends only on `contracts/` (and `src/shared/observability` for generic OTEL helpers). Never imports `infrastructure/`. | `src/chatbot/app/`, `src/ingest/app/` |
| `infrastructure/` | All concrete Protocol implementations (Ollama, OpenAI-compatible, Qdrant, Haystack converters, vision LLM, …). May depend on `contracts/` and `src/shared/`. | `src/chatbot/infrastructure/`, `src/ingest/infrastructure/` |
| `ui/` or `cli/` | Entry points + composition root. Parses input, instantiates infrastructure adapters, wires them into app-layer classes, dispatches work. | `src/chatbot/ui/`, `src/ingest/cli/` |
| `src/shared/` | Cross-subsystem concerns: settings, generic observability helpers, shared Qdrant infrastructure (both subsystems build the same `QdrantDocumentStore`). | `src/shared/settings/`, `src/shared/observability/`, `src/shared/qdrant/` |

Minimal rules:
- **`app/` never imports `infrastructure/`** in either subsystem. Use Protocols defined in `contracts/`.
- **No subsystem imports another subsystem**. Cross-subsystem code lives in `src/shared/`.
- **Composition lives only in `*/ui/composition.py` or `*/cli/composition.py`**: settings → config mapping and concrete adapter instantiation happen there.
- **Public API per `__init__.py`** in every sub-package. Internal modules use a `_` prefix (e.g. `_qdrant_hybrid.py`, `_ollama.py`).
- **No compatibility facades**: when paths change, imports are migrated directly.

## Module Boundaries
```
src/
├── shared/
│   ├── settings/                   # Settings + get_settings()
│   ├── observability/              # tracing.py, openinference.py (generic), logging.py
│   └── qdrant/                     # _config.py, _document_store.py, embeddings_sparse/
│
├── chatbot/
│   ├── contracts/
│   │   ├── chat.py                 # ChatMessage, ToolCallInfo, ChatStreamItem, ChatModel, ModelProfile, ThinkingContent
│   │   ├── tools.py                # Tool Protocol, ToolSchema
│   │   ├── citation.py             # Citation family, CiteableTool, CitableUnit, RawCitation, marker constants
│   │   ├── retrieval.py            # Retriever Protocol, SourceChunk
│   │   ├── credentials.py          # CredentialStore, UsernamePasswordCredentials, AuthRequiredException
│   │   ├── process.py              # ProcessEvent + variants
│   │   ├── observability.py        # SPAN_CHAT_* constants
│   │   ├── i18n.py                 # I18nMessage, JsonObject
│   │   └── prompts.py              # Prompts dataclass
│   ├── app/
│   │   ├── orchestrator.py
│   │   ├── chat_prompts.py         # DEFAULT_PROMPTS
│   │   ├── citation/               # CitationModel + parser + messages
│   │   └── credential_store.py     # InMemoryCredentialStore (default impl)
│   ├── infrastructure/
│   │   ├── chat/                   # Ollama, OpenAI-compatible, TextToolCallParsingWrapper, _model_profile
│   │   ├── embeddings_text/
│   │   ├── observability/          # chatbot-specific OpenInference attribute builders
│   │   ├── retrieval/              # QdrantHybridRetriever
│   │   └── tools/                  # RetrievalTool, VacationDaysTool + adapter
│   └── ui/
│       ├── app.py                  # Chainlit handlers (thin)
│       ├── composition.py          # settings → config mappers + composition root
│       ├── citation_view.py
│       └── i18n_messages.py
│
└── ingest/
    ├── contracts/
    │   ├── converters.py           # FileConverter Protocol, ImageDescriptionPayload
    │   └── images.py               # IMAGE_SUFFIXES, IMAGE_KIND_DESCRIPTION
    ├── app/
    │   ├── pipeline.py             # IngestionPipeline (receives pre-built FormatHandlers)
    │   └── vision_prompts.py       # build_image_description_prompt
    ├── infrastructure/
    │   ├── converters/             # implements FileConverter Protocol (text, markdown, pdf, image)
    │   ├── embeddings_document/    # write-side embedder
    │   ├── image_cache/            # ExtractedImageStore, ImageDescriptionCache
    │   ├── image_description.py    # ImageDescriptionService composite
    │   └── vision/                 # vision LLM adapter
    └── cli/
        ├── __main__.py             # argparse entry (`reindex`, `reset`)
        └── composition.py          # settings → config mappers + format handler builder
```

## Configuration
Use environment variables for:
- model names,
- Ollama endpoint,
- Qdrant host/port,
- corpus path,
- retrieval parameters (top-k, score thresholds),
- ingestion parameters (embedding dimension, split length, split overlap, batch sizing).

Runtime note:
- Per-user credentials for tool-protected services are exclusively session-scoped runtime state held in the `CredentialStore` and are not sourced from environment variables.

## Risk Register (Architecture)
- Local model quality variance may reduce answer quality and citation precision.
- PDF extraction quality can dominate downstream retrieval quality.
- Prompt injection in documents can affect generation behaviour.
- **Citation hallucination**: the model may emit citation markers with tokens not present in any tool result. `CitationModel` mitigates this by validating every `RawCitation.ref` against a global token index built from prior tool results and surfacing failures as `HallucinatedCitation` events.
- Latency can increase with large corpora if chunking/retrieval strategy is not tuned.
- Memory pressure can spike during ingestion if micro-batch limits are not enforced.
