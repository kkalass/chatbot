# Architecture

## High-Level Design
The application is a locally-run RAG chatbot with explicit separation of concerns.

## Components
- UI Layer (Chainlit)
  - Handles user interaction, streaming output, and source display.
  - Renders validated citations in two surfaces: side-panel citation elements and a compact deduplicated "Sources" section appended to the answer.
- Application Layer (Chat Orchestrator)
  - Coordinates message history, the agentic tool-call loop, model generation, and citation extraction.
  - Retrieval is modelled as a regular `Tool` (`search_documents`): the LLM decides if and when to call it and formulates the query itself — the orchestrator applies no pre-retrieval heuristics.
  - The system prompt instructs the model to restrict answers to information available from tools and retrieved documents, and to express uncertainty when evidence is insufficient rather than drawing on parametric knowledge.
  - After the agentic loop produces a plain-text response, the orchestrator runs a citation pass only for turns that executed `search_documents`. If no `SourceCitationEvent` was emitted during the main loop, it builds a dedicated citation request from (a) rendered search results correlated to that turn's `search_documents` call IDs and (b) the final answer text, then invokes the model with only `cite_sources` registered as a tool.
  - The citation pass uses a dedicated citation system prompt (separate from the main answer-generation system prompt) to reduce instruction interference.
  - If the model returns text instead of a native tool call in citation pass, the orchestrator attempts to recover a serialized `cite_sources` JSON payload from text and dispatch it as a normal tool call.
  - `process_message` yields a typed `ProcessEvent` stream (`type ProcessEvent = str | ToolEvent`); text chunks are streamed live, `ToolEvent` instances (e.g. `SourceCitationEvent`) are emitted as they occur during tool execution.
  - All prompts are centralised in `src/chatbot/app/prompts.py` as `@dataclass(frozen=True) class Prompts`. The orchestrator receives a model-specific `PromptProfile` via constructor injection and derives effective prompts by applying it to `DEFAULT_PROMPTS`.
- Retrieval Layer
  - Query embedding, vector search (Qdrant), reranking/filtering, source packaging.
- Ingestion Layer
  - File discovery, converter routing by input type, character-aware chunking strategy selection, embedding, indexing.
  - Sidecar metadata merge (`<document>.meta.json`) at conversion time; sidecars are not treated as standalone documents.
  - Micro-batch orchestration for bounded-memory ingestion.
- Tool Integration Layer
  - `Tool` Protocol: each tool exposes a `schema: ToolSchema` field and an `execute(args: JsonObject, context: ToolContext) -> tuple[JsonObject, list[ToolEvent]]` coroutine. The orchestrator appends the `JsonObject` to history as the `role="tool"` result and yields any `ToolEvent` items into the outer `process_message` stream.
  - `ToolContext`: a read-only value object carrying a snapshot of the current conversation history (`history: tuple[ChatMessage, ...]`). Passed to every tool on execution; tools must not mutate it.
  - `ToolEvent`: a `type` alias union in `src/chatbot/app/protocols.py` (e.g. `type ToolEvent = SourceCitationEvent`). No base class or Protocol — the union itself enumerates what tools may emit. `type ProcessEvent = str | ToolEvent` is the public stream type of `process_message`; pyright flattens both unions, so `match`/`case` + `typing.assert_never` on `ProcessEvent` gives exhaustive dispatch over all variants including future additions. Additional event types may be added in future phases by extending `ToolEvent`.
  - `CitationTool` (`cite_sources`): a regular tool that receives cited `source` + `chunk_id` pairs from the LLM, validates them against the `search_documents` tool-results in the `ToolContext` history snapshot, and emits a `SourceCitationEvent` carrying only the validated `SourceChunk` objects. Its `JsonObject` result (containing validated and unvalidated pairs) is appended to history like any other tool result, giving subsequent turns full citation context.
  - `RetrievalTool` (`search_documents`): wraps the `Retriever` Protocol. Returns retrieved chunks as structured JSON; emits no `ToolEvent`.
  - `ToolInputModel`: shared Pydantic base model for LLM-facing tool arguments. Structured fields (`list`/`dict`) are pre-coerced from JSON-serialized strings before validation.
  - `VacationDaysAuth`: a service-local Protocol used by `VacationDaysTool`. The default implementation is `InteractiveVacationDaysAuthSession`, which receives an `ask_user: AskUser` callable at construction time. On first call it collects one username/password pair from the user, caches it as instance state, and returns it. On adapter rejection it clears the cache so the next call re-collects.
  - Service adapters (e.g. `SimulatedVacationDaysAdapter`) are wrapped by tool classes; the orchestrator and the LLM never import adapters directly.
- Model Runtime
  - Ollama-hosted local models for generation in MVP.
  - Ingestion embedder is consumed through an injected boundary so provider/runtime can be swapped without changing ingestion orchestration.
- Observability Layer
  - OpenTelemetry tracing is configured at app startup via `arize-phoenix-otel`'s `register()`, which creates a `TracerProvider` that exports spans to a Phoenix-compatible OTLP endpoint (default: `http://localhost:6006/v1/traces`).
  - Spans are mapped to **OpenInference semantic conventions** using the helper layer in `src/chatbot/observability/openinference.py`, which wraps `openinference.instrumentation` builders and `openinference.semconv.trace` constants. Span kinds used: `CHAIN` (UI turn, orchestrator round, tool dispatch, citation pass), `LLM` (Ollama streaming), `RETRIEVER` (Qdrant query), `TOOL` (retrieval and citation tools).
  - **Session propagation**: `using_session_attributes(session_id)` from `openinference.instrumentation` is used in the UI `on_message` handler to propagate the session context to all child spans in a turn.
  - Optional **Haystack auto-instrumentation** is enabled by default (`OTEL_AUTO_INSTRUMENT_HAYSTACK=true`) using `openinference-instrumentation-haystack`.
  - All spans carry both OpenInference attributes (`input.value`, `output.value`, `llm.model_name`, `llm.input_messages`, etc.) and project-specific diagnostic attributes (`chat.round.*`, `llm.request.*`, `chat.retriever.*`).
  - Citation-pass spans capture recovery telemetry (`recovery_attempted`, `recovery_succeeded`), while the UI turn span carries evaluation metadata and emitted-response summaries.
  - Tracing-only payload construction is kept out of business logic: request/response/error attribute assembly lives in module-level `_trace_*` helper functions, and control-flow code only invokes those helpers.
  - Trace payload attributes are previewed with truncation to keep spans inspectable while avoiding unbounded attribute size growth.
  - **Local Phoenix setup**: run `uv run python -m phoenix.server.main serve` (or `pip install arize-phoenix && phoenix serve`) to start the Phoenix UI at `http://localhost:6006`. Traces are visible in the Traces view under the project name `chatbot-local` (configurable via `PHOENIX_PROJECT_NAME`).

## Binding Points
- Binding points are documented at consumer boundaries (tool constructor and composition root), not via mandatory nominal inheritance on implementation classes.
- Vacation-days wiring in `on_chat_start`:
  - `SimulatedVacationDaysAdapter` is bound to `VacationDaysService`.
  - `InteractiveVacationDaysAuthSession` is bound to `VacationDaysAuth`.
- This keeps Protocol usage structural (PEP 544) while making runtime wiring explicit and discoverable.

## Data Flow
1. Ingestion flow:
   - Load corpus files (txt/md/pdf).
  - Route each file to a type-specific converter (txt/text, md/markdown, pdf/extractor).
  - Merge sidecar metadata (`<document>.meta.json`) into document metadata when present.
  - Select splitter strategy based on document type/character and create chunks.
  - Process chunks in micro-batches: embed and write each batch before loading the next.
  - Upsert vectors to Qdrant with metadata preserved for citation.
2. Query flow:
   - User message arrives via Chainlit.
   - Orchestrator sends message history and registered tool schemas (including `search_documents` and `cite_sources`) to the LLM.
   - LLM either responds with text (streamed to user) or with one or more `tool_calls`.
   - For each `tool_call`: the orchestrator calls `tool.execute(args, context)` where `context` carries a read-only history snapshot. The returned `JsonObject` is appended to history as `role="tool"`; any `ToolEvent` items are yielded into the `process_message` stream immediately.
   - Loop continues until the LLM produces a plain-text response with no pending tool calls.
   - Citation pass after the loop:
     - **Trigger condition**: only if at least one `search_documents` call happened in the current turn and no `SourceCitationEvent` was already emitted.
     1. Collect `search_documents` tool-result chunks correlated by call ID from the current turn.
     2. Build a dedicated citation request prompt containing `<search_results>` and `<answer>` blocks.
    3. Call the model with the citation-specific system prompt and only `cite_sources` in the tool schema list.
     4. If the model emits no native tool call, attempt serialized-tool-call recovery by parsing text for a `cite_sources` JSON payload.
     5. Dispatch at most one `cite_sources` call and emit resulting `SourceCitationEvent` values.
     6. If no citations are recovered/returned, emit nothing — the UI displays no sources.
  - `process_message` yields `ProcessEvent` (`str | ToolEvent`); the UI uses `match`/`case` + `assert_never` to handle each variant (render text chunks live, render `SourceCitationEvent` as Chainlit citation elements plus appended deduplicated source markdown).
  - In parallel, OpenTelemetry spans record request/response previews and counters for each stage to enable deterministic debugging in Phoenix.

## Authentication Flows (MVP)
- The Chainlit UI is accessible without mandatory global app-level login in MVP.
- Authentication is required only for the external service simulation and happens at tool-call time.
- Credentials are scoped to the active chat session and not persisted as long-term application user accounts.
- Tool-call authentication failures are returned as user-safe errors without exposing sensitive details.
- Optional app-level Chainlit authentication can be added later as a separate access-control concern.

## Auth-Protected Tool Sequence (MVP)
Credential collection is handled inside the tool via `VacationDaysAuth` — the LLM is not involved in the auth flow.

1. LLM calls `get_vacation_days(year=...)` — no credentials in arguments.
2. `VacationDaysTool.execute` delegates to `VacationDaysAuth.get_credentials()`.
3. `InteractiveVacationDaysAuthSession` checks its internal cached credentials field:
   - If present: returns cached credentials immediately.
   - If absent: calls the injected `ask_user(...)` callable twice (username, then password), caches the pair, and returns them. If the user cancels either prompt, returns `None` and the tool returns a user-safe cancellation message to the LLM.
4. The tool calls the service adapter with the credentials.
5. On `ToolAuthenticationError`: `VacationDaysAuth.clear_credentials()` resets the cache and the tool returns a user-safe auth-failure string to the LLM. The LLM informs the user; on the next vacation-days request the collection path runs again.

Implementation rules:
- Credentials must never be sourced from model-generated tool arguments or retrieval context; they are managed exclusively by the injected `VacationDaysAuth` implementation via its `ask_user` callable and internal cache.
- Tools must not import `chainlit` or any UI module directly; all UI interaction is mediated through the `ask_user: AskUser` callable injected at construction time.
- The auth collaborator is not a tool and is never registered in the tool list exposed to the LLM.

## Key Design Decisions
- Retrieval is modelled as a tool (`search_documents`), not as eager pre-retrieval: the LLM decides when to search and formulates its own queries, enabling multi-hop retrieval and query reformulation from conversation context.
- Citations are LLM-driven via `cite_sources`, a regular tool that validates cited `source` + `chunk_id` pairs against actual `search_documents` results in the `ToolContext` history snapshot and emits a `SourceCitationEvent`.
- The fallback citation pass is isolated from full history and instead uses a dedicated prompt with current-turn search results plus final answer text; this reduces tool-selection drift in weaker models.
- Serialized citation tool-call recovery (`{"name":"cite_sources","parameters":...}` emitted as text) is supported as a robustness fallback.
- Citation presentation is metadata-first: title/author/date are preferred for display; where `source_url` exists, labels are linkified and standalone raw URL lines are omitted.
- `Tool.execute` accepts a `ToolContext` and returns `tuple[JsonObject, list[ToolEvent]]`: the `JsonObject` enters history as the tool result; `ToolEvent` items flow directly into the `process_message` stream. This keeps the orchestrator free of per-tool special cases while giving tools a clean channel to emit typed events to the UI.
- Two `type` aliases separate concerns: `ToolEvent` enumerates what tools may emit; `ProcessEvent = str | ToolEvent` is the public stream type of `process_message`. Pyright flattens both unions, so `match`/`case` + `assert_never` on `ProcessEvent` gives exhaustive, statically-checked dispatch over all variants — the Python equivalent of Dart's sealed class pattern.
- UI is Chainlit-first for chat-focused UX and rapid conversational iteration.
- CLI remains available for ingestion/evaluation workflows.
- Gradio fallback is documented to de-risk UI framework blockers. Switch trigger: Chainlit blocks a key user-facing feature for > 1 working days with no viable workaround.
- Multi-modal strategy uses extraction-first approach rather than shared embedding tricks.
- Ingestion architecture is hardened before PDF rollout: converter routing, splitter strategies, injected embedder boundary, and micro-batching are introduced first to avoid compounding migration risk.
- MCP is intentionally excluded from MVP to reduce complexity.

## Suggested Module Boundaries
- src/settings/
- src/chatbot/
  - `app/`: orchestrator, protocols, prompts
    - `prompts.py`: `@dataclass(frozen=True) class Prompts` with callable prompt fields (request-time system prompts and parameterised citation prompt builders). Module-level `DEFAULT_PROMPTS` constant provides production defaults. Callers customise via `dataclasses.replace(DEFAULT_PROMPTS, field=value)` — no subclassing.
  - `ui/`: Chainlit UI layer and session lifecycle
  - `tools/`: typed tool schemas and external-service adapters
    - Each tool that requires a service adapter is a sub-package: `src/chatbot/tools/<name>/` with
      `service.py` (service Protocols, boundary request/response models, and service-level domain errors),
      `adapter.py` (concrete adapter implementation),
      `auth.py` (service-local auth Protocol and implementation, when needed),
      `tool.py` (Tool implementation), and
      `__init__.py` (re-exports the public surface used by the composition root).
    - Simple tools with no adapter may be a single `src/chatbot/tools/<name>.py` file.
  - `infrastructure/`: Ollama/Qdrant adapters for chatbot concerns
    - `chat/`, `embeddings_text/`, `retrieval/`
  - `config.py`: Settings → chatbot infrastructure config converters
- src/ingest/
  - `pipeline.py`: file discovery, converter routing, chunking, embedding, indexing
  - `cli.py`: developer CLI (`reindex`, `reset`)
  - `infrastructure/`: Haystack/Qdrant adapters for ingestion concerns
    - `embeddings_document/`, `document_store/`
  - `config.py`: Settings → ingest infrastructure config converters
- tests/unit/
- tests/integration/

## Configuration
Use environment variables for:
- model names,
- Ollama endpoint,
- Qdrant host/port,
- corpus path,
- retrieval parameters (top-k, score thresholds),
- ingestion parameters (embedding dimension, split length, split overlap, batch sizing).

Runtime note:
- Per-user credentials for the simulated external service are exclusively session-scoped runtime state handled by the orchestrator and are not sourced from environment variables.

## Risk Register (Architecture)
- Local model quality variance may reduce answer quality.
- PDF extraction quality can dominate downstream retrieval quality.
- Prompt injection in documents can affect generation behavior.
- Citation hallucination: the citation pass may cause the model to invent source pairs not present in actual `search_documents` results. `CitationTool` mitigates this by validating all claimed `source` + `chunk_id` pairs against tool-result history and discarding unvalidated ones.
- Latency can increase with large corpora if chunking/retrieval strategy is not tuned.
- Memory pressure can spike during ingestion if micro-batch limits are not enforced.
