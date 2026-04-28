# Architecture

## High-Level Design
The application is a locally-run RAG chatbot with explicit separation of concerns.

## Components
- UI Layer (Chainlit)
  - Handles user interaction, streaming output, and source display.
- Application Layer (Chat Orchestrator)
  - Coordinates message history, the agentic tool-call loop, model generation, and citation extraction.
  - Retrieval is modelled as a regular `Tool` (`search_documents`): the LLM decides if and when to call it and formulates the query itself — the orchestrator applies no pre-retrieval heuristics.
  - The system prompt instructs the model to restrict answers to information available from tools and retrieved documents, and to express uncertainty when evidence is insufficient rather than drawing on parametric knowledge.
  - After the agentic loop produces a plain-text response, the orchestrator runs a citation pass: it checks whether a `SourceCitationEvent` was already emitted during the loop (via `cite_sources`). If not and at least one `search_documents` result exists in the full conversation history, it re-runs the same agentic loop code with only `cite_sources` registered as a tool, discarding any text output. If the model still produces no citation call, no `SourceCitationEvent` is emitted and the UI shows no sources.
  - `process_message` yields a typed `ProcessEvent` stream (`type ProcessEvent = str | ToolEvent`); text chunks are streamed live, `ToolEvent` instances (e.g. `SourceCitationEvent`) are emitted as they occur during tool execution.
  - All prompts are centralised in `src/chatbot/app/prompts.py` as `@dataclass(frozen=True) class Prompts`. The orchestrator receives a `Prompts` instance via constructor injection.
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
  - `CitationTool` (`cite_sources`): a regular tool that receives the cited filenames from the LLM, validates them against the `search_documents` tool-results in the `ToolContext` history snapshot, and emits a `SourceCitationEvent` carrying only the validated `SourceChunk` objects. Its `JsonObject` result (containing validated and unvalidated filenames) is appended to history like any other tool result, giving subsequent turns full citation context.
  - `RetrievalTool` (`search_documents`): wraps the `Retriever` Protocol. Returns retrieved chunks as structured JSON; emits no `ToolEvent`.
  - `VacationDaysAuth`: a service-local Protocol used by `VacationDaysTool`. The default implementation is `InteractiveVacationDaysAuthSession`, which receives an `ask_user: AskUser` callable at construction time. On first call it collects one username/password pair from the user, caches it as instance state, and returns it. On adapter rejection it clears the cache so the next call re-collects.
  - Service adapters (e.g. `SimulatedVacationDaysAdapter`) are wrapped by tool classes; the orchestrator and the LLM never import adapters directly.
- Model Runtime
  - Ollama-hosted local models for generation in MVP.
  - Ingestion embedder is consumed through an injected boundary so provider/runtime can be swapped without changing ingestion orchestration.

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
     - **Trigger condition**: only if the full conversation history contains at least one `search_documents` tool-result (i.e. RAG was ever used, including in prior turns). If no `search_documents`-result is present, skip the citation pass entirely.
     1. If a `SourceCitationEvent` was emitted during the loop (i.e. the LLM called `cite_sources`): no further action needed.
     2. If no `SourceCitationEvent` was observed: re-run the same agentic loop code with only `cite_sources` registered as a tool and the complete history; discard any text output. Tool dispatch is identical to the main loop.
     3. If still no citations returned: emit nothing — the UI displays no sources. This is **not an error**: it is the expected outcome when the answer was derived from a non-RAG tool call (e.g. `get_vacation_days`) even though RAG results exist elsewhere in history.
   - `process_message` yields `ProcessEvent` (`str | ToolEvent`); the UI uses `match`/`case` + `assert_never` to handle each variant (render text chunks live, render `SourceCitationEvent` as Chainlit citation elements).

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
- Citations are LLM-driven via `cite_sources`, a regular tool that validates cited filenames against actual `search_documents` results in the `ToolContext` history snapshot and emits a `SourceCitationEvent`. The fallback citation pass (same agentic loop code, only `cite_sources` registered, text discarded) is triggered only when the full history contains at least one `search_documents` result. No citations returned after the fallback is not an error — it is the expected outcome when the answer was derived from a non-RAG tool.
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
    - `prompts.py`: `@dataclass(frozen=True) class Prompts` with `str` fields for plain prompts and `Callable[..., str]` fields for parameterised prompts. Module-level `DEFAULT_PROMPTS` constant provides production defaults. Callers customise via `dataclasses.replace(DEFAULT_PROMPTS, field=value)` — no subclassing.
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
- Citation hallucination: the fallback citation pass may cause the model to invent source filenames not present in actual `search_documents` results. `CitationTool` mitigates this by validating all claimed sources against the tool-result history and discarding unvalidated ones.
- Latency can increase with large corpora if chunking/retrieval strategy is not tuned.
- Memory pressure can spike during ingestion if micro-batch limits are not enforced.
