# Architecture

## High-Level Design
The application is a locally-run RAG chatbot with explicit separation of concerns.

## Components
- UI Layer (Chainlit)
  - Handles user interaction, streaming output, and source display.
- Application Layer (Chat Orchestrator)
  - Coordinates retrieval, prompt construction, model generation, and optional tool calling.
- Retrieval Layer
  - Query embedding, vector search (Qdrant), reranking/filtering, source packaging.
- Ingestion Layer
  - File discovery, content extraction (multi-modal), chunking, embedding, indexing.
- Tool Integration Layer
  - `Tool` Protocol: each tool exposes a `schema: ToolSchema` field (name, description, parameters JSON Schema) and an `execute(args)` coroutine that returns a `dict[str, Any]`.  The orchestrator serialises this to JSON before appending it to message history.
  - `VacationDaysAuth`: a service-local Protocol used by `VacationDaysTool`. The default implementation is `InteractiveVacationDaysAuthSession`, which receives an `ask_user: AskUser` callable at construction time. On first call it collects one username/password pair from the user, caches it as instance state, and returns it. On adapter rejection it clears the cache so the next call re-collects.
  - Service adapters (e.g. `SimulatedVacationDaysAdapter`) are wrapped by tool classes; the orchestrator and the LLM never import adapters directly.
- Model Runtime
  - Ollama-hosted local models for generation and extraction support.

## Binding Points
- Binding points are documented at consumer boundaries (tool constructor and composition root), not via mandatory nominal inheritance on implementation classes.
- Vacation-days wiring in `on_chat_start`:
  - `SimulatedVacationDaysAdapter` is bound to `VacationDaysService`.
  - `InteractiveVacationDaysAuthSession` is bound to `VacationDaysAuth`.
- This keeps Protocol usage structural (PEP 544) while making runtime wiring explicit and discoverable.

## Data Flow
1. Ingestion flow:
   - Load corpus files (txt/md/pdf).
   - Extract normalized textual representation.
   - Chunk content with metadata (document id, section, page if available).
   - Generate embeddings and upsert vectors to Qdrant.
2. Query flow:
   - User message arrives via Chainlit.
   - Orchestrator sends message history and registered tool schemas to the LLM.
   - LLM either responds with text (streamed to user) or with one or more `tool_calls`.
   - For each `tool_call`, the orchestrator records the assistant tool-call request in history, then looks up the tool by name, calls `tool.execute(args)`, serialises the returned `dict` to JSON, and appends it to message history as a `role="tool"` message.
   - Loop continues until the LLM produces a text response with no pending tool calls.
   - For knowledge queries (Phase 3+): retrieval is integrated as additional context before generation.

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
- UI is Chainlit-first for chat-focused UX and rapid conversational iteration.
- CLI remains available for ingestion/evaluation workflows.
- Gradio fallback is documented to de-risk UI framework blockers. Switch trigger: Chainlit blocks a key user-facing feature for > 1 working days with no viable workaround.
- Multi-modal strategy uses extraction-first approach rather than shared embedding tricks.
- MCP is intentionally excluded from MVP to reduce complexity.

## Suggested Module Boundaries
- src/ui/
- src/app/
- src/retrieval/
- src/ingestion/
- src/tools/
  - Each tool that requires a service adapter is a sub-package: `src/tools/<name>/` with
    `service.py` (service Protocols, boundary request/response models, and service-level domain errors),
    `adapter.py` (concrete adapter implementation),
    `auth.py` (service-local auth Protocol and implementation, when needed),
    `tool.py` (Tool implementation), and
    `__init__.py` (re-exports the public surface used by the composition root).
  - Simple tools with no adapter may be a single `src/tools/<name>.py` file.
- src/config/
- tests/unit/
- tests/integration/

## Configuration
Use environment variables for:
- model names,
- Ollama endpoint,
- Qdrant host/port,
- corpus path,
- retrieval parameters (top-k, score thresholds).

Runtime note:
- Per-user credentials for the simulated external service are exclusively session-scoped runtime state handled by the orchestrator and are not sourced from environment variables.

## Risk Register (Architecture)
- Local model quality variance may reduce answer quality.
- PDF extraction quality can dominate downstream retrieval quality.
- Prompt injection in documents can affect generation behavior.
- Latency can increase with large corpora if chunking/retrieval strategy is not tuned.
