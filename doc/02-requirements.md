# Requirements

## Functional Requirements

### FR-01 Chat Experience
- The system shall provide a conversational chat interface in Chainlit.
- The system shall stream partial model output to the user.
- The system shall persist conversation context for the active session.
- The UI shall render validated citations both (a) as side-panel citation elements and (b) as a compact deduplicated source list appended to the assistant response.
- Where `source_url` exists, the displayed citation title/label shall be rendered as a link; raw URLs shall not be shown as standalone text lines.

### FR-02 Retrieval-Augmented Generation
- The system shall retrieve relevant chunks from indexed content by exposing retrieval as a tool (`search_documents`) that the LLM invokes with its own query formulation.
- The LLM shall explicitly declare which sources it used by calling the `cite_sources` tool with `source` + `chunk_id` pairs.
- `cite_sources` validates declared `source` + `chunk_id` pairs against the `search_documents` results present in tool-result history and emits a `SourceCitationEvent` carrying only the validated `SourceChunk` objects.
- If the LLM does not call `cite_sources` in its primary response **and** at least one `search_documents` call occurred in the same turn, the orchestrator shall run a dedicated citation pass with only `cite_sources` registered as a tool. The citation pass prompt shall contain only (a) rendered search results from that turn and (b) the final answer text. If no correlated `search_documents` result exists, the citation pass is skipped.
- The citation pass shall use a dedicated citation system prompt distinct from the main answer-generation system prompt.
- During the citation pass, if the model returns no native tool call but emits a serialized `cite_sources` JSON payload in text, the system shall attempt to recover and dispatch it as a `cite_sources` call.
- If the model returns no citations after the fallback call, no sources shall be displayed — this is not an error (the answer may have been derived from a non-RAG tool call). The system shall not fabricate or guess cited sources.
- The system prompt shall instruct the model to restrict answers to information available from tools and retrieved documents, and to express uncertainty when evidence is insufficient rather than drawing on parametric knowledge.
- Serialized `search_documents` tool results written to history shall preserve citation metadata fields (`title`, `author`, `publication_date`, `source_url`) so downstream citation rendering has full provenance context.
- The system shall use a retrieval strategy configurable via constants/env for top-k and similarity threshold.

### FR-03 Multi-Modal Ingestion
- The system shall ingest txt, md, and pdf files from a configured corpus path.
- The system shall route files to type-specific converters (at minimum: txt -> text converter, md -> markdown converter).
- The system shall support document-character-aware chunking strategies instead of enforcing one splitter for all inputs.
- The system shall use an injected embedder boundary for ingestion, not a hard dependency on a single provider implementation.
- The system shall process ingestion in bounded micro-batches to avoid unbounded in-memory growth.
- The system shall preserve metadata continuity across converter -> splitter -> embedder -> writer so citation metadata is available at retrieval time.
- The system shall support sidecar metadata files (`<document>.meta.json`): sidecar files themselves must not be ingested as standalone documents; their fields must be merged into the corresponding document metadata.
- The system shall support re-ingestion without creating uncontrolled duplicate vectors.

### FR-04 Tool Calling
- The system shall expose typed tools in MVP: vacation days lookup, document retrieval (`search_documents`), and citation declaration (`cite_sources`).
- All tool input/output schemas shall be defined with Pydantic models.
- The LLM shall decide when to invoke a tool based on its declared schema and description; the orchestrator must not apply keyword heuristics or classify intent itself.
- The orchestrator shall run an agentic loop: send message history + tool schemas to the LLM, execute any returned `tool_calls`, append `role="tool"` results to history, and repeat until the LLM produces a plain-text response.
- `Tool.execute` shall accept a `ToolContext` (containing a read-only snapshot of the current conversation history) and return a `tuple[JsonObject, list[ToolEvent]]`. The `JsonObject` is serialised into history as the `role="tool"` result; the `ToolEvent` list is yielded into the outer `process_message` stream after the tool result is recorded. Most tools return an empty event list.
- `cite_sources` is a regular tool: its `JsonObject` result (containing validated and unvalidated sources) is appended to history like any other tool, and it emits a `SourceCitationEvent` in its `ToolEvent` list for the UI to render.
- `ToolContext` provides a read-only history snapshot; tools must not mutate it.
- Tool implementations must not import UI or infrastructure modules; all session-specific dependencies (such as the `ask_user` callable) are injected at construction time.
- Tool input models shall tolerate common weak-model argument mistakes by coercing JSON-serialized list/object fields into native JSON values before schema validation.

### FR-05 Authentication for External Service Simulation
- The external service simulation shall require simple username/password identity context at tool-call time.
- The system shall support session-scoped credentials for the external service without requiring global Chainlit UI login in MVP.
- The vacation-days tool shall use a dedicated session-scoped username/password auth collaborator; it receives an `ask_user` callable at construction time, collects credentials on first use, caches them as instance state, and clears them on authentication failure.
- Credentials must never be passed as LLM-generated tool arguments; they are managed exclusively by the tool's injected auth collaborator via its own instance state (the auth object is constructed once per session in `on_chat_start`).
- Failed authentication shall be surfaced as a user-safe error message returned by the tool to the LLM.
- Optional app-level Chainlit authentication is out of scope for MVP and may be added later.

### FR-06 Developer Operations
- The project shall provide CLI commands/scripts for:
  - corpus ingestion,
  - reindex/reset as needed,
  - running evaluation set.
- The project shall provide optional OpenTelemetry tracing export configurable via environment variables.
- The project shall document how developers run a local trace backend (Jaeger) and inspect traces.

## Non-Functional Requirements

### NFR-01 Reliability
- The system should avoid unsupported claims and respond with uncertainty when evidence is insufficient.
- Retrieval and generation failures shall return graceful user-facing errors.

### NFR-02 Performance
- p95 response time target: <= 10s for normal local queries.
- Ingestion should process at least 100 medium-sized docs without manual intervention.
- Ingestion shall use bounded-memory processing (micro-batches) so peak RAM grows with batch size, not total corpus size.

### NFR-03 Maintainability
- Core concerns shall be separated into modules:
  - UI,
  - orchestration,
  - retrieval,
  - ingestion,
  - tool integrations.
- Typed boundaries shall be used between modules.
- Ingestion boundaries shall separate conversion routing, splitting strategy selection, embedding, and persistence to keep each stage independently replaceable and testable.
- All prompts shall be centralised in `src/chatbot/app/prompts.py` as a `@dataclass(frozen=True) class Prompts` with callable fields (for request-time context such as current date and citation payloads). A module-level `DEFAULT_PROMPTS` constant provides the production defaults. The orchestrator derives the effective prompts from `DEFAULT_PROMPTS` via an injected model-specific `PromptProfile`; callers customise prompts through the profile implementation (typically via `adjust_prompts(...)`) rather than by directly injecting a `Prompts` instance into the orchestrator.
- Observability instrumentation shall be implemented via OpenTelemetry standards rather than ad-hoc custom tracing formats.

### NFR-04 Testability
- Unit tests are required for retrieval orchestration and tool-calling decision logic.
- Integration tests are required for end-to-end question answering with a small fixture corpus.

### NFR-05 Security (MVP Level)
- No secrets hardcoded in source files.
- Environment variables shall be used for credentials and runtime config.
- Prompt-injection resilience shall be partially addressed through instruction policy and source-grounded answering.

### NFR-06 Code Quality
- Code shall be idiomatic, clean Python — no copy-paste duplication, no spaghetti logic.
- YAGNI: no speculative abstractions or features beyond what a current requirement explicitly demands.
- KISS: prefer the simplest solution that satisfies the requirement; complexity must be justified.
- DRY: shared logic shall be extracted into well-named, single-responsibility helpers; duplication is not acceptable.
- All modules shall have clear, single responsibilities with explicit typed boundaries between them.
- Functions and classes shall be small and focused; oversized components indicate a design problem and must be refactored.
- Core orchestration logic shall not import directly from infrastructure modules (Qdrant client, HTTP clients). Retrieval and tool adapter boundaries shall use `Protocol`-based interfaces to enable testability without a full hexagonal structure.
- Dependencies shall be injected via constructor parameters (manual constructor injection, no DI framework). Components must not instantiate their own infrastructure dependencies internally. Composition shall happen in a dedicated factory function invoked at application startup (e.g. in the Chainlit session lifecycle hook).
- Test coverage is required for all non-trivial business logic; tests shall be readable and intention-revealing.
- All code shall be fully type-annotated; use of `Any` must be explicitly justified and minimized.
- Static type checking via `pyright` in strict mode is required; the codebase must pass without errors.
- Use modern Python 3.10+ union syntax (`X | None` over `Optional[X]`, `X | Y` over `Union[X, Y]`).
- Use `Protocol` for structural typing at module boundaries instead of concrete base classes where appropriate.
- Use `@dataclass(frozen=True)` or `NamedTuple` for immutable value objects.
- Use Pydantic models at system boundaries (tool schemas, external service responses, environment config via `pydantic-settings`); use `@dataclass(frozen=True)` for internal value objects between own modules. Do not use Pydantic internally where there is no external validation need.
- Project dependency and environment management shall use `uv` as the single standard tool (`uv.lock` committed, reproducible installs via `uv sync`).
- Linting and formatting shall use `ruff` as the single standard toolchain (`ruff check`, `ruff format`).
- Static typing checks shall run via `pyright` with strict configuration enabled in `pyproject.toml` (`typeCheckingMode = "strict"`) in local validation and CI.
- Structured logging shall use `structlog`; log events shall be key-value structured (not interpolated strings). Rendered as human-readable console output in development and as JSON to stdout in production/CI.

## Acceptance Criteria
- Given a known answer in corpus, the bot returns an answer with relevant citations.
- Given no relevant corpus evidence, the bot states uncertainty and does not fabricate references.
- Given a vacation-days query, the bot performs the typed tool call and returns structured result.
- Given a mixed txt/md corpus, ingestion uses type-specific converters and completes successfully.
- Given a document with `<document>.meta.json`, retrieved chunks include sidecar metadata fields required for citation rendering.
- Given retrieved chunks with `source_url`, citation titles/labels are rendered as links and raw URLs are not shown as standalone lines.
- CI test suite passes on a fresh environment setup.
