# Requirements

## Functional Requirements

### FR-01 Chat Experience
- The system shall provide a conversational chat interface in Chainlit.
- The system shall stream partial model output to the user.
- The system shall persist conversation context for the active session.

### FR-02 Retrieval-Augmented Generation
- The system shall retrieve relevant chunks from indexed content for each question.
- The system shall include source references in responses.
- The system shall use a retrieval strategy configurable via constants/env for top-k and similarity threshold.

### FR-03 Multi-Modal Ingestion
- The system shall ingest txt, md, and pdf files from a configured corpus path.
- The system shall extract content via a document content extraction approach for robust multi-modal handling.
- The system shall chunk, embed, and index extracted content into Qdrant.
- The system shall support re-ingestion without creating uncontrolled duplicate vectors.

### FR-04 Tool Calling
- The system shall expose exactly one typed tool in MVP: vacation days lookup.
- The tool input/output schema shall be defined with Pydantic models.
- The chatbot shall call the tool only when user intent matches tool semantics.
- The chatbot shall implement a two-stage execution for auth-protected tools: intent detection/precheck, then tool execution after auth context is available.

### FR-05 Authentication for External Service Simulation
- The external service simulation shall require simple username/password identity context at tool-call time.
- The system shall support session-scoped credentials for the external service without requiring global Chainlit UI login in MVP.
- The orchestrator shall inject auth context from server-side session state into the tool adapter and must not rely on model-generated credentials.
- Failed authentication shall be surfaced as a user-safe error message.
- Optional app-level Chainlit authentication is out of scope for MVP and may be added later.

### FR-06 Developer Operations
- The project shall provide CLI commands/scripts for:
  - corpus ingestion,
  - reindex/reset as needed,
  - running evaluation set.

## Non-Functional Requirements

### NFR-01 Reliability
- The system should avoid unsupported claims and respond with uncertainty when evidence is insufficient.
- Retrieval and generation failures shall return graceful user-facing errors.

### NFR-02 Performance
- p95 response time target: <= 10s for normal local queries.
- Ingestion should process at least 100 medium-sized docs without manual intervention.

### NFR-03 Maintainability
- Core concerns shall be separated into modules:
  - UI,
  - orchestration,
  - retrieval,
  - ingestion,
  - tool integrations.
- Typed boundaries shall be used between modules.

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
- CI test suite passes on a fresh environment setup.
