# Delivery Plan

## Phase 0: Setup and Foundations
- Standardize on `uv` for Python version, virtual environment, dependency, and lockfile management.
- Pin project Python version (3.12) and commit `.python-version`.
- Initialize and commit `pyproject.toml` and `uv.lock` for reproducible installs.
- Define baseline developer commands: `uv sync`, `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`.
- Add CI quality gate running `ruff check` and `pyright` on pull requests.
- Define folder layout and configuration strategy (.env + typed settings).
- Write a comprehensive README covering everything a developer needs to get started:
  - Prerequisites (Python version, uv, Ollama, Docker).
  - Step-by-step setup: clone, `uv sync`, `.env` configuration.
  - How to start local dependencies (Qdrant via Docker, Ollama with required models).
  - How to run the app (`uv run chainlit run ...`).
  - How to run ingestion, reindex, and evaluation CLI commands.
  - How to run the test suite (`uv run pytest`).
  - How to run linting and type checks (`uv run ruff check .`, `uv run pyright`).
  - Known limitations and out-of-scope items.
  - The README is the authoritative onboarding document; it must be kept up to date as the project evolves.

## Phase 1: Baseline Chat
- Implement minimal Chainlit app connected to local Ollama model.
- Add streaming response support.
- Validate end-to-end local chat loop.

## Phase 2: Tool Calling
- Implement `Tool` Protocol and `ToolSchema` dataclass as the typed boundary between the orchestrator and tools.
- Implement agentic tool loop in the orchestrator: send tool schemas to the LLM, execute returned `tool_calls` by dispatching to the registered tool by name, loop until plain-text response.
- Introduce `src/chatbot/app/prompts.py`: `@dataclass(frozen=True) class Prompts` with `str` fields for plain prompts and `Callable[..., str]` fields for parameterised prompts; module-level `DEFAULT_PROMPTS` constant. The orchestrator derives its effective `Prompts` from `DEFAULT_PROMPTS` via an injected `PromptProfile` at construction time. All future prompts (system prompt, citation fallback message, etc.) live here only.
- Implement `VacationDaysAuth` and `InteractiveVacationDaysAuthSession`: receive `ask_user: AskUser` at construction, collect one username/password pair on first use, cache it as instance state, and clear it on auth failure.
- Implement typed vacation-days tool with Pydantic schemas; delegates auth to `VacationDaysAuth`.
- Wire all session-scoped dependencies (model, tools, vacation-days auth session) in `app.py` `on_chat_start`; inject `ask_user` wrapper there as the single Chainlit seam.
- Add unit tests for vacation-days auth, tool execution paths, and agentic loop behaviour.

## Phase 3: Text RAG
- Implement ingestion for txt/md.
- Implement explicit CLI workflows for `reindex` and `reset` as required by FR-06.
- Add chunking, embedding, and Qdrant indexing.
- Integrate retrieval as a tool (`search_documents`) so the LLM decides when and with what query to retrieve.
- Introduce `ToolContext` (read-only history snapshot) and change `Tool.execute` signature to `(args, context) -> tuple[JsonObject, list[ToolEvent]]`. Update all existing tools (`VacationDaysTool`, `RetrievalTool`) accordingly.
- Introduce `ToolEvent` type alias union and `ProcessEvent = str | ToolEvent` in `src/chatbot/app/protocols.py`; add `SourceCitationEvent` as the first `ToolEvent` variant.
- Implement `CitationTool` (`cite_sources`) as a regular tool: validates cited filenames against `search_documents` results in `ToolContext.history`, emits `SourceCitationEvent` in its `ToolEvent` list, and appends its `JsonObject` result to history.
- Update the orchestrator to propagate `ToolEvent` items from tool execution into the `process_message` stream. After the agentic loop, trigger the fallback citation pass only if the full conversation history contains at least one `search_documents` tool-result; if so, re-run the same agentic loop code with only `cite_sources` registered as a tool, discarding text output. No citations returned after the fallback is not an error.
- Update `process_message` to yield `ProcessEvent` (`str | ToolEvent`); update the UI to use `match`/`case` + `assert_never` and render `SourceCitationEvent` as Chainlit citation elements.
- Add integration tests for grounded QA and citation validation.

## Phase 3.5: Ingestion Architecture Hardening (Pre-PDF)
- Introduce type-specific converter routing (minimum: txt -> text converter, md -> markdown converter).
- Introduce splitter strategy selection by document type/character instead of a single global splitter policy.
- Replace hard provider coupling in ingestion by an injected embedder boundary at composition time.
- Introduce bounded micro-batch processing for ingestion by file count (RAM-bound control): `ingest` accepts iterable paths and processes them in bounded file batches; `ingest_corpus` remains the discovery wrapper that delegates to `ingest`.
- Formalize metadata continuity contract, including sidecar merge semantics (`<document>.meta.json` is merged into the owning document and never ingested standalone).
- Add unit and integration coverage for converter routing, splitter selection, metadata propagation, and batch behavior.

## Phase 3.6: OpenTelemetry Tracing (Implemented)

### Rollout Plan
1. Foundation
- Add OpenTelemetry SDK + OTLP exporter dependencies.
- Extend typed settings and `.env.example` with tracing toggles and OTLP endpoint.
- Configure tracing once at startup with sampling and service metadata.

2. Instrumentation
- Add root span for each UI message turn.
- Add orchestrator spans for round execution, tool dispatch, and citation fallback pass.
- Add model adapter spans capturing request/response previews.
- Add retrieval and citation tool spans with validated counters and chunk/citation previews.
- Add retriever infrastructure spans for query parameters and top-k result preview.

3. Developer Experience
- Document local Jaeger setup and trace verification workflow in `README.md`.
- Keep payload previews bounded/truncated to avoid oversized span attributes.

### Acceptance Criteria
- With tracing enabled, a single chat interaction yields a navigable trace in Jaeger.
- Trace hierarchy includes UI -> orchestrator -> model/tool/retriever spans.
- Disabling tracing via env vars returns to no-op behavior without code changes.

## Phase 4: PDF + Multi-Modal Extraction
- Add pdf ingestion with extraction-first strategy on top of Phase 3.5 ingestion architecture.
- Extend metadata handling (page, section, modality-specific provenance if available).
- Re-run evaluation set and tune retrieval parameters.

## Phase 5: Evaluation and Hardening
- Establish benchmark dataset and automated evaluation script.
- Add CI pipeline for tests and evaluation checks.
- Enforce CI gates: `uv run ruff check .`, `uv run pyright`, `uv run pytest`.
- Enforce CI evaluation gates against MVP thresholds (correctness, citation relevance, unsupported-claim rate).
- Improve error handling and logging quality.

## Risks and Mitigations
- Risk: Chainlit integration friction.
  - Mitigation: switch to Gradio if Chainlit blocks progress (see trigger criteria in 03-architecture.md); decision requires explicit human approval — never an autonomous agent call.
- Risk: extraction quality issues for PDFs.
  - Mitigation: add preprocessing checks and quality diagnostics.
- Risk: local model quality instability.
  - Mitigation: model pinning and benchmark-based acceptance gates.

## Exit Criteria for MVP
- All core FRs implemented.
- Evaluation thresholds met.
- README provides reproducible setup and run instructions.
- Known limitations documented explicitly.

## Working Agreement for AI-Assisted Implementation
- AI agents must not unilaterally change, simplify, or reinterpret agreed requirements during implementation.
- If a conflict or problem is discovered (e.g. a requirement seems technically infeasible or ambiguous), the agent must stop and raise it explicitly with the human before proceeding.
- Trade-off decisions are never the agent's call to make autonomously; all trade-offs require explicit human approval.
- Agents must not rationalize deviations as "acceptable" — that is the human's judgment to make.
- Agreed design decisions are intentional and coordinated; apparent conflicts between pieces are a signal to discuss, not to "fix" silently.
- We favour clean design over "small diff" or "small amount of files" etc. 

## Path to Production Readiness
The following concerns are explicitly out of MVP scope but represent the known gap between the learning/demo build and a production-grade deployment. Documenting them here avoids surprises later.

### Observability and Instrumentation
- Replace stdout structured logs with a proper log aggregation pipeline.
- Add model latency and token usage metrics.
- Add alerting on error rates and latency regressions.

### Authentication and Authorization
- Replace username/password simulation with realistic identity provider (e.g. OAuth2/OIDC), possibly as global auth before the tool can be used.
- Enable Chainlit app-level authentication gating.
- Implement session expiry and token revocation.

### Multi-User and Stateless Architecture
- Chainlit is inherently stateful (WebSocket-based, server-side session via `cl.user_session`). It is not designed for horizontal scaling or stateless load-balancing — this is an architectural constraint, not a configuration option.
- For multi-user deployments, the recommended path is to decouple the UI from the backend: Chainlit (or a custom frontend) becomes a thin UI shell; the orchestrator moves to a separate stateless REST/gRPC service; session state is either fully client-side (JWT, signed tokens) or stored in an external session store (e.g. Redis).
- Per-user credentials must never be held in server memory across requests in a multi-user context — move to short-lived signed tokens issued by an identity provider (OAuth2/OIDC).

### Deployment and Operations
- Containerize application (Dockerfile + docker-compose for full local stack).
- Publish Qdrant, Ollama, and app as composable service units.
- Consider a server deployment image (e.g. for internal hosting).
- Define backup/restore strategy for the vector index.

### Security Hardening
- Red-team prompt injection attack vectors (especially via ingested documents).
- Add rate limiting and abuse prevention for the chat endpoint.
- Regular dependency vulnerability scanning (e.g. `pip audit` in CI).

### Quality and Evaluation at Scale
- Automated CI evaluation pipeline with threshold gates (e.g. CircleCI).
- Advanced RAG retrieval methods (hybrid search, reranking, HyDE).
- Human-in-the-loop feedback loop for continuous improvement.

### Extensibility
- Make model provider switchable via configuration.
- Add document upload capability for end users.
- Explore MCP-based tool ecosystem if more tools are added.
