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
- Introduce `src/chatbot/app/prompts.py`: `@dataclass(frozen=True) class Prompts` with callable prompt fields (request-time system prompts and parameterised citation prompt builders); module-level `DEFAULT_PROMPTS` constant. The orchestrator derives its effective `Prompts` from `DEFAULT_PROMPTS` via an injected `PromptProfile` at construction time. All future prompts (system prompt, citation-system prompt, citation request message, etc.) live here only.
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
- Introduce `ToolInputModel` as the shared Pydantic base model for LLM-facing tool arguments, including coercion of JSON-serialized structured values.
- Introduce `ToolEvent` type alias union and `ProcessEvent = str | ToolEvent` in `src/chatbot/app/protocols.py`; add `SourceCitationEvent` as the first `ToolEvent` variant.
- Implement `CitationTool` (`cite_sources`) as a regular tool: validates cited `source` + `chunk_id` pairs against `search_documents` results in `ToolContext.history`, emits `SourceCitationEvent` in its `ToolEvent` list, and appends its `JsonObject` result to history.
- Update the orchestrator to propagate `ToolEvent` items from tool execution into the `process_message` stream.
- Implement dedicated citation pass prompting: after the main loop, if the current turn used `search_documents` and no citation event was emitted, issue a citation-only prompt containing rendered turn-local search results and the final answer, with only `cite_sources` exposed.
- Use a dedicated citation system prompt (separate from answer generation) for citation rounds to reduce prompt interference.
- Add citation-pass robustness fallback: if the model emits serialized `cite_sources` JSON as text instead of a native tool call, parse and recover it into a dispatched tool call.
- Add model-specific `SmallModelPromptProfile` for llama-family models to tighten tool-calling instructions and simplify `cite_sources` schema.
- Update `process_message` to yield `ProcessEvent` (`str | ToolEvent`); update the UI to use `match`/`case` + `assert_never` and render `SourceCitationEvent` as Chainlit citation elements plus a compact deduplicated sources section with linkified labels.
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


## Phase 4.1: Arize Phoenix Observability Alignment
Decision baseline for this phase:
- Replace Jaeger as the primary local trace backend with local Arize Phoenix.
- No Arize Cloud in this phase (local-only deployment).
- Prefer auto-instrumentation where it provides clear value.
- Keep manual tracing only for domain-critical spans/attributes that auto-instrumentation cannot infer.

Why parallel Jaeger + Phoenix is not selected now:
- Parallel export increases setup complexity, duplicate storage, and analysis drift.
- Parallel mode is only justified as a short migration fallback when validating parity.
- Exit rule for fallback: remove Jaeger once parity checks pass.

### Scope
1. Backend switch
- Route OpenTelemetry export to local Phoenix OTLP endpoint.
- Remove Jaeger from the default developer runbook and troubleshooting path.

2. Auto-instrumentation rollout (now, in this phase)
- Add OpenInference/Phoenix instrumentation for LLM stack components where supported by the used libraries.
- Add auto-instrumentation for framework/tooling components where stable support exists and signal quality is useful.
- Keep auto-instrumentation selective: disable noisy instrumentors that do not improve debugging or evaluation outcomes.

3. Manual tracing simplification
- Keep as manual only when auto-instrumentation cannot provide equivalent signal:
  - all LLM calls with request/response previews and tool-call payload visibility,
  - all tool executions with input/output summaries and error status.
- Keep (high value):
  - orchestrator control-flow spans for round boundaries and tool dispatch decisions,
  - citation-pass recovery telemetry and counters,
  - bounded preview attributes for safe inspectability.
- Remove or reduce:
  - manual spans that duplicate auto-generated spans without additional domain signal.
  - any fallback manual span once an auto span reaches parity for name stability, payload quality, and error visibility.

### Keep/Drop Matrix
- Keep: `chat.ui.on_message`
  - Reason: session-scoped turn boundary, emitted-response summary, and citation-event aggregation are application-level semantics and not expected from auto-instrumentation.
- Keep: `chat.orchestrator.round`
  - Reason: round boundaries and round-local tool-call decisions are agentic control-flow signals and not inferable from lower layers.
- Drop: `chat.orchestrator.process_message`
  - Reason: the UI turn span already provides the top-level turn boundary, so a second root-like orchestrator span added little value and made the trace hierarchy noisier.
- Keep: `chat.orchestrator.tool_dispatch`
  - Reason: it still adds application-level causality between model-produced tool calls and the subsequent concrete tool span, which Phoenix does not infer from the custom tool dispatch code path.
- Keep: `chat.orchestrator.citation_pass`
  - Reason: citation-pass isolation, fallback recovery attempts, and success/failure reasons are domain-specific behavior.
- Conditional keep: `chat.model.ollama.stream`
  - Keep until auto-instrumentation provides OpenInference-compliant `LLM` spans with equivalent request/response visibility, tool-call payload visibility, provider/model fields, and error visibility.
  - Drop only after parity is demonstrated in Phoenix for the real Ollama integration used in this project.
- Keep: `chat.tool.search_documents`
  - Reason: this is a custom tool span with domain-specific request/response summaries; generic auto-instrumentation cannot infer tool semantics from internal Python code.
- Keep: `chat.tool.cite_sources`
  - Reason: citation validation counts and validated/unvalidated pair summaries are domain-specific and not auto-derivable.
- Conditional keep: `chat.retriever.qdrant.retrieve`
  - Keep if needed to attach OpenInference `RETRIEVER` semantics and retrieved-document previews.
  - Drop only if future instrumentation emits equally useful retriever spans with document IDs, scores, and content previews.

### OpenInference Mapping Plan
Use OpenInference semantic conventions wherever they improve Phoenix rendering. Existing `chat.*` attributes may remain as project-local diagnostics, but Phoenix-facing GenAI structure should be expressed with OpenInference-compliant span kinds and attributes.

1. UI turn span: `chat.ui.on_message`
- Span kind: `CHAIN`
- Purpose: represent one end-user interaction as the top-level GenAI workflow span.
- Required attributes:
  - `openinference.span.kind=CHAIN`
  - `session.id` from the Chainlit session trace identifier.
  - `input.value` as the raw user message text.
  - `input.mime_type=text/plain`
  - `output.value` as the final emitted assistant text including any appended source section.
  - `output.mime_type=text/plain`
- Optional attributes:
  - `metadata` for bounded JSON metadata such as citation-event count.

2. Orchestrator round span: `chat.orchestrator.round`
- Span kind: `CHAIN`
- Purpose: represent one agentic loop iteration.
- Required attributes:
  - `openinference.span.kind=CHAIN`
  - `input.value` as bounded message-summary input for the round.
  - `output.value` as bounded round output summary.
- Project-local attributes to keep:
  - `chat.round`
  - `chat.round.tool_call_count`
  - `chat.round.tool_calls`
  - `chat.round.text_chars`

3. LLM span: `chat.model.ollama.stream`
- Target state: OpenInference-compliant `LLM` span, preferably from auto-instrumentation; manual fallback if parity is missing.
- Span kind: `LLM`
- Required attributes:
  - `openinference.span.kind=LLM`
  - `llm.model_name`
  - `llm.provider`
  - `input.value` as bounded serialized request payload when full chat-message structure is not available.
  - `input.mime_type=application/json`
  - `output.value` as bounded serialized response payload.
  - `output.mime_type=application/json`
- Preferred richer attributes when feasible:
  - `llm.input_messages.*.message.role`
  - `llm.input_messages.*.message.content`
  - `llm.output_messages.*.message.role`
  - `llm.output_messages.*.message.content`
  - `llm.invocation_parameters`
  - `llm.token_count.prompt`
  - `llm.token_count.completion`
  - `llm.token_count.total`
- Tool-calling visibility:
  - include tool-call payloads in the LLM output structure using OpenInference message/tool-call representation when the used instrumentor supports it;
  - otherwise keep the bounded project-local fallback `llm.response.tool_calls` until parity exists.

4. Tool spans: `chat.tool.search_documents`, `chat.tool.cite_sources`
- Span kind: `TOOL`
- Purpose: represent explicit tool execution as first-class Phoenix tool spans.
- Required attributes:
  - `openinference.span.kind=TOOL`
  - `tool.name`
  - `tool.parameters` as JSON string for validated tool input.
  - `input.value` as bounded serialized tool input.
  - `input.mime_type=application/json`
  - `output.value` as bounded serialized tool result.
  - `output.mime_type=application/json`
- Project-local attributes to keep:
  - for `search_documents`: query, chunk count, chunk previews.
  - for `cite_sources`: claimed/validated/unvalidated counts and pair previews.

5. Retriever span: `chat.retriever.qdrant.retrieve`
- Span kind: `RETRIEVER`
- Purpose: expose retrieval quality directly in Phoenix rather than only as a generic infrastructure span.
- Required attributes:
  - `openinference.span.kind=RETRIEVER`
  - `input.value` as the bounded retrieval query text.
  - `input.mime_type=text/plain`
- Preferred retrieval-document attributes:
  - `retrieval.documents.*.document.id`
  - `retrieval.documents.*.document.content`
  - `retrieval.documents.*.document.score`
- Project-local attributes to keep:
  - `chat.retriever.top_k`
  - `chat.retriever.score_threshold`
  - `chat.retriever.result_count`

6. Citation-pass span: `chat.orchestrator.citation_pass`
- Span kind: `CHAIN`
- Purpose: represent the isolated post-answer citation workflow.
- Required attributes:
  - `openinference.span.kind=CHAIN`
  - `input.value` as bounded citation-pass prompt summary.
  - `output.value` as bounded citation-pass result summary.
- Project-local attributes to keep:
  - recovery attempted/succeeded
  - no-tool-call / wrong-tool-call diagnostics
  - dispatch result preview

### OpenInference Compliance Rules
- Always prefer official OpenInference semantic keys for Phoenix-facing GenAI structure; avoid inventing project-local substitutes for concepts already covered by OpenInference.
- Implementation rule: use exported OpenInference constants and helper APIs where available; do not hand-write OpenInference attribute keys as ad-hoc string literals in application code.
- Implementation rule: any work done solely for tracing, including bounded preview generation and temporary payload shaping, must live in top-level `_trace_*` helper functions rather than inline in business logic.
- Keep `chat.*` attributes only for project-specific diagnostics not covered by OpenInference.
- All large payloads must remain bounded/truncated before being written to span attributes.
- Do not duplicate the same semantic payload in both OpenInference and project-local attributes unless required temporarily for migration parity.
- Acceptance test for each conditional-drop span: Phoenix must still show an equally usable view for request, response, errors, and tool/retrieval relationships after the manual span is removed.

4. Phoenix project organization (local recommendation)
- Use one default local project name for day-to-day development (e.g. `chatbot-local`).
- Distinguish runs by resource attributes/tags (`service.name`, `deployment.environment`, `git.branch`, optional `run.id`) rather than many project names.
- Keep naming stable to preserve longitudinal comparison in Phoenix.

5. Documentation and developer workflow
- Update README from Jaeger-first to Phoenix-first setup and validation.
- Document a concise "trace quality checklist" (what must be visible per chat turn).
- Document the keep/remove tracing policy to prevent uncontrolled span growth.

### Acceptance Criteria
- Local Phoenix receives traces via OTLP and shows a complete, navigable trace for one chat turn.
- For each turn, Phoenix clearly shows:
  - LLM call spans with request/response previews and tool-call information.
  - Tool execution spans with request/response summaries and failures.
- Manual orchestrator/citation spans remain only where they add domain-specific value beyond auto-instrumentation.
- Default developer docs no longer require Jaeger.
- Tracing remains optional and can still be disabled via environment variables with no code changes.

### Non-Goals (explicit)
- No Arize Cloud integration in this phase.
- No production-grade observability hardening (alerting, retention strategy, SLO dashboards) in this phase.

## Phase 4.2: Additional Local Model Support
- Add model profile and configuration support for DeepSeek R1 (`deepseek-r1:32b` or `deepseek-r1:14b`) via Ollama.
- Add model profile and configuration support for Qwen3 (`qwen3:32b` or `qwen3:14b`) via Ollama.
- Ensure prompt-profile/tool-calling compatibility checks for both models.
- Extend smoke/integration evaluation runs to compare answer quality, citation behavior, and latency across supported models.
- Update setup docs with explicit pull/run commands and recommended defaults per hardware tier.

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
