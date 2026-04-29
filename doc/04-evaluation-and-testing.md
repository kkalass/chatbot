# Evaluation and Testing

Operational execution details for model/prompt comparison are defined in `doc/06-evaluation.md`.

## Quality Goals
- Verify factual grounding and reduce hallucinations.
- Measure retrieval quality independently from generation quality.
- Prevent regressions after ingestion/prompt/tooling changes.

## Evaluation Dataset
Create a small but representative benchmark set:
- 20-50 domain questions.
- For each question:
  - expected answer intent,
  - expected supporting source document(s),
  - optional disallowed claims.

## Metrics
- Answer correctness rate.
- Citation relevance rate.
- Unsupported claim rate (hallucination proxy).
- Retrieval hit rate@k (does top-k include expected source).
- Latency (p50/p95 end-to-end).
- Ingestion throughput and peak memory usage under bounded micro-batch settings.

## Test Strategy

### Unit Tests
- Chunking logic.
- Converter routing by file type (txt vs md as baseline).
- Splitter strategy selection by document character/type.
- Metadata continuity across conversion -> splitting -> embedding -> write payload mapping.
- Sidecar metadata semantics (`<document>.meta.json` merged into owner document, sidecar file not ingested as standalone source).
- Embedder injection boundary behavior (ingestion orchestration independent from concrete provider implementation).
- Retrieval filtering and ranking behavior.
- Prompt assembly with source context.
- Tool input/output validation with Pydantic.
- Tool-input coercion for weak-model payloads (JSON-serialized list/object fields parsed before validation).
- `InteractiveVacationDaysAuthSession`: cache-hit path, first-use collection path (via fake `ask_user` callable), cancellation path, clear-on-auth-failure path.
- `VacationDaysTool`: happy path (credentials present), auth-failure path, cancellation path.
- Orchestrator agentic loop: single tool call round-trip, multi-tool sequence, loop termination on plain-text response.
- Citation pass behavior: dedicated citation prompt assembly (`<search_results>` + `<answer>`), wrong-tool-call rejection, serialized `cite_sources` text recovery, and malformed serialized payload recovery.
- Citation pass prompting isolation: citation rounds use the citation-specific system prompt, while main answer rounds use the general system prompt.
- Citation UI rendering helpers: metadata-first headers, linked title/label when `source_url` exists, deduplicated sources list generation, and omission of standalone raw URL lines.

### Integration Tests
- Ingest fixture corpus and query known facts.
- Ingest mixed txt/md fixture corpus and assert converter routing succeeds.
- Verify citations are present and linked to expected docs.
- Verify citation metadata fields originating from sidecar files are present in retrieved chunk metadata.
- Verify citation rendering uses metadata labels and linked titles (when `source_url` exists) without displaying standalone raw URL lines.
- Verify uncertainty behavior when evidence is missing.
- Verify tool-call path for vacation-days scenarios.

### Smoke Tests
- Application startup with required services available.
- Basic query/answer roundtrip via UI or API hook.

## Regression Guardrails
- Keep a fixed evaluation set in repository.
- Run evaluation in CI on pull requests.
- Fail CI if correctness/citation metrics drop below threshold.
- Add ingestion regression checks for metadata continuity and bounded-memory batch processing behavior.

## Observability Recommendations
- Structured logs for:
  - retrieval candidates and scores,
  - selected context chunks,
  - tool-call decisions,
  - model/token timing.
- Persist evaluation runs with timestamp and model/retrieval config.
- Use OpenTelemetry (OTLP) tracing for request-level spans covering UI turn handling, orchestrator rounds, model generation, retrieval, and tool execution.
- Inspect local traces in Phoenix and validate at least one complete end-to-end trace per integration test session when diagnosing behavior.
- Keep tracing code structurally separate from business logic: any trace-only payload shaping, preview generation, or error annotation belongs in top-level `_trace_request`, `_trace_response`, or `_trace_error` helpers rather than inline in control-flow code.
- Keep logs and traces complementary: logs for broad event streams, traces for per-turn causal analysis.

## MVP Thresholds
- Correctness >= 80% on benchmark set.
- Citation relevance >= 90% for answerable questions.
- Unsupported claim rate <= 10%.
