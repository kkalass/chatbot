# Evaluation and Testing

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

## Test Strategy

### Unit Tests
- Chunking logic.
- Retrieval filtering and ranking behavior.
- Prompt assembly with source context.
- Tool input/output validation with Pydantic.
- `InteractiveVacationDaysAuthSession`: cache-hit path, first-use collection path (via fake `ask_user` callable), cancellation path, clear-on-auth-failure path.
- `VacationDaysTool`: happy path (credentials present), auth-failure path, cancellation path.
- Orchestrator agentic loop: single tool call round-trip, multi-tool sequence, loop termination on plain-text response.

### Integration Tests
- Ingest fixture corpus and query known facts.
- Verify citations are present and linked to expected docs.
- Verify uncertainty behavior when evidence is missing.
- Verify tool-call path for vacation-days scenarios.

### Smoke Tests
- Application startup with required services available.
- Basic query/answer roundtrip via UI or API hook.

## Regression Guardrails
- Keep a fixed evaluation set in repository.
- Run evaluation in CI on pull requests.
- Fail CI if correctness/citation metrics drop below threshold.

## Observability Recommendations
- Structured logs for:
  - retrieval candidates and scores,
  - selected context chunks,
  - tool-call decisions,
  - model/token timing.
- Persist evaluation runs with timestamp and model/retrieval config.
- Consider OpenTelemetry (OTLP) tracing for request-level spans covering retrieval, tool-call, and generation stages. This is out of scope for MVP but the call boundaries in the orchestrator should be instrumented-friendly (no spaghetti logic that makes tracing impractical to add later).
- Recommended MVP minimum: `structlog` with `ConsoleRenderer` in development and `JSONRenderer` to stdout in production/CI; OTEL spans as a Phase 5+ addition.

## MVP Thresholds
- Correctness >= 80% on benchmark set.
- Citation relevance >= 90% for answerable questions.
- Unsupported claim rate <= 10%.
