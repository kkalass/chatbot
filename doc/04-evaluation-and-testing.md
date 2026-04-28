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
- `InteractiveVacationDaysAuthSession`: cache-hit path, first-use collection path (via fake `ask_user` callable), cancellation path, clear-on-auth-failure path.
- `VacationDaysTool`: happy path (credentials present), auth-failure path, cancellation path.
- Orchestrator agentic loop: single tool call round-trip, multi-tool sequence, loop termination on plain-text response.

### Integration Tests
- Ingest fixture corpus and query known facts.
- Ingest mixed txt/md fixture corpus and assert converter routing succeeds.
- Verify citations are present and linked to expected docs.
- Verify citation metadata fields originating from sidecar files are present in retrieved chunk metadata.
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
- Configure local trace inspection with Jaeger and validate at least one complete trace per integration test session when diagnosing behavior.
- Keep logs and traces complementary: logs for broad event streams, traces for per-turn causal analysis.

## MVP Thresholds
- Correctness >= 80% on benchmark set.
- Citation relevance >= 90% for answerable questions.
- Unsupported claim rate <= 10%.
