# Evaluation Playbook

## Purpose
This document defines the operational process for evaluating model and prompt changes in a reproducible way.

It complements the high-level guidance in `doc/04-evaluation-and-testing.md` and is the source of truth for day-to-day evaluation execution.

## Scope
Use this playbook for:
- Prompt changes (system prompt, citation prompt, profile tweaks).
- Model substitutions or upgrades.
- Retrieval parameter changes that can affect answer quality.

Out of scope:
- Pure infrastructure performance tuning with no expected impact on answer/citation quality.

## Principles
- Separate exploration from decision making.
- Keep one variable per candidate when possible.
- Never promote a candidate based on ad-hoc traces alone.
- All release decisions must be backed by Dataset + Experiment runs.

## Evaluation Artifacts
Required artifacts per evaluation cycle:
- `evaluation_name`: Human-readable cycle name (for example, `rag-prompt-tuning-2026-04`).
- `dataset_version`: Immutable dataset snapshot identifier.
- `corpus_version`: Corpus snapshot or ingestion commit reference.
- `baseline_candidate`: The current production/stable reference candidate.
- Candidate definitions table (model, prompts, retrieval config).

## Candidate Identity
Every run must be uniquely identifiable with these metadata keys:
- `evaluation_name`
- `run_id`
- `candidate_id`
- `model_name`
- `prompt_version_answer`
- `prompt_version_citation`
- `retrieval_version`
- `corpus_version`
- `dataset_version` (for experiment runs)
- `temperature`
- `seed` (if supported)
- `environment` (`local`, `staging`, `ci`)

Emission behavior in application traces:
- Always emitted automatically: `run_id`, `trace_session_id`, `model_name`, `retrieval_version`, `temperature`, `seed`, `environment`.
- Emitted only when configured: `evaluation_name`, `candidate_id`, `prompt_version_answer`, `prompt_version_citation`, `corpus_version`, `dataset_version`.
- `run_id` is auto-generated once per application process when `EVAL_RUN_ID` is not set.

Recommended naming convention:
- `candidate_id = <model>__ans-<N>__cit-<N>__ret-<N>`

## Phases

### Phase 1: Exploratory Tracing (Hypothesis Building)
Goal:
- Quickly detect failure modes and generate candidate hypotheses.

Execution:
- Run representative interactive queries.
- Inspect traces and spans in Phoenix Traces.
- Compare candidates using the same query subset and tagging scheme.

Mandatory telemetry checks:
- Spans are present for UI turn, orchestrator, retriever, LLM, and tools.
- Session context is propagated across child spans.
- Input/output attributes are present and readable.
- Citation recovery telemetry is populated when applicable.

Expected outcomes:
- Shortlist of candidates.
- Concrete hypothesis per candidate (for example, "better grounding on multi-hop questions, slightly higher latency").

Exit criteria to Phase 2:
- At least one clear baseline and one challenger candidate defined.
- Known failure modes documented.
- Candidate metadata tagging verified.

### Phase 2: Dataset + Experiments (Decision Evidence)
Goal:
- Produce reproducible, comparable metrics for decision making.

Execution:
- Use a fixed dataset snapshot.
- Run baseline and challengers as separate Phoenix experiment runs.
- Keep non-target variables constant (corpus, retrieval version, runtime env).
- For stochastic models: run multiple repetitions per candidate.

Required controls:
- Same dataset across candidates.
- Same retrieval config unless retrieval is the explicit variable under test.
- Same environment and service availability assumptions.

Expected outcomes:
- Per-candidate scorecards.
- Ranking with explicit pass/fail decision.

Exit criteria to Phase 3:
- Minimum sample size reached.
- Metrics complete for all mandatory dimensions.
- Decision can be justified with experiment evidence.

### Phase 3: Promotion Decision
Goal:
- Approve or reject challenger candidate for adoption.

Decision policy:
- Reject if unsupported-claim behavior regresses materially.
- Reject if citation quality drops below threshold.
- Prefer higher factual correctness if latency/cost remain within guardrails.
- In ties, keep current baseline unless challenger has clear operational benefit.

Required outputs:
- Selected candidate id.
- Decision rationale (2-5 bullets).
- Rollback condition and trigger.

## Dataset Design Rules
- Include answerable and non-answerable questions.
- Cover easy, ambiguous, and multi-hop scenarios.
- Include citation-sensitive questions that require source grounding.
- Freeze dataset version for each decision cycle.

Minimum structure per item:
- `query`
- `expected_intent`
- `expected_supporting_sources` (when answerable)
- `disallowed_claims` (optional)
- `difficulty` label (`easy`, `ambiguous`, `multi-hop`, `negative`)

## Metric Set

### Quality
- `factual_correctness`
- `answer_completeness`

### Grounding
- `citation_precision`
- `citation_relevance`
- `unsupported_claim_rate`

### Retrieval
- `retrieval_hit_rate_at_k`

### Operations
- `latency_p50`
- `latency_p95`
- `tool_call_rate`
- `citation_recovery_rate`

### Cost (if available)
- `input_tokens`
- `output_tokens`

## Suggested Guardrail Thresholds
Use these as defaults unless a cycle explicitly overrides them:
- `factual_correctness >= 0.80`
- `citation_relevance >= 0.90` on answerable items
- `unsupported_claim_rate <= 0.10`
- `latency_p95` must not regress by more than 20% versus baseline

## Required OTel/OpenInference Evidence in Phase 1
These attributes should be visible for debugging and comparability:
- Session and run identity metadata keys listed in "Candidate Identity".
- Input/output attributes on key spans.
- LLM metadata (`provider`, `model_name`, invocation parameters where available).
- Retriever document context attributes for retrieval spans.
- Tool execution attributes for tool spans.

If these are missing, fix observability first before trusting evaluation outcomes.

## Reporting Template (Per Evaluation Cycle)
- Evaluation name:
- Date/time window:
- Baseline candidate:
- Challenger candidates:
- Dataset version:
- Corpus version:
- Summary table (metrics per candidate):
- Decision:
- Rationale:
- Rollback trigger:
- Follow-up actions:

## Anti-Patterns
- Choosing a winner from one or two anecdotal traces.
- Comparing runs with different dataset versions.
- Mixing prompt and retrieval changes in a single candidate without intent.
- Changing corpus content during candidate comparison.
- Ignoring missing telemetry and still drawing conclusions.

## Process Summary
1. Build hypotheses in Traces.
2. Validate candidates in Experiments on a fixed dataset.
3. Decide with guardrails and documented rationale.
4. Keep artifacts and metadata for reproducibility.
