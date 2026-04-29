---
description: "Implement Epic/Story work for Phase 7 Inline Quote Streaming (WP1-WP7)."
argument-hint: "Work package or story to implement (e.g. 'WP1', 'WP4', 'WP7', or a custom story title)"
agent: "agent"
---

Implement **${input:work_item}** for Phase 7 as specified in [doc/phases/07-inline-quote-streaming.md](../../doc/phases/07-inline-quote-streaming.md).

## Context To Read First
1. [doc/phases/07-inline-quote-streaming.md](../../doc/phases/07-inline-quote-streaming.md) — locked decisions, touchpoints, WP order, acceptance.
2. [doc/03-architecture.md](../../doc/03-architecture.md) — boundaries, event stream, tool orchestration, auth constraints.
3. [doc/02-requirements.md](../../doc/02-requirements.md) — FR/NFR requirements.
4. [doc/04-evaluation-and-testing.md](../../doc/04-evaluation-and-testing.md) — validation expectations.
5. [doc/05-delivery-plan.md](../../doc/05-delivery-plan.md) — project sequencing context.

## Execution Mode
1. Determine whether `${input:work_item}` maps to one of `WP1..WP7` in the phase document.
2. If it maps to a WP, implement exactly that WP scope and acceptance criteria.
3. If it is a custom story title, map it to the relevant WP section(s), state the mapping, and implement only that bounded scope.
4. Keep changes minimal and architecture-aligned; no speculative refactors.
5. Include tests in the same change set for any behavior/interface updates.

## Mandatory Technical Rules
- Follow all locked decisions in the phase doc exactly.
- Keep stream processing non-blocking on quote parse failures.
- Enforce hard quote buffer limits.
- Use canonical structural dedup keys only.
- Validate quotes strictly against actual orchestrator history.
- Keep model-specific prompt profile adjustments where needed.
- Keep legacy citation round-trip behavior only as specified by the rollout flags.

## Required Quality Gates (run after each completed task)
- `uv run ruff check .`
- `uv run pyright`
- `uv run pytest`

If a gate fails, fix issues before continuing.

## Hard Architecture Constraints
- Never import infrastructure directly into orchestration logic.
- Use Protocol-based boundaries and constructor injection.
- Keep Pydantic at system boundaries; use immutable internal value objects.
- Never introduce `Any` without explicit justification comment.
- Never add compatibility shims to avoid updating call sites; update all callers and tests.

## Completion Output Format
At the end, provide:
1. Implemented scope: what part of `${input:work_item}` was completed.
2. Files changed: list with purpose.
3. Tests added/updated.
4. Quality gate results.
5. Acceptance checklist: each criterion marked `done` or `not done` with reason.
6. Follow-ups (if any) explicitly marked as out of scope.

## Stop Conditions
Stop and ask for guidance instead of guessing if:
- A requirement conflicts with locked decisions.
- A trade-off decision is required.
- The requested story implies scope outside the selected WP without explicit approval.
