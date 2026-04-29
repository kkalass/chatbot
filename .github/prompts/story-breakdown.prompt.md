---
description: "Break down a feature or phase into implementation-ready Epic -> Stories -> Subtasks (planning only, no code changes)."
argument-hint: "Feature/phase to break down (e.g. 'Phase 7 Inline Quote Streaming')"
agent: "agent"
---

Create a delivery-ready work breakdown for **${input:scope}**.

## Objective
Produce a planning artifact that can be copied into Jira/Azure Boards/GitHub Issues.
This prompt is **planning only**: do not modify source code and do not run implementation steps.

## Inputs To Read First
1. [doc/phases/07-inline-quote-streaming.md](../../doc/phases/07-inline-quote-streaming.md) if relevant to the requested scope.
2. [doc/05-delivery-plan.md](../../doc/05-delivery-plan.md)
3. [doc/03-architecture.md](../../doc/03-architecture.md)
4. [doc/02-requirements.md](../../doc/02-requirements.md)
5. [doc/04-evaluation-and-testing.md](../../doc/04-evaluation-and-testing.md)

If the requested scope is outside Phase 7, still follow the same breakdown quality rules.

## Required Output Structure

### 1) Epic Summary
- Problem statement
- Business/engineering outcome
- In-scope / out-of-scope
- Risks and assumptions

### 2) Stories (Implementation Units)
For each story provide:
- `Story ID`: short stable ID (e.g. `QTS-01`)
- `Title`
- `Goal`
- `Scope`
- `Out of scope`
- `Dependencies`
- `Primary files/modules likely touched`
- `Acceptance criteria` (testable, binary)
- `Test plan` (unit/integration/regression)
- `Observability/metrics impact`
- `Rollback or feature-flag strategy` if relevant

### 3) Subtasks per Story
For each story provide subtasks that are implementation-ready:
- concrete action
- expected artifact/output
- done condition

### 4) Delivery Plan
- Recommended execution order
- Parallelization opportunities
- Critical path
- Suggested PR slicing strategy

### 5) Definition of Done (Epic Level)
- Cross-story completion gates
- Quality gates (`ruff`, `pyright`, `pytest`)
- Performance and regression validation expectations

## Quality Rules
- Keep story granularity between 0.5 and 2 days of implementation effort where possible.
- Every story must be independently testable.
- Avoid stories that mix unrelated concerns.
- Prefer sequencing that minimizes interface churn.
- Ensure explicit treatment of edge cases and failure modes.

## Architecture and Governance Rules
- Respect existing architecture boundaries and dependency inversion.
- Do not introduce implicit trade-off decisions; call them out explicitly under assumptions/questions.
- If a required design choice is unresolved, include a `Decision Needed` section with options and impacts.

## Output Style
- Use concise markdown.
- Use checklists for acceptance criteria and DoD.
- Keep content actionable and implementation-oriented.
- Do not include speculative future work unless explicitly requested.
