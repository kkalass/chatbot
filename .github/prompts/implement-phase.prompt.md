---
description: "Implement a delivery phase from the project plan. Use when asked to implement Phase 0, Phase 1, Phase 2, Phase 3, Phase 4, or Phase 5 of the chatbot project."
argument-hint: "Which phase to implement (e.g. 'Phase 0', 'Phase 1')"
agent: "agent"
---

Implement **${input:phase}** of the RAG chatbot project as defined in [doc/05-delivery-plan.md](../../doc/05-delivery-plan.md).

## Instructions

1. Read the delivery plan and identify all tasks for the requested phase.
2. Read the relevant spec documents for that phase:
   - [doc/01-product-scope.md](../../doc/01-product-scope.md) — goals and scope
   - [doc/02-requirements.md](../../doc/02-requirements.md) — FR/NFR, especially NFR-06 code quality
   - [doc/03-architecture.md](../../doc/03-architecture.md) — module structure, interfaces, auth sequence
   - [doc/04-evaluation-and-testing.md](../../doc/04-evaluation-and-testing.md) — test strategy and coverage requirements
3. Implement each task in the phase sequentially. For each task:
   - Announce what you are about to implement and why.
   - Implement the code following all NFR-06 rules (pyright strict, ruff, structlog, Protocol interfaces, constructor DI, Pydantic at boundaries, frozen dataclass internally).
   - Write tests as required by the spec before or alongside the implementation.
   - After each task, run `uv run ruff check .`, `uv run pyright`, and `uv run pytest` to verify the codebase stays clean.
4. After all tasks are done, confirm the phase exit criteria are met.

## Hard Rules
- Never deviate from the agreed architecture or requirements. If something seems infeasible or ambiguous, **stop and ask** before proceeding.
- Never make trade-off decisions autonomously — present them and wait for explicit approval.
- We favour clean design over "small diff" or "small amount of files" etc.
- Never import infrastructure directly in orchestration code — always go through Protocol interfaces.
- Never use `logging.getLogger()` — always `structlog.get_logger()`.
- The codebase must pass `pyright` and `ruff check` after every task — do not defer fixes.
