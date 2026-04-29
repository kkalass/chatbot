# Phase 7: Inline Quote Streaming (Replace Citation Round-Trip)

## Motivation
The current citation approach introduces a second model round-trip after answer generation. This has three practical drawbacks:
- Additional latency per user turn.
- Occasional citation quality issues due to post-hoc citation generation.
- A failure mode for non-retrieval answers (for example vacation-days tool answers), where some models keep running with high GPU load but never return a usable result.

This phase replaces post-hoc citation generation with **inline, streamed quote extraction** from the primary response stream.

## Goals
- Remove the dedicated citation round-trip.
- Produce quote references inline while text is streamed.
- Preserve strict validation against real tool/search history before final citation rendering.
- Keep orchestrator and UI typed and event-driven.
- Avoid any model work for citation handling on turns that do not need it.

## Non-Goals
- Redesign retrieval ranking/chunking behavior.
- Redesign Chainlit UI layout in this phase.
- Guarantee perfect quote extraction on every model from day one.

## Proposed Design

### 1. Protocol and Event Model Changes
Extend the stream contract so model output can include quote objects in addition to text and tool calls.

Planned type shape (conceptual):
- `ProcessEvent = str | ToolEvent | QuoteReferenceEvent`
- Model stream item from chat adapter: `str | list[ToolCallInfo] | Quote`
- `Quote` is a tagged union:
  - `SearchResultQuote`
  - `ToolCallQuote`

Recommended quote fields:
- Common:
  - `kind`: `search_result` or `tool_call`
  - `claim`: short natural-language statement the quote supports
  - `quote_text`: optional extracted snippet if provided by model
- `SearchResultQuote`:
  - `tool_call_id` (of `search_documents`)
  - `source`
  - `chunk_id`
- `ToolCallQuote`:
  - `tool_call_id` (of non-retrieval tool)
  - `tool_name`
  - `output_path` (optional JSON path into tool result)

`QuoteReferenceEvent` should carry at least:
- `reference_number` (for rendering like `[1]`)
- canonicalized quote identity (for dedup)

### 2. ChatModel Wrapper for Inline Quote Parsing
Introduce a wrapper model adapter around the current chat model (composition, not inheritance-heavy coupling).

Responsibilities:
- Consume upstream stream chunks.
- Forward normal text immediately.
- Detect quote markers in streamed text:
  - start marker: `<°_quote_°>`
  - end marker: `</°_quote_°>`
- Buffer only the marked segment.
- Parse buffered JSON into `Quote`.
- Yield parsed `Quote` as a first-class stream item.
- Continue forwarding following text.

Parser requirements:
- Marker-safe finite-state parser over chunk boundaries.
- Robust when markers are split across chunks.
- On parse failure: emit no quote item, log trace-safe warning, and forward original raw segment as plain text fallback.
- Bounded buffer guard (size limit) to avoid runaway memory use on malformed output.

### 3. Prompt Contract (System Prompt)
Update the system prompt to instruct the model:
- Whenever a statement is grounded in `search_documents` results, emit a `SearchResultQuote` JSON object between markers.
- Whenever a statement is grounded in another tool result, emit a `ToolCallQuote` JSON object between markers.
- Keep normal user-facing answer text outside markers.
- Do not emit markers for unsupported or uncertain claims.

Important prompt constraints:
- JSON must be strict and self-contained.
- Exactly one quote object per marker block.
- Use IDs and source/chunk identifiers from tool outputs already present in conversation context.

### 4. Orchestrator Behavior
While streaming:
- Emit normal text chunks as before.
- On `Quote` item:
  - Validate against turn history references (tool call exists, source/chunk exists, tool output exists).
  - Build canonical identity key and deduplicate.
  - Assign incremental reference number only on first occurrence.
  - Emit `QuoteReferenceEvent` so UI can append `[n]` inline.

After stream completion:
- Perform final validation pass over collected quotes.
- Build citation presentation payload from validated unique quotes.
- Append compact markdown “Sources” section to response text (as today).
- Populate right sidebar elements with full citation/tool evidence content (as today).

Validation rules:
- Invalid quotes are ignored for citation rendering.
- Quotes referencing unknown tool calls or unknown source/chunk pairs are dropped.
- If all quotes are invalid, no sources section is appended.

### 5. UI Behavior
Initial UI handling remains minimal:
- `QuoteReferenceEvent` is rendered inline as plain text reference token like `[1]`.
- End-of-turn source appendix and sidebar behavior remain equivalent to the current citation UX.

This preserves user-visible behavior while removing citation round-trip latency.

## Deduplication Strategy
Use a canonical key:
- Search quote: `(kind, tool_call_id, source, chunk_id)`
- Tool quote: `(kind, tool_call_id, tool_name, output_path?)`

Policy:
- First occurrence gets next reference number.
- Later duplicates reuse existing number.

## Observability
Add tracing counters/attributes:
- `quote.detected.count`
- `quote.parsed.count`
- `quote.parse_failed.count`
- `quote.validated.count`
- `quote.invalid.count`
- `quote.duplicate.count`

This is required to compare quality and latency against current citation-pass behavior.

## Migration Plan
1. Add quote types and event types.
2. Implement parsing wrapper model.
3. Add parser unit tests for chunk-boundary edge cases.
4. Update prompt templates.
5. Wire orchestrator quote collection, dedup, numbering, and final validation.
6. Update UI event rendering for `QuoteReferenceEvent`.
7. Remove citation fallback round-trip logic.
8. Run integration and regression test suite.

## Test Plan

### Unit Tests
- Quote parser:
  - marker in one chunk
  - marker split across chunks
  - malformed JSON
  - missing end marker
  - oversized buffer handling
- Orchestrator:
  - dedup numbering behavior
  - invalid quote rejection
  - mixed text/tool-call/quote stream handling
- Prompt profile/system prompt formatting assertions.

### Integration Tests
- Retrieval-grounded answer emits inline references and final sources.
- Tool-grounded answer (vacation-days) emits no GPU-hanging citation behavior.
- Non-grounded conversational answer emits no quotes and no source appendix.
- Weak-model behavior: malformed quote blocks do not break streaming.

### Performance/Behavioral Checks
- Latency reduction versus baseline citation round-trip.
- GPU load profile on non-retrieval turns.
- No regression in source-appendix/sidebar content quality.

## Risks and Mitigations
- Model does not follow marker contract reliably.
  - Mitigation: strict parser fallback to plain text, post-stream validation, prompt tightening per model profile.
- Marker collision with normal content.
  - Mitigation: use rare sentinel tokens and explicit instruction not to use them in prose.
- Over-citation noise in output.
  - Mitigation: prompt guidance on citation granularity and dedup in orchestrator.
- Incorrect tool/source IDs from model.
  - Mitigation: hard validation against actual history before rendering.

## Implementation Guardrails (Must-Haves)
The following constraints are mandatory for rollout and directly address known failure modes:
1. Streaming parser correctness across chunk boundaries
  - Parser must be stateful and marker-aware even when start/end markers are split across arbitrary token chunks.
2. Hard buffer safety limits
  - Quote-buffer accumulation must have strict byte/character limits and safe truncation/fallback behavior.
3. Never block text streaming on quote errors
  - JSON parse failure, malformed markers, or timeouts must degrade to plain text pass-through.
4. Canonical deduplication only
  - Dedup must use canonical structural keys, never free-text similarity.
5. Strict history-backed validation
  - Quotes are accepted only when referenced tool calls and retrieval identifiers exist in orchestrator history.
6. Model-specific prompt tuning as a first-class mechanism
  - Prompt profiles remain per model family to improve marker/JSON reliability on weaker models.
7. Regression tests for the known GPU-hang scenario are release-blocking
  - Non-retrieval turns (for example vacation tool flows) must be covered by automated regression tests.

## Locked Decisions for Implementation
To avoid ambiguity during execution, the following defaults are fixed for this phase:
- Marker protocol:
  - Start marker: `<°_quote_°>`
  - End marker: `</°_quote_°>`
  - Exactly one JSON object per marker block.
- Reference insertion behavior:
  - `QuoteReferenceEvent` is emitted at the exact stream position where the parsed quote block was received.
  - UI writes the token inline as `[n]` without additional formatting in this phase.
- Quote payload strictness:
  - `quote_text` is optional for both quote kinds.
  - `claim` is required and used for diagnostics only (not dedup identity).
- Dedup identity:
  - Search quote key: `(kind, tool_call_id, source, chunk_id)`
  - Tool quote key: `(kind, tool_call_id, tool_name, output_path?)`
- Invalid quote handling:
  - Invalid quotes are dropped from citation rendering.
  - Raw quote block content is forwarded as plain text when parse fails.
- Rollout switch:
  - Introduce `INLINE_QUOTES_ENABLED` setting.
  - Default value for this phase: enabled in local/dev, disabled in production until validation sign-off.
- Legacy citation pass:
  - Keep behind `CITATION_ROUND_TRIP_ENABLED` for one migration window.
  - Default value after feature readiness: disabled.

## Handoff: Concrete Code Touchpoints
The implementation is expected to touch the following files/modules:
- Protocols and event model:
  - `src/chatbot/app/protocols.py`
  - `src/chatbot/observability/schema.py` (if event payload schema updates are needed)
- Model streaming and wrapper integration:
  - `src/chatbot/infrastructure/chat/_ollama.py`
  - New wrapper module under `src/chatbot/infrastructure/chat/` for marker parsing
  - `src/chatbot/ui/app.py` composition root wiring
- Prompt contract:
  - `src/chatbot/app/prompts.py`
  - `src/chatbot/infrastructure/chat/_prompt_profile.py`
- Orchestrator and citation finalization:
  - `src/chatbot/app/orchestrator.py`
  - `src/chatbot/app/citation_support.py`
- UI event rendering:
  - `src/chatbot/ui/app.py`
  - `src/chatbot/ui/citation_view.py` (if source appendix/sidebar contract changes)
- Settings/config mapping:
  - `src/settings/__init__.py`
  - `src/chatbot/config.py`
- Tests:
  - `tests/unit/test_orchestrator.py`
  - `tests/unit/test_chat_prompt_profiles.py`
  - new unit tests for quote parser under `tests/unit/`
  - `tests/integration/test_rag.py`
  - `tests/unit/test_ui_citation_view.py` (if UI token handling changes)

## Work Packages (Implementation Order)

### WP1: Types and Settings Contract
Scope:
- Add `Quote`, `SearchResultQuote`, `ToolCallQuote`, and `QuoteReferenceEvent` models.
- Extend process stream unions and chat model stream item union types.
- Add feature flags (`INLINE_QUOTES_ENABLED`, `CITATION_ROUND_TRIP_ENABLED`) in settings and config mapping.

Acceptance:
- Pyright passes with strict typing.
- Existing tests compile against new stream/event types.

### WP2: Marker Parser Wrapper
Scope:
- Implement a stateful streaming parser wrapper around chat model output.
- Support marker detection across chunk boundaries.
- Enforce buffer limit and parse-fallback behavior.

Acceptance:
- Dedicated parser unit tests pass for all edge cases in this document.
- No stream-blocking behavior under malformed marker/JSON input.

### WP3: Prompt and Profile Update
Scope:
- Update system prompt contract for marker-delimited quote JSON.
- Apply profile-specific hardening for weaker models.

Acceptance:
- Prompt tests verify marker contract language is present.
- Model-specific profiles retain existing tool-calling constraints.

### WP4: Orchestrator Inline Quote Pipeline
Scope:
- Consume `Quote` stream items.
- Validate quotes against actual tool/search history.
- Assign stable deduplicated reference numbers.
- Emit `QuoteReferenceEvent` inline.
- Build end-of-turn validated citation payload for markdown sources + sidebar.

Acceptance:
- Unit tests cover dedup behavior, invalid quote rejection, mixed stream handling.
- No regression in normal text and tool-call flow.

### WP5: UI Rendering and Finalization
Scope:
- Render `QuoteReferenceEvent` as inline `[n]`.
- Keep end-of-turn sources and sidebar parity with current citation UX.

Acceptance:
- UI tests verify reference token rendering and final source presentation behavior.

### WP6: Migration and Legacy Path Sunset
Scope:
- Gate new flow by feature flag.
- Keep old citation round-trip behind compatibility flag for one release window.
- Default to inline flow in dev; rollout to prod after validation.

Acceptance:
- Both flows can be toggled via settings for A/B verification.
- Inline flow can run end-to-end without invoking citation round-trip.

### WP7: Integration, Regression, and Performance Validation
Scope:
- Add and run integration and regression tests.
- Compare latency and GPU behavior against baseline.

Acceptance:
- Regression test for non-retrieval GPU-hang scenario is passing.
- Measured latency improvement is documented.
- Observability counters visible in Phoenix.

## Execution Checklist for Engineering Handoff
1. Create implementation branch and keep each WP in a separate PR where possible.
2. Land WP1 + WP2 first; do not start orchestrator changes without parser tests green.
3. Land WP3 before WP4 to avoid temporary prompt/protocol mismatch.
4. Land WP4 + WP5 together to keep stream/UI contract synchronized.
5. Keep legacy path available until WP7 validation completes.
6. Remove or disable legacy round-trip default only after validation sign-off.

## Open Questions
- Should invalid quote attempts be surfaced in debug UI in non-production mode?

Note:
- The first three previously open points are now fixed by the locked decisions above to keep implementation unblocked.

## Expected Outcome
If implemented as above, the system should:
- Remove citation-specific second inference pass.
- Reduce turn latency.
- Prevent the known non-retrieval citation hang pattern.
- Keep citation trust model based on strict validation against real tool/search history.

## Definition of Done
- Stream contract is updated and type-checked end-to-end (`ChatModel` stream items, `ProcessEvent`, UI event handling).
- `Quote` union (`SearchResultQuote`, `ToolCallQuote`) and `QuoteReferenceEvent` are implemented with strict validation models.
- Inline quote parsing wrapper is integrated in composition root and covered by unit tests for chunk-boundary marker parsing.
- Parser fallback behavior is implemented: malformed quote blocks do not block streaming and are treated as plain text.
- Buffer limits are enforced for quote block accumulation and verified by tests.
- System prompt contract for marker-based quote JSON emission is implemented and documented in prompt code.
- Model-specific prompt profiles are updated to preserve marker/JSON reliability for weaker models.
- Orchestrator quote handling is implemented: validation, canonical deduplication, stable numbering, inline reference emission.
- Finalization path is implemented: validated quotes produce the markdown sources appendix and sidebar evidence elements.
- Legacy citation round-trip path is removed (or fully disabled behind a flag with default off).
- Unit tests pass for parser, orchestrator dedup/validation, and prompt contract formatting.
- Integration tests pass for retrieval-grounded, tool-grounded, and non-grounded turns.
- Regression test for the known non-retrieval GPU-hang scenario is present and passing.
- Tracing counters for quote detection/parsing/validation are emitted and visible in Phoenix.
- Performance check confirms latency improvement compared to baseline citation round-trip.
