# Phase 09 — CitationModel & CiteableTool: Refactoring the Citation Architecture

> Status: Draft proposal for discussion. No implementation before explicit sign-off.

## 1. Concept (as described by the user)

### 1.1 Cleanup

- The former `cite_sources` tool is removed without replacement. Inline quotes
  have proven to be the superior mechanism.
- The mechanism by which a `Tool` can return a list of `ToolEvent` alongside its
  JSON result is also removed. Tools return only their JSON result. Should this
  mechanism ever be needed again, it can be reintroduced later (YAGNI).

### 1.2 Responsibilities re-assigned (SRP)

Components are divided along the data flow:

- **`ChatModel`** (e.g. Ollama adapter): receives history and prompts, emits
  text chunks and tool calls. No citation concept.
- **`CitationModel`**: decorator around `ChatModel`. Augments the prompt with
  citation instructions, parses inline citations from the text stream, and
  delivers `text`, `tool_calls`, `Citation`, and `HallucinatedCitation`
  events to the caller. Details in §1.3–1.5.
- **`Orchestrator`**: consumes the `CitationModel` stream, executes tool calls,
  and manages the history. Converts each incoming `Citation` to a
  `NumberedCitation` (assigning a new `ref_number` on first occurrence, reusing
  the existing one for duplicates) and yields `text`, `NumberedCitation`, and `HallucinatedCitation` to the UI.
  No citation logic beyond numbering.
- **UI**: receives `NumberedCitation`, `HallucinatedCitation`, and `text` chunks.
  Renders `[N]` markers inline at each `NumberedCitation` position; decides
  whether and how to surface `HallucinatedCitation` events (hide, placeholder
  `[?]`, warning, A/B-testable, etc.). Builds the citation list and side panel
  from the numbered citations.

### 1.3 Tool-specific citation logic: `CiteableTool`

Currently, specifics of the `search_documents` tool are hardcoded in the
app-level system prompt (XML format of chunks, `chunk_id` attribute, exact JSON
form of the citation). This violates SRP in multiple ways. Proposal:

- A new `CiteableTool` interface extends `Tool` with three citation-specific
  responsibilities:
  - **Cite instructions**: the tool tells the model which `RawCitation` subtype
    to use for its citations (e.g. `DocumentRawCitation` vs `ToolRawCitation`)
    and how to map fields from the tool result into that subtype's required
    fields (e.g. which field in the tool's result data serves as the `chunk_id`
    in a `DocumentRawCitation`). The JSON schema inside the markers is owned by
    the `CitationModel`; `cite_instructions` selects the subtype and defines the
    field mapping. The `CitationModel` collects these fragments from all registered
    `CiteableTool`s and assembles the citation section of the system prompt from them.
  - **Tool-output rendering** for the model history: the tool decides the format
    in which its result is fed back to the model (e.g. as structured XML of
    search result chunks). This moves `_format_search_chunks_as_markdown` from
    the orchestrator into the tool itself.
  - **Validation + enrichment**: the tool receives a `RawCitation` and a
    narrow read-only `CitationContext` (see §1.4a) and returns either `None`
    (invalid) or a `Citation` with all tool-specific metadata (author, title,
    year, page, quote text, …). Both decisions are tool-specific — even the
    structural check "does `chunk_id` X exist in my last result?" requires
    knowledge of the output schema. Tools never see the `CitationModel`'s
    internal history representation; the `CitationContext` Protocol is the only
    coupling point and lives in `app/` so `tools/` does not import
    citation-layer internals.
- The `CitationModel` defines the citation format vocabulary as paired typed
  subtypes: a `RawCitation` subtype (produced by the parser from the marker
  JSON, e.g. `DocumentRawCitation`) and a corresponding `Citation` subtype
  (returned by `validate_and_enrich`, e.g. `DocumentCitation`). Each
  `CiteableTool` selects the appropriate pair via `cite_instructions`:
  `search_documents` uses the `Document` pair; generic tools use the `Tool` pair.

### 1.4 Citation encapsulation: `CitationModel` as a wrapper

The `CitationModel` is a decorator around a `ChatModel` with three clearly
separated responsibilities:

1. **Augment prompt with citation instructions**: when building the system
   prompt, the `CitationModel` injects the cite instructions of all registered
   `CiteableTool`s as well as the general marker directives (quote start/end
   markers, expected JSON schema). After this, `Prompts` contains no
   citation-specific prompt fragments — `Prompts.user_message` in particular
   becomes a pure pass-through (optional application-level framing only).

2. **Inline citation parsing**: the text streamed by the inner `ChatModel` is
   passed through a streaming parser that detects marker tokens and instantiates
   typed `RawCitation` subtypes from the embedded JSON (e.g.
   `DocumentRawCitation`, `ToolRawCitation`). The concrete subtype is determined
   by the format the responsible `CiteableTool` declared in `cite_instructions`.
   Each `RawCitation` carries a `raw_marker_text` field containing the complete
   marker block as emitted by the model (e.g. `<°_quote_°>{...}</°_quote_°>`).
   Marker blocks are replaced in-place by `RawCitation` events in the
   `str | RawCitation` stream — surrounding text chunks contain no marker syntax.

3. **Validation + enrichment via `CiteableTool`**: for each parsed `RawCitation`
   the `CitationModel` looks up the `CitationModelToolMessage` with matching
   `tool_call_id` in the history, reads its `tool_name`, resolves the
   `CiteableTool` from the tool registry by name, and calls
   `validate_and_enrich(raw, history)`. On success the resulting `Citation` is
   yielded to the caller; on failure a `HallucinatedCitation` (carrying the
   `RawCitation` and the rejection reason) is yielded instead — the decision of
   whether and how to surface it is delegated to the UI.

The orchestrator holds exclusively a reference to a `CitationModel` and sees in
its stream only `text`, `tool_calls`, `Citation`, and `HallucinatedCitation` —
no raw markers or `RawCitation` objects.

**Composition (decided).** `CitationModel` is constructed at composition time
with an explicit `list[CiteableTool]` plus the inner `ChatModel`. The same
`CiteableTool` instances are also registered with the `Orchestrator` as regular
`Tool`s for dispatch — there is one source of truth (the composition root
builds both lists from the same instances). The `CitationModel` never inspects
the orchestrator's tool registry, and the orchestrator never inspects the
citation registry.

**`PromptProfile` placement (decided).** The `PromptProfile` stays with the
`Orchestrator` as today — it adjusts both prompts and tool schemas, neither of
which is a citation concern. Per turn, the orchestrator computes the
profile-adjusted base system prompt and hands it to
`citation_layer.make_system_message(adjusted_base_prompt)`. The `CitationModel`
appends only citation instructions and is unaware of `PromptProfile`.

### 1.4a `CitationContext`: the only coupling between tools and the citation layer

```python
@runtime_checkable
class CitationContext(Protocol):
    """Read-only, narrow view passed to ``CiteableTool.validate_and_enrich``.

    Defined in ``app/`` so ``tools/`` need not import any citation-layer
    internals. The default implementation is a thin adapter over the
    orchestrator's ``list[CitationMessage]``; tests can supply trivial
    fakes.
    """

    def tool_result_for(self, tool_call_id: str) -> JsonObject | None:
        """Return the raw JSON result for a specific prior tool call."""
        ...

    def tool_results_for(self, tool_name: str) -> Sequence[JsonObject]:
        """Return all prior tool-result JSON objects emitted by *tool_name*,
        in chronological (oldest-first) order. Used for fallbacks such as
        "is this chunk_id globally unique across all my prior results?"."""
        ...
```

The `CitationModel` constructs the context once per validation call from its
internal history and passes it to the resolved `CiteableTool`. Tools never see
`CitationMessage` directly.

### 1.5 Logical history and LLM-side representation

The orchestrator maintains a history of `CitationMessage` entries. Each
variant carries **both** the LLM-ready content (pre-computed at creation time)
and optional citation-layer metadata alongside it:

```python
@dataclass(frozen=True)
class CitationModelAssistantMessage:
    # Stream-order log of the assistant turn; preserved so raw_marker_text can
    # be spliced back into ``llm_content`` at the correct positions.
    parts: list[str | Citation | HallucinatedCitation]
    # Pre-computed at message creation: plain text with each Citation /
    # HallucinatedCitation replaced by its original ``raw_marker_text``.
    # Rationale: feeding the model its own ``[N]`` reference numbers in
    # subsequent turns proved to confuse it (observed during Phase 7); the raw
    # marker block is the form the model itself produced and works reliably.
    llm_content: str

@dataclass(frozen=True)
class CitationModelToolMessage:
    tool_call_id: str
    tool_name: str                   # stored by make_tool_message for later routing
    result: JsonObject               # raw JSON for validate_and_enrich
    # Pre-computed: ``CiteableTool.format_for_history(result)`` if the tool is
    # a ``CiteableTool``, else default JSON serialisation.
    llm_content: str

@dataclass(frozen=True)
class CitationModelUserMessage:
    # Pre-computed: citation reminder + ``Prompts.user_message(user_text)``,
    # in the same order and form as today's combined output (see §1.5a).
    llm_content: str

@dataclass(frozen=True)
class CitationModelSystemMessage:
    # Pre-computed: the *already profile-adjusted* base prompt supplied by the
    # orchestrator, with citation instructions appended. The CitationModel is
    # unaware of ``PromptProfile``.
    llm_content: str

CitationMessage = (
    CitationModelAssistantMessage
    | CitationModelToolMessage
    | CitationModelUserMessage
    | CitationModelSystemMessage
)
```

`CitationModelAssistantMessage.parts` preserves the exact order in which text
chunks, `Citation`s, and `HallucinatedCitation`s arrived in the stream — only this allows
`raw_marker_text` to be spliced back in at the correct positions when computing
`llm_content`. **`HallucinatedCitation.raw_marker_text` is also spliced back into
`llm_content`** (decided): keeping the assistant's own marker text in subsequent
turns is consistent with how validated `Citation`s are handled and matches
today's behavior. If this turns out to reinforce hallucinations in practice, it
can be revisited — explicitly out of scope for this refactor.

The `CitationModel` acts as the **factory** for these messages, exposing four
creation methods the orchestrator calls directly:

- `make_tool_message(tool_call_id: str, tool_name: str, result: JsonObject) -> CitationModelToolMessage`:
  looks up `tool_name` in the `CitationModel`'s `CiteableTool` registry; if
  found, calls `format_for_history(result)` to compute `llm_content`,
  otherwise falls back to default JSON serialisation. The orchestrator passes
  primitives only and never holds a `CiteableTool` reference for this purpose.
- `make_assistant_message(parts: list[str | Citation | HallucinatedCitation]) -> CitationModelAssistantMessage`:
  splices `raw_marker_text` from each `Citation` or `HallucinatedCitation` back
  into the text at the correct position to compute `llm_content`.
- `make_system_message(adjusted_base_prompt: str) -> CitationModelSystemMessage`:
  receives an *already profile-adjusted* base prompt from the orchestrator and
  appends citation instructions (marker directives + per-tool cite fragments)
  to produce `llm_content`. The `CitationModel` is unaware of `PromptProfile`;
  the orchestrator continues to own profile adjustment for prompts and tool
  schemas alike.
- `make_user_message(user_text: str) -> CitationModelUserMessage`: see §1.5a.

When delegating to the inner `ChatModel`, the `CitationModel` builds the
`list[ChatMessage]` trivially from `[msg.llm_content for msg in history]` — no
separate transformation pass or cache needed, since `llm_content` is
pre-computed at message creation time.

### 1.5a `make_user_message`: behavior-preserving ordering

Goal of this refactor is **clean separation of concerns, not a behavior
change**. Today's `Prompts.user_message` produces `<reminder text> + user_text`
in a single string. After the refactor:

- `Prompts.user_message` becomes a pure pass-through by default (returns
  `user_text` unchanged); it remains a hook for application-level framing only.
- `CitationModel.make_user_message(user_text)` produces
  `<reminder text> + Prompts.user_message(user_text)` — the reminder block is
  identical in content and position to today's, and `Prompts.user_message` no
  longer carries citation concerns. End-to-end the wire format equals today's.

Any model-specific framing later added to `Prompts.user_message` will wrap the
user text *without* duplicating the citation reminder.

### 1.5b Validation runs per `RawCitation`, dedup happens in the orchestrator

Decided: the `CitationModel` calls `validate_and_enrich` for every parsed
`RawCitation`, even when the model emits the same marker repeatedly. The
orchestrator deduplicates `Citation`s by canonical key when assigning
`ref_number`. Trade-off: a small amount of redundant validation work in
exchange for keeping the citation layer simple (no canonical-key knowledge,
no cross-tool dedup policy). Currently `validate_and_enrich` is
local/in-memory work; if a future `CiteableTool` performs expensive
validation (e.g. network calls), it should cache internally.

### 1.6 Data flow sketch (one turn iteration)

```
Orchestrator (history: list[CitationMessage])
  CitationModelAssistantMessage(parts=[str, Citation, ...], llm_content=...)
  CitationModelToolMessage(tool_call_id, tool_name, result, llm_content)
  CitationModelSystemMessage(llm_content)  # citation_layer.make_system_message(profile_adjusted_base_prompt)
  CitationModelUserMessage(llm_content)    # citation_layer.make_user_message(user_text)
     │
     │ stream(history, ...)
     ▼
CitationModel
  ├─ build list[ChatMessage] from history  (trivial: msg.llm_content per entry)
  │    (CitationModelSystemMessage.llm_content already contains citation instructions)
  └─ delegate to inner ChatModel(chat_messages)
       ▼
     ChatModel (base)  →  text (with markers) | tool_calls
       ▼
  CitationModel stream processing:
  ├─ text (markers stripped)   → yield to Orchestrator  (UI sees no markers)
  ├─ tool_calls                → yield to Orchestrator
  └─ parsed RawCitation        → CiteableTool.validate_and_enrich(raw, ctx)
       (raw_marker_text kept)    where ctx is a CitationContext built from history
                                 → Citation: yield Citation to Orchestrator
                                 → None:    yield HallucinatedCitation(raw, reason) to Orchestrator
     ▼
Orchestrator
  ├─ text          → yield to UI; accumulate str parts
  ├─ Citation             → assign ref_number (new, or reuse if seen before)
  │                         → yield NumberedCitation to UI; accumulate in parts
  ├─ HallucinatedCitation → yield to UI as-is; accumulate in parts
  └─ tool_calls    → execute tool
                     → citation_layer.make_tool_message(tc.call_id, tc.name, result)
                     → append CitationModelToolMessage to history
  end of turn → citation_layer.make_assistant_message(accumulated_parts)
              → append CitationModelAssistantMessage to history
UI (app.py)
  ├─ text chunks + NumberedCitation events (in stream order)
  │    → render [N] inline at each NumberedCitation position
  ├─ HallucinatedCitation → product decision: hide / [?] placeholder / warning
  └─ build citation list / side-panel from NumberedCitation events
```

## 2. Test Migration Plan

Goal: preserve current behavior coverage while moving each test to the layer
that now owns the responsibility under test. No coverage regression.

### 2.1 Files to delete

- `tests/unit/test_citation_tool.py` — the `cite_sources` tool is removed (§1.1).
  Any still-relevant assertions (e.g. "validation rejects unknown chunk_id")
  migrate to `test_citeable_retrieval_tool.py` (§2.3).

### 2.2 Files to rename / refocus

- `tests/unit/test_inline_quote_parser.py` →
  `tests/unit/test_citation_layer_parser.py`. Same parser cases (marker
  detection, partial chunks across feeds, malformed JSON, oversized blocks);
  the parser is now an internal of the `CitationModel` rather than a standalone
  chat-model wrapper. Tests target the streaming parse function, not the
  wrapper class.
- `tests/unit/test_quote_models.py` →
  `tests/unit/test_raw_citation_models.py`. `SearchResultQuote`/`ToolCallQuote`
  become `DocumentRawCitation`/`ToolRawCitation`. Field-validation tests carry
  over 1:1 with the renamed classes.
- `tests/unit/test_orchestrator.py` — keep the file, narrow its scope to:
  tool-call dispatch loop, repeat-call detection, max-step fallback, and
  `Citation → NumberedCitation` numbering (new vs. reuse). Move all
  validation-of-quotes-against-history assertions out (see §2.3, §2.4).
- `tests/unit/test_ui_citation_view.py` — update to consume
  `NumberedCitation` and `HallucinatedCitation` instead of
  `QuoteReferenceEvent` + `SourceCitationEvent`/`ToolCitationEvent`. Add cases
  for the UI's `HallucinatedCitation` rendering decision (default: hide; assert
  no placeholder unless explicitly enabled).
- `tests/unit/test_chat_prompt_profiles.py` — unchanged. `PromptProfile`
  remains an orchestrator concern (§1.4).

### 2.3 New test files (per-tool citation behavior)

One file per `CiteableTool`, exercising its three citation responsibilities in
isolation against a fake `CitationContext`:

- `tests/unit/test_citeable_retrieval_tool.py`
  - `cite_instructions()` returns the expected `RawCitation` subtype and field
    mapping; the rendered prompt fragment matches a golden string.
  - `format_for_history()` produces the same XML output as today's
    `_format_search_chunks_as_markdown` (golden file from current behavior).
  - `validate_and_enrich()`: positive cases (exact match,
    normalized-source match, globally-unique chunk_id fallback) and negative
    cases (unknown source, unknown chunk_id, ambiguous chunk_id), all driven
    by a `FakeCitationContext`.
- `tests/unit/test_citeable_vacation_days_tool.py`
  - Same three responsibilities; the validation reduces to "matching
    `tool_call_id` exists".

### 2.4 New test files (citation-layer end-to-end)

- `tests/unit/test_citation_layer.py` — drives `CitationModel` against a fake
  `ChatModel` and a small set of `CiteableTool`s:
  - System-prompt augmentation: assemble fragments from registered tools and
    inject after the orchestrator-supplied (already profile-adjusted) base.
  - Streaming: text chunks pass through marker-free; `RawCitation` is parsed
    and routed to the correct `CiteableTool` by `tool_name` resolved from the
    `CitationModelToolMessage` matching `tool_call_id`.
  - Validation outcomes: success → `Citation` event; failure → `HallucinatedCitation`
    with rejection reason; **duplicate raw markers re-trigger validation**
    (documenting §1.5b).
  - Factory methods: `make_tool_message` falls back to JSON serialization for
    non-`CiteableTool` names; `make_assistant_message` splices
    `raw_marker_text` for both `Citation` and `HallucinatedCitation`;
    `make_user_message` reproduces today's reminder ordering byte-for-byte
    against a golden string (§1.5a).
- `tests/unit/test_citation_context.py` — the default `CitationContext`
  adapter over `list[CitationMessage]`:
  - `tool_result_for(call_id)` returns the matching tool message's `result`
    or `None`.
  - `tool_results_for(tool_name)` returns chronological order across multiple
    turns.

### 2.5 Integration test

- `tests/integration/test_rag.py` — keep as the behavioral baseline. Add one
  assertion that a deliberately fabricated `chunk_id` in the model output
  surfaces as a `HallucinatedCitation` event (instead of being silently
  dropped, as today). The happy path's wire-format output to the UI must
  remain identical to current behavior.

### 2.6 Coverage diff (informal checklist)

| Behavior under test                          | Today                          | After refactor                         |
|---------------------------------------------|--------------------------------|----------------------------------------|
| Marker stream parsing                        | `test_inline_quote_parser.py`  | `test_citation_layer_parser.py`        |
| Quote payload schema validation              | `test_quote_models.py`         | `test_raw_citation_models.py`          |
| Search-quote validation against history      | `test_orchestrator.py` (mixed) | `test_citeable_retrieval_tool.py`      |
| Tool-quote validation against history        | `test_orchestrator.py` (mixed) | `test_citeable_vacation_days_tool.py`  |
| Numbering / dedup of validated citations     | `test_orchestrator.py`         | `test_orchestrator.py` (kept, narrowed)|
| Tool-output formatting for model history     | implicit in orchestrator tests | `test_citeable_retrieval_tool.py`      |
| System-prompt assembly                        | `test_chat_prompt_profiles.py` | unchanged + `test_citation_layer.py`   |
| Hallucinated citations surfaced              | (none — silently dropped)      | `test_citation_layer.py` + integration |
| UI rendering of references and side panel    | `test_ui_citation_view.py`     | `test_ui_citation_view.py` (updated)   |
