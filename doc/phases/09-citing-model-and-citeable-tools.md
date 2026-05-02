# Phase 09 — CitationLayer & CiteableTool: Refactoring the Citation Architecture

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
- **`CitationLayer`**: decorator around `ChatModel`. Augments the prompt with
  citation instructions, parses inline citations from the text stream, and
  delivers `text`, `tool_calls`, `Citation`, and `HallucinatedCitation`
  events to the caller. Details in §1.3–1.5.
- **`Orchestrator`**: consumes the `CitationLayer` stream, executes tool calls,
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
    the `CitationLayer`; `cite_instructions` selects the subtype and defines the
    field mapping. The `CitationLayer` collects these fragments from all registered
    `CiteableTool`s and assembles the citation section of the system prompt from them.
  - **Tool-output rendering** for the model history: the tool decides the format
    in which its result is fed back to the model (e.g. as structured XML of
    search result chunks). This moves `_format_search_chunks_as_markdown` from
    the orchestrator into the tool itself.
  - **Validation + enrichment**: the tool receives a `RawCitation` and the
    history (i.e. `list[CitationLayerMessage]` as maintained by the `Orchestrator`;
    each entry carries both the citation-layer metadata and the pre-computed
    `llm_content`) and returns either `None` (invalid) or a `Citation` with all
    tool-specific metadata (author, title, year, page, quote text, …). Both
    decisions are tool-specific — even the structural check "does `chunk_id` X
    exist in my last result?" requires knowledge of the output schema.
- The `CitationLayer` defines the citation format vocabulary as paired typed
  subtypes: a `RawCitation` subtype (produced by the parser from the marker
  JSON, e.g. `DocumentRawCitation`) and a corresponding `Citation` subtype
  (returned by `validate_and_enrich`, e.g. `DocumentCitation`). Each
  `CiteableTool` selects the appropriate pair via `cite_instructions`:
  `search_documents` uses the `Document` pair; generic tools use the `Tool` pair.

### 1.4 Citation encapsulation: `CitationLayer` as a wrapper

The `CitationLayer` is a decorator around a `ChatModel` with three clearly
separated responsibilities:

1. **Augment prompt with citation instructions**: when building the system
   prompt, the `CitationLayer` injects the cite instructions of all registered
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
   the `CitationLayer` looks up the `CitationLayerToolMessage` with matching
   `tool_call_id` in the history, reads its `tool_name`, resolves the
   `CiteableTool` from the tool registry by name, and calls
   `validate_and_enrich(raw, history)`. On success the resulting `Citation` is
   yielded to the caller; on failure a `HallucinatedCitation` (carrying the
   `RawCitation` and the rejection reason) is yielded instead — the decision of
   whether and how to surface it is delegated to the UI.

The orchestrator holds exclusively a reference to a `CitationLayer` and sees in
its stream only `text`, `tool_calls`, `Citation`, and `HallucinatedCitation` —
no raw markers or `RawCitation` objects.

- **`PromptProfile` construction in the model** (decided, see §1.5 and §2.2.d):
  `make_system_message` internalizes both the `PromptProfile` adjustment and the
  citation instruction injection; the orchestrator is decoupled from both.

### 1.5 Logical history and LLM-side representation

The orchestrator maintains a history of `CitationLayerMessage` entries. Each
variant carries **both** the LLM-ready content (pre-computed at creation time)
and optional citation-layer metadata alongside it:

```python
@dataclass(frozen=True)
class CitationLayerAssistantMessage:
    parts: list[str | Citation | HallucinatedCitation]  # CitationLayer layer: stream order preserved
    llm_content: str                 # pre-computed: text + raw_marker_text spliced in

@dataclass(frozen=True)
class CitationLayerToolMessage:
    tool_call_id: str
    tool_name: str                   # CitationLayer layer: stored by make_tool_message for routing
    result: JsonObject               # CitationLayer layer: raw JSON for validate_and_enrich
    llm_content: str                 # pre-computed: CiteableTool.format_for_history(result)
                                     #   or default JSON serialisation for non-CiteableTools

@dataclass(frozen=True)
class CitationLayerUserMessage:
    llm_content: str                 # pre-computed: citation reminder injected + Prompts.user_message(user_text)

@dataclass(frozen=True)
class CitationLayerSystemMessage:
    llm_content: str                 # pre-computed: base_prompt + PromptProfile adjustment + citation instructions injected

CitationLayerMessage = (
    CitationLayerAssistantMessage
    | CitationLayerToolMessage
    | CitationLayerUserMessage
    | CitationLayerSystemMessage
)
```

`CitationLayerAssistantMessage.parts` preserves the exact order in which text
chunks, `Citation`s, and `HallucinatedCitation`s arrived in the stream — only this allows
`raw_marker_text` to be spliced back in at the correct positions when computing
`llm_content`.

The `CitationLayer` acts as the **factory** for these messages, exposing four
creation methods the orchestrator calls directly:

- `make_tool_message(tool: Tool, result: JsonObject) -> CitationLayerToolMessage`:
  stores `tool.name` in the message for later routing, checks internally whether
  `tool` is a `CiteableTool` and calls `tool.format_for_history(result)` to
  compute `llm_content`; falls back to default JSON serialisation for
  non-`CiteableTool`s. The orchestrator never checks tool type itself.
- `make_assistant_message(parts: list[str | Citation | HallucinatedCitation]) -> CitationLayerAssistantMessage`:
  splices `raw_marker_text` from each `Citation` or `HallucinatedCitation` back into the text at the
  correct position to compute `llm_content`.
- `make_system_message(base_prompt: str) -> CitationLayerSystemMessage`:
  internally applies the `PromptProfile` adjustment, then injects citation
  instructions (marker directives + per-tool cite fragments) to produce
  `llm_content`. The orchestrator supplies the raw base prompt; the `CitationLayer`
  owns the entire augmentation chain.
- `make_user_message(user_text: str) -> CitationLayerUserMessage`:
  injects the citation reminder (marker tokens, per-sentence rules, `!UNBELEGT!`
  obligation) and then delegates to `Prompts.user_message(user_text)` for any
  application-level framing. `Prompts.user_message` is a pure pass-through by
  default; model-specific configurations may add further wrapping without
  touching citation concerns.

When delegating to the inner `ChatModel`, the `CitationLayer` builds the
`list[ChatMessage]` trivially from `[msg.llm_content for msg in history]` — no
separate transformation pass or cache needed, since `llm_content` is
pre-computed at message creation time.

### 1.6 Data flow sketch (one turn iteration)

```
Orchestrator (history: list[CitationLayerMessage])
  CitationLayerAssistantMessage(parts=[str, Citation, ...], llm_content=...)
  CitationLayerToolMessage(tool_call_id, result=json, llm_content=formatted)
  CitationLayerSystemMessage(llm_content=...)  # via citation_layer.make_system_message(base)
  CitationLayerUserMessage(llm_content=...)    # via citation_layer.make_user_message(prompts.user_message(user_text))
     │
     │ stream(history, ...)
     ▼
CitationLayer
  ├─ build list[ChatMessage] from history  (trivial: msg.llm_content per entry)
  │    (CitationLayerSystemMessage.llm_content already contains citation instructions)
  └─ delegate to inner ChatModel(chat_messages)
       ▼
     ChatModel (base)  →  text (with markers) | tool_calls
       ▼
  CitationLayer stream processing:
  ├─ text (markers stripped)   → yield to Orchestrator  (UI sees no markers)
  ├─ tool_calls                → yield to Orchestrator
  └─ parsed RawCitation        → CiteableTool.validate_and_enrich(raw, history)
       (raw_marker_text kept)    → Citation: yield Citation to Orchestrator
                                 → None:    yield HallucinatedCitation(raw, reason) to Orchestrator
     ▼
Orchestrator
  ├─ text          → yield to UI; accumulate str parts
  ├─ Citation             → assign ref_number (new, or reuse if seen before)
  │                         → yield NumberedCitation to UI; accumulate in parts
  ├─ HallucinatedCitation → yield to UI as-is; accumulate in parts
  └─ tool_calls    → execute tool
                     → citation_layer.make_tool_message(tool, result)
                     → append CitationLayerToolMessage to history
  end of turn → citation_layer.make_assistant_message(accumulated_parts)
              → append CitationLayerAssistantMessage to history
UI (app.py)
  ├─ text chunks + NumberedCitation events (in stream order)
  │    → render [N] inline at each NumberedCitation position
  ├─ HallucinatedCitation → product decision: hide / [?] placeholder / warning
  └─ build citation list / side-panel from NumberedCitation events
```

---

## 2. Assessment & Recommendations

### 2.1 Clear wins

1. **Cleanup of `cite_sources` and the `ToolEvent` list**: straightforwardly
   the right call. Neither has proven its worth; YAGNI applies. This simplifies
   `Tool.execute` to return `JsonObject` and removes the entire `ToolEvent`
   union along with its routing code in the orchestrator.

2. **Extracting quote logic from `Prompts` into a `CitationLayer`**: yes. The
   marker tokens, quote schema, streaming parser, and related prompt fragments
   form a cohesive concern. They belong behind an interface that encapsulates
   them. `Prompts` becomes a lean application-level configuration object.

3. **Extracting search-result formatting (`_format_search_chunks_as_markdown`)
   from the orchestrator**: yes. This is clearly `search_documents`-specific
   and has no business in the orchestrator.

### 2.2 Points of friction / recommendations

#### a) Numbering ownership — decided

**Decided.** The `Orchestrator` owns the numbering: it converts each incoming
`Citation` to a `NumberedCitation`, assigning a new `ref_number` on first
occurrence and reusing the existing one for duplicates (same `Citation`
identity/equality). It yields `NumberedCitation` events in stream order — no
placeholder tokens or position indices needed. The UI (app.py) renders `[N]`
inline at each `NumberedCitation` position in the stream and builds the citation
list and side panel from the `NumberedCitation` events.

#### b) Validation **and** enrichment in `CiteableTool` (revised)

An earlier version proposed placing validation in a generic `CitationValidator`
alongside the `CitationLayer`. The argument does not hold: even the structural
question "does `chunk_id` X exist in the result of `tool_call_id` Y?" requires
knowledge of the tool output schema. A generic validator would have to either
guess the schema or have the tool declare it — both are roundabouts for something
trivial inside the tool. There is also no reusable validation core in the
current code that would be DRY across tool boundaries.

**Design**:
- `CiteableTool.validate_and_enrich(raw_citation, history) -> Citation | None`
  is **the** citation method of the tool. It decides both validity and the shape
  of the enriched citation.
- The `CitationLayer` handles routing: it looks up the `CitationLayerToolMessage`
  with matching `tool_call_id` in the history, reads `tool_name`, resolves the
  `CiteableTool` from the tool registry by name, calls the method; forwards a `Citation` on success or a
  `HallucinatedCitation` (carrying the `RawCitation` and rejection reason) on
  failure.
- If a shared implementation base emerges later (e.g. multiple search tools with
  an identical chunk schema), a `BaseChunkCiteableTool` can be introduced as a
  pure implementation-reuse vehicle — without changing the interface.

#### c) `CitationLayer` must **not** mutate the history passed to the inner model

This is a hard line. Since the `CiteableTool` renders its own result in the
desired format, there is no reason for the `CitationLayer` to alter history
entries after the fact. History is a single source of truth; post-hoc mutation
has proven to be a source of subtle bugs in this codebase.
**Recommendation**: record this explicitly as a design invariant.

#### d) `PromptProfile` construction in the model

**Decided.** The reasoning:
- `PromptProfile` is a *model-specific* strategy; instantiating it at the same
  site as the model is consistent (the model knows best what it needs).
- With `make_system_message` the `CitationLayer` internalizes both the
  `PromptProfile` adjustment and citation instruction injection — the orchestrator
  no longer holds any explicit profile reference.
- **Recommendation**: move `PromptProfile` construction into the model. With
  `make_system_message` and `make_user_message` internalizing all citation
  injection, the orchestrator calls only
  `model.make_system_message(prompts.system_prompt(now))` and
  `model.make_user_message(prompts.user_message(user_text))` — no external
  `adjust_prompts` call needed.

#### e) Slimming down `ChatStreamItem` union

Today the stream carries `str | list[ToolCallInfo] | Quote | RawAssistantText`.
With the new design:
- `RawAssistantText` exists only because the orchestrator needs the raw text to
  reconstruct the history entry. With the `CitationLayer` providing
  `make_assistant_message(parts)`, the orchestrator simply passes the accumulated
  `list[str | Citation | HallucinatedCitation]` parts at end of turn — `RawAssistantText` can be
  removed from the stream contract entirely.
- `Quote` (now `RawCitation`) remains — it is the central new item type.

#### f) Naming

- `QuotingModel` vs. `CitationLayer`: **`CitationLayer`** is clearer (quote = verbatim
  excerpt; citation = attributed reference — we are doing the latter).
- `Quote` → `RawCitation`, `validated chunk` → `Citation`,
  `validated chunk with ref number` → `NumberedCitation`. Unifies vocabulary
  across the entire codebase.

#### g) Suggested implementation order

1. Remove `cite_sources` tool; clean up tests and references.
2. Reduce `Tool.execute` return type to `JsonObject`; remove the `ToolEvent`
   union and all related code.
3. Introduce `CiteableTool` protocol (`cite_instructions`, `format_for_history`,
   `validate_and_enrich`); migrate `search_documents`; move
   `_format_search_chunks_as_markdown` into the tool.
4. Introduce `CitationLayer`; move quote markers, parser, and quote prompt fragments
   there; remove citation content from `Prompts`.
5. `CitationLayer`: routing `RawCitation` → responsible `CiteableTool` via
   `tool_call_id`/history; call `validate_and_enrich` and yield `Citation`.
6. Introduce `CitationLayerMessage` types and `CitationLayer` factory methods
   (`make_tool_message`, `make_assistant_message`, `make_system_message`,
   `make_user_message`); internalize `PromptProfile` and citation reminder into
   the model; migrate orchestrator to build history from these; remove
   `RawAssistantText` from the stream.
7. Orchestrator converts `Citation` → `NumberedCitation` (assign/reuse
   `ref_number`); UI (app.py) renders `[N]` inline and builds citation list.
Each step is independently testable and does not change observable behaviour.

### 2.3 Risks / points that need discussion before implementation

1. **Who owns the tool map?** Since the `CitationLayer` performs Raw→`Citation`
   transformation, it needs read access to the tool map and history. This is a
   deliberate coupling of the `CitationLayer` to the tool space and should be
   modelled as an explicit contract (e.g. via a slim `CiteResolutionPort`, not a
   direct orchestrator dependency).
### 2.4 Recommended design in one sentence

> `CiteableTool` renders, describes, validates, and enriches its own citations;
> `CitationLayer` encapsulates quote markers, parsing, prompts, and the
> Raw→`Citation` transformation via `CiteableTool`; the `Orchestrator`
> orchestrates tool execution, manages history, and converts `Citation` →
> `NumberedCitation`; the UI renders `[N]` markers and the citation list.
