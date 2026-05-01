# Phase 09 — CitingModel & CiteableTool: Refactoring the Citation Architecture

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
- **`CitingModel`**: decorator around `ChatModel`. Augments the prompt with
  citation instructions, parses inline citations from the text stream, and
  delivers exclusively `text`, `tool_calls`, and `RichCitation` events to the
  caller. Details in §1.3–1.5.
- **`Orchestrator`**: consumes the `CitingModel` stream, yields `text` and
  `RichCitation` to the UI, executes tool calls, and manages the history. No
  citation logic.
- **UI**: collects `RichCitation` events, deduplicates them, assigns reference
  numbers, and renders citation lists and side panels.

### 1.3 Tool-specific citation logic: `CiteableTool`

Currently, specifics of the `search_documents` tool are hardcoded in the
app-level system prompt (XML format of chunks, `chunk_id` attribute, exact JSON
form of the citation). This violates SRP in multiple ways. Proposal:

- A new `CiteableTool` interface extends `Tool` with three citation-specific
  responsibilities:
  - **Cite instructions**: the tool provides a prompt fragment describing *how*
    and *in what format* its outputs should be cited (e.g. which fields a
    `RawCitation` must contain, which `chunk_id` attributes it exposes). The
    `CitingModel` collects these fragments from all registered `CiteableTool`s
    and assembles the citation section of the system prompt from them.
  - **Tool-output rendering** for the model history: the tool decides the format
    in which its result is fed back to the model (e.g. as structured XML of
    search result chunks). This moves `_format_search_chunks_as_markdown` from
    the orchestrator into the tool itself.
  - **Validation + enrichment**: the tool receives a `RawCitation` and the
    history (i.e. `list[CitingModelMessage]` as maintained by the `Orchestrator`;
    each entry carries both the CitingModel-layer metadata and the pre-computed
    `llm_content`) and returns either `None` (invalid) or a `RichCitation` with all
    tool-specific metadata (author, title, year, page, quote text, …). Both
    decisions are tool-specific — even the structural check "does `chunk_id` X
    exist in my last result?" requires knowledge of the output schema.
- The `CitingModel` provides several citation format building blocks
  (e.g. `DocumentCitation`, `ToolCitation`) from which a `CiteableTool` selects
  the appropriate one. `search_documents` uses `DocumentCitation`; generic tools
  default to `ToolCitation`.

### 1.4 Citation encapsulation: `CitingModel` as a wrapper

The `CitingModel` is a decorator around a `ChatModel` with three clearly
separated responsibilities:

1. **Augment prompt with citation instructions**: when building the system
   prompt, the `CitingModel` injects the cite instructions of all registered
   `CiteableTool`s as well as the general marker directives (quote start/end
   markers, expected JSON schema). After this, neither the orchestrator nor the
   `Prompts` object contains any citation-specific prompt fragments.

2. **Inline citation parsing**: the text streamed by the inner `ChatModel` is
   passed through a streaming parser that detects marker tokens and extracts
   embedded JSON objects as `RawCitation` values. Each `RawCitation` carries a
   `raw_marker_text` field containing the complete marker block as emitted by
   the model (e.g. `<°_quote_°>{...}</°_quote_°>`). This field is preserved
   unchanged in the `RichCitation`. The caller receives text *without* marker
   blocks — only the visible response text.

3. **Validation + enrichment via `CiteableTool`**: for each parsed `RawCitation`
   the `CitingModel` resolves the responsible `CiteableTool` via `tool_call_id`
   and history, and calls `validate_and_enrich(raw, history)`. Only on success
   is the resulting `RichCitation` yielded to the caller; invalid citations are
   silently discarded (and logged).

The orchestrator holds exclusively a reference to a `CitingModel` and sees in
its stream only `text`, `tool_calls`, and `RichCitation` — no raw markers or
`RawCitation` objects.

- **`PromptProfile` construction in the model** (decided, see §1.5 and §2.2.d):
  `make_system_message` internalizes both the `PromptProfile` adjustment and the
  citation instruction injection; the orchestrator is decoupled from both.

### 1.5 Logical history and LLM-side representation

The orchestrator maintains a history of `CitingModelMessage` entries. Each
variant carries **both** the LLM-ready content (pre-computed at creation time)
and optional CitingModel-layer metadata alongside it:

```python
@dataclass(frozen=True)
class CitingModelAssistantMessage:
    parts: list[str | RichCitation]  # CitingModel layer: stream order preserved
    llm_content: str                 # pre-computed: text + raw_marker_text spliced in

@dataclass(frozen=True)
class CitingModelToolMessage:
    tool_call_id: str
    result: JsonObject               # CitingModel layer: raw JSON for validate_and_enrich
    llm_content: str                 # pre-computed: CiteableTool.format_for_history(result)
                                     #   or default JSON serialisation for non-CiteableTools

@dataclass(frozen=True)
class CitingModelUserMessage:
    llm_content: str                 # pass-through, no transformation

@dataclass(frozen=True)
class CitingModelSystemMessage:
    llm_content: str                 # pre-computed: base_prompt + PromptProfile adjustment + citation instructions injected

CitingModelMessage = (
    CitingModelAssistantMessage
    | CitingModelToolMessage
    | CitingModelUserMessage
    | CitingModelSystemMessage
)
```

`CitingModelAssistantMessage.parts` preserves the exact order in which text
chunks and `RichCitation`s arrived in the stream — only this allows
`raw_marker_text` to be spliced back in at the correct positions when computing
`llm_content`.

The `CitingModel` acts as the **factory** for these messages, exposing two
creation methods the orchestrator calls directly:

- `make_tool_message(tool: Tool, result: JsonObject) -> CitingModelToolMessage`:
  checks internally whether `tool` is a `CiteableTool` and calls
  `tool.format_for_history(result)` to compute `llm_content`; falls back to
  default JSON serialisation for non-`CiteableTool`s. The orchestrator never
  checks tool type itself.
- `make_assistant_message(parts: list[str | RichCitation]) -> CitingModelAssistantMessage`:
  splices `raw_marker_text` from each `RichCitation` back into the text at the
  correct position to compute `llm_content`.
- `make_system_message(base_prompt: str) -> CitingModelSystemMessage`:
  internally applies the `PromptProfile` adjustment, then injects citation
  instructions (marker directives + per-tool cite fragments) to produce
  `llm_content`. The orchestrator supplies the raw base prompt; the `CitingModel`
  owns the entire augmentation chain.

User messages are created directly by the orchestrator — they require no
CitingModel-specific transformation.

When delegating to the inner `ChatModel`, the `CitingModel` builds the
`list[ChatMessage]` trivially from `[msg.llm_content for msg in history]` — no
separate transformation pass or cache needed, since `llm_content` is
pre-computed at message creation time.

### 1.6 Data flow sketch (one turn iteration)

```
Orchestrator (history: list[CitingModelMessage])
  CitingModelAssistantMessage(parts=[str, RichCitation, ...], llm_content=...)
  CitingModelToolMessage(tool_call_id, result=json, llm_content=formatted)
  CitingModelSystemMessage(llm_content=...)  # via citing_model.make_system_message(base)
  CitingModelUserMessage(llm_content=...)    # created directly by orchestrator
     │
     │ stream(history, ...)
     ▼
CitingModel
  ├─ build list[ChatMessage] from history  (trivial: msg.llm_content per entry)
  │    (CitingModelSystemMessage.llm_content already contains citation instructions)
  └─ delegate to inner ChatModel(chat_messages)
       ▼
     ChatModel (base)  →  text (with markers) | tool_calls
       ▼
  CitingModel stream processing:
  ├─ text (markers stripped)   → yield to Orchestrator  (UI sees no markers)
  ├─ tool_calls                → yield to Orchestrator
  └─ parsed RawCitation        → CiteableTool.validate_and_enrich(raw, history)
       (raw_marker_text kept)    → RichCitation | None
                                 → yield RichCitation to Orchestrator
     ▼
Orchestrator
  ├─ text          → yield to UI; accumulate str parts
  ├─ RichCitation  → yield to UI; accumulate RichCitation parts
  └─ tool_calls    → execute tool
                     → citing_model.make_tool_message(tool, result)
                     → append CitingModelToolMessage to history
  end of turn → citing_model.make_assistant_message(accumulated_parts)
              → append CitingModelAssistantMessage to history
UI
  ├─ collect RichCitations
  ├─ dedupe / numbering
  └─ render references + side-panel
```

### 1.7 Open points (flagged by the user)

- Validation + enrichment live in `CiteableTool` (decided, see §2.2.b).
- **`PromptProfile` construction in the model** (decided, see §1.5 and §2.2.d):
  `make_system_message` internalizes the `PromptProfile` adjustment and citation
  instruction injection; the orchestrator calls only
  `model.make_system_message(prompts.system_prompt(now))`.
- **History design** (decided, see §1.5): `CitingModelMessage` carries both the
  LLM-ready content and CitingModel-layer metadata. The `CitingModel` is the
  factory; the orchestrator delegates all message creation to it and never
  inspects tool type directly.


---

## 2. Assessment & Recommendations

### 2.1 Clear wins

1. **Cleanup of `cite_sources` and the `ToolEvent` list**: straightforwardly
   the right call. Neither has proven its worth; YAGNI applies. This simplifies
   `Tool.execute` to return `JsonObject` and removes the entire `ToolEvent`
   union along with its routing code in the orchestrator.

2. **Extracting quote logic from `Prompts` into a `CitingModel`**: yes. The
   marker tokens, quote schema, streaming parser, and related prompt fragments
   form a cohesive concern. They belong behind an interface that encapsulates
   them. `Prompts` becomes a lean application-level configuration object.

3. **Extracting search-result formatting (`_format_search_chunks_as_markdown`)
   from the orchestrator**: yes. This is clearly `search_documents`-specific
   and has no business in the orchestrator.

### 2.2 Points of friction / recommendations

#### a) UI as owner of dedup & numbering — with a caveat

Conceptually clean. Practically, the current mechanism for inserting `[N]`
inline into the text stream relies on the orchestrator knowing a stable
reference number the moment a quote arrives and splicing it into the text stream
(`collected.append(f"[{quote_ref_counter}]")`). If numbering moves to the UI,
the orchestrator must instead stream a `QuoteReferenceEvent`-equivalent *without*
a reference number, and the UI must replace the placeholder token in the text
with the assigned number.
- **Recommendation**: this is feasible, but it is a real change to the stream
  protocol. We should define the exact contract (marker in text vs. separate
  event with a position index) explicitly before refactoring. Suggestion:
  orchestrator streams `RichCitation` events plus a placeholder token in the
  text; UI replaces the token. This keeps streaming order trivial and avoids
  position indices.

#### b) Validation **and** enrichment in `CiteableTool` (revised)

An earlier version proposed placing validation in a generic `CitationValidator`
alongside the `CitingModel`. The argument does not hold: even the structural
question "does `chunk_id` X exist in the result of `tool_call_id` Y?" requires
knowledge of the tool output schema. A generic validator would have to either
guess the schema or have the tool declare it — both are roundabouts for something
trivial inside the tool. There is also no reusable validation core in the
current code that would be DRY across tool boundaries.

**Design**:
- `CiteableTool.validate_and_enrich(raw_citation, history) -> RichCitation | None`
  is **the** citation method of the tool. It decides both validity and the shape
  of the enriched citation.
- The `CitingModel` handles routing: from `raw.tool_call_id` via the history it
  resolves the responsible tool, calls the method, and only forwards a
  `RichCitation` on success.
- If a shared implementation base emerges later (e.g. multiple search tools with
  an identical chunk schema), a `BaseChunkCiteableTool` can be introduced as a
  pure implementation-reuse vehicle — without changing the interface.

#### c) `CitingModel` must **not** mutate the history passed to the inner model

This is a hard line. Since the `CiteableTool` renders its own result in the
desired format, there is no reason for the `CitingModel` to alter history
entries after the fact. History is a single source of truth; post-hoc mutation
has proven to be a source of subtle bugs in this codebase.
**Recommendation**: record this explicitly as a design invariant.

#### d) `PromptProfile` construction in the model

**Decided.** The reasoning:
- `PromptProfile` is a *model-specific* strategy; instantiating it at the same
  site as the model is consistent (the model knows best what it needs).
- With `make_system_message` the `CitingModel` internalizes both the
  `PromptProfile` adjustment and citation instruction injection — the orchestrator
  no longer holds any explicit profile reference.
- **Recommendation**: move `PromptProfile` construction into the model. With
  `make_system_message` internalizing both the `PromptProfile` adjustment and
  citation instruction injection, the orchestrator calls only
  `model.make_system_message(prompts.system_prompt(now))` — no external
  `adjust_prompts` call needed.

#### e) Slimming down `ChatStreamItem` union

Today the stream carries `str | list[ToolCallInfo] | Quote | RawAssistantText`.
With the new design:
- `RawAssistantText` exists only because the orchestrator needs the raw text to
  reconstruct the history entry. With the `CitingModel` providing
  `make_assistant_message(parts)`, the orchestrator simply passes the accumulated
  `list[str | RichCitation]` parts at end of turn — `RawAssistantText` can be
  removed from the stream contract entirely.
- `Quote` (now `RawCitation`) remains — it is the central new item type.

#### f) Naming

- `QuotingModel` vs. `CitingModel`: **`CitingModel`** is clearer (quote = verbatim
  excerpt; citation = attributed reference — we are doing the latter).
- `Quote` → `RawCitation`, `validated chunk` → `RichCitation`. Unifies vocabulary
  across the entire codebase.

#### g) Suggested implementation order

1. Remove `cite_sources` tool; clean up tests and references.
2. Reduce `Tool.execute` return type to `JsonObject`; remove the `ToolEvent`
   union and all related code.
3. Introduce `CiteableTool` protocol (`cite_instructions`, `format_for_history`,
   `validate_and_enrich`); migrate `search_documents`; move
   `_format_search_chunks_as_markdown` into the tool.
4. Introduce `CitingModel`; move quote markers, parser, and quote prompt fragments
   there; remove citation content from `Prompts`.
5. `CitingModel`: routing `RawCitation` → responsible `CiteableTool` via
   `tool_call_id`/history; call `validate_and_enrich` and yield `RichCitation`.
6. Introduce `CitingModelMessage` types and `CitingModel` factory methods
   (`make_tool_message`, `make_assistant_message`, `make_system_message`);
   internalize `PromptProfile` into the model; migrate orchestrator to build
   history from these; remove `RawAssistantText` from the stream.
7. UI layer takes over dedup and reference numbering; orchestrator streams only
   validated `RichCitation` events plus a placeholder token in the text.
Each step is independently testable and does not change observable behaviour.

### 2.3 Risks / points that need discussion before implementation

1. **Stream protocol change for UI numbering** (see §2.2.a). This is the only
   place where a non-trivial breaking contract change is required.
2. **Who owns the tool map?** Since the `CitingModel` performs Raw→Rich
   transformation, it needs read access to the tool map and history. This is a
   deliberate coupling of the `CitingModel` to the tool space and should be
   modelled as an explicit contract (e.g. via a slim `CiteResolutionPort`, not a
   direct orchestrator dependency).
### 2.4 Recommended design in one sentence

> `CiteableTool` renders, describes, validates, and enriches its own citations;
> `CitingModel` encapsulates quote markers, parsing, prompts, and the
> Raw→Rich transformation via `CiteableTool`; the orchestrator orchestrates
> tool execution and streaming; the UI handles dedup, numbering, and rendering.
