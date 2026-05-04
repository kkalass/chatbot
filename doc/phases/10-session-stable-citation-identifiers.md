# Phase 10 — Session-Stable Citation Identifiers

> Status: Design signed off. Ready for implementation.

## 1. Goal

Make citation reference numbers stable across the whole chat session, not only
within one request/turn.

Current behavior:
- `NumberedCitation.reference_number` is assigned per turn by `_CitationNumberer`,
  a dedicated class instantiated fresh inside `_gen()` on each `process_message()` call.
- Numbering restarts at `1` for every new user request.
- This works for a single answer but is confusing in long sessions, especially
  when UI surfaces aggregate citations from multiple answers.

Target behavior:
- Reference numbers are session-stable and monotonic.
- The same evidence item reuses its existing number across turns.
- New evidence gets the next free number.

## 2. Scope and non-scope

In scope:
- Identity model (`citation -> stable session key`)
- Number allocation lifecycle in orchestrator/session state
- Backward-compatible changes to UI rendering interfaces
- Making plain-tool `citation_token`s deterministic (prerequisite for stable identity)

Out of scope for this phase:
- UX redesign of the sidebar
- Persistence across app restarts (this proposal is session-memory only)

## 3. Identity model

Session-stable numbering relies on session-stable `citation_token` values.
The existing `canonical_key()` function already uses `citation_token` as
the deduplication key (`document:{citation_token}` / `tool:{citation_token}`),
so no key-format change is needed — only the token generation for plain tools
and the numberer scope must change.

### 3.1 DocumentCitation key

No change needed. `RetrievalTool` already sets `citation_token = chunk.chunk_id`,
a content-derived hash assigned at ingestion time. The key is therefore identical
across all turns that retrieve the same chunk:

```
document:{chunk_id}
```

### 3.2 ToolCitation key (CiteableTool)

Each `CiteableTool` controls its own `citation_token` via `render_for_history()`.
As long as each implementation emits a deterministic token (as `RetrievalTool`
already does), the key is session-stable without any code changes in the citation layer.

### 3.3 ToolCitation key (plain Tool — generic wrapper)

`_generic_render_for_history` currently mints a fresh `uuid4()` as
`citation_token` per call. This prevents cross-turn stability.

**Decision:** Replace the UUID with a deterministic token derived from the
tool result content:

```python
token = hashlib.sha256(
    json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()[:16]
```

Same result → same token → same reference number across turns.
Different results from the same tool get distinct numbers, which is correct.

The LLM copies this token verbatim from the rendered XML, so a hex string is
fully compatible with existing prompt instructions.

### 3.4 Why this is different from today

`canonical_key()` already uses `citation_token` as the key — it does **not**
use `tool_call_id`. The per-turn reset is purely because `_CitationNumberer`
is instantiated fresh on every `process_message()` call (inside `_gen()`).

The only two changes required:
1. `_generic_render_for_history`: UUID → deterministic content hash (§ 3.3).
2. `_CitationNumberer`: per-`_gen()` local → session-scoped field on `ChatOrchestrator` (§ 4).

## 4. Orchestrator lifecycle changes

Current:
- `_CitationNumberer` is instantiated as a local variable inside `_gen()`,
  the async generator returned by `process_message()`.
- Its `_ref_by_key` dict and `_ref_counter` int are discarded when `_gen()` finishes.

Proposed:
- Promote `_CitationNumberer` to a session-scoped instance field:
  `self._numberer: _CitationNumberer`
- Initialize once in `ChatOrchestrator.__init__`.
- `_gen()` captures `self._numberer` via closure (same pattern as `self._history`).

Expected behavior:
- If `canonical_key(citation)` already exists in `_numberer._ref_by_key`, reuse number.
- Else increment `_numberer._ref_counter` and assign a new number.
- State persists across all `process_message()` calls on the same orchestrator instance.

## 5. API and model adjustments

### 5.1 Canonical key function

`canonical_key(citation: Citation) -> str` in `protocols.py` keeps its name
and format. Its docstring must be updated to reflect session-stable semantics:

- Current: "stable structural key for citation deduplication"
  (implicitly per-turn)
- Updated: explicitly states the key is session-stable for all citation types,
  provided `citation_token` is deterministic (guaranteed after § 3.3).

No separate `session_canonical_key()` helper is introduced.

### 5.2 NumberedCitation docs

Update the `NumberedCitation` dataclass docstring:

- Current: `"stable per-turn reference number"`
- Updated: `"stable per-session reference number"`

## 6. UI implications

- Inline markers (`[N]`) remain unchanged in format.
- Numbers may become larger over long sessions (`[17]`, `[24]`, ...).
- Side-panel rendering should not assume that low numbers are local to the
  latest answer.

This phase does not force showing numbers in side-panel labels; it only ensures
that if shown, they are globally meaningful within the session.

## 7. Migration and safety

- No data migration needed (session-memory only).
- Existing tests that expect per-turn reset must be updated.
- Add tests for cross-turn reuse:
  - same `DocumentCitation` in turn 2 reuses number from turn 1
  - same plain-tool result in turn 2 reuses number (requires deterministic token)
  - same plain-tool name + different result gets new number

## 8. Open decisions

All design decisions are resolved:

1. **Document key granularity** — `chunk_id` is globally unique in the corpus
   (content-derived hash during ingestion); no need to include `source`.
   → Resolved: keep `document:{citation_token}` (`citation_token = chunk_id`).

2. **Plain-tool token generation** — UUID replaced with
   `sha256(canonical_json(result))[:16]`.
   → Resolved: Option A (deterministic hash in `_generic_render_for_history`).

3. **`canonical_key()` naming** — no rename.
   → Resolved: keep `canonical_key()`, update docstring only.

4. **Numberer placement** — direct field on `ChatOrchestrator`.
   → Resolved: `self._numberer: _CitationNumberer` in `__init__`.

5. **Session reset boundary** — current Chainlit user session lifetime.
   → Resolved: no cross-session persistence.

## 9. Implementation sketch

1. `_generic_render_for_history` in `citation_model.py`: replace `uuid4()` token
   with `sha256(canonical_json(result))[:16]`.
2. `ChatOrchestrator.__init__`: add `self._numberer = _CitationNumberer()`.
3. `process_message._gen()`: capture `self._numberer` via closure instead of
   creating a new `_CitationNumberer()` per call.
4. `canonical_key()` docstring: update to reflect session-stable semantics.
5. `NumberedCitation` docstring: `"per-turn"` → `"per-session"`.
6. Unit tests in `test_orchestrator.py`: update tests that assert numbering resets
   between turns; add cross-turn reuse tests.
7. Unit tests in `test_citation_model.py`: add test that same result → same token
   from `_generic_render_for_history`.
