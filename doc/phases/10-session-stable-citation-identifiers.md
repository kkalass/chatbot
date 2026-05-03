# Phase 10 — Session-Stable Citation Identifiers

> Status: Draft proposal for discussion. No implementation before explicit sign-off.

## 1. Goal

Make citation reference numbers stable across the whole chat session, not only
within one request/turn.

Current behavior:
- `NumberedCitation.reference_number` is assigned per turn in the orchestrator.
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

Out of scope for this phase:
- UX redesign of the sidebar
- Persistence across app restarts (this proposal is session-memory only)

## 3. Identity model (Option 3)

Session-stable numbering requires session-stable identity keys that are
independent of per-call transport IDs.

### 3.1 DocumentCitation key

Use evidence identity, not tool-call identity.

Proposed key:
- `document:{source}:{chunk_id}`

Rationale:
- `tool_call_id` changes between calls and must not define identity.
- `source + chunk_id` represents the cited chunk itself.

### 3.2 ToolCitation key

Use tool identity plus normalized tool response.

Proposed key:
- `tool:{tool_name}:{canonical_json(result)}`

This follows the decision in this discussion: for tool citations, key by
`tool id + response`, because one tool can be cited for different responses.

Notes:
- `tool id` here means tool name / tool identity exposed to the model.
- `canonical_json(result)` must be deterministic (sorted keys, stable separators).

### 3.3 Why this is different from today

Today the key function uses `tool_call_id` for both document and tool
citations. That is correct for per-turn dedup, but it prevents stable identity
across turns.

## 4. Orchestrator lifecycle changes

Current:
- `ref_by_key` and `ref_counter` are local variables inside one
  `process_message()` call.

Proposed:
- Move them to session-scoped orchestrator state:
  - `self._session_ref_by_key: dict[str, int]`
  - `self._session_ref_counter: int`
- Keep assignment logic the same, but use session fields.

Expected behavior:
- If citation key already exists in `self._session_ref_by_key`, reuse number.
- Else increment `self._session_ref_counter` and assign a new number.

## 5. API and model adjustments

### 5.1 Canonical key function

Update canonical key derivation to session semantics.

Potential shape:
- Keep existing helper for turn-local dedup if needed.
- Add a dedicated helper for session-stable identity, e.g.
  `session_canonical_key(citation: Citation) -> str`.

### 5.2 NumberedCitation docs

Update docs to reflect session-stable semantics.

Current wording says "stable per-turn".
Proposed wording: "stable per-session".

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
  - same tool + same result reuses number
  - same tool + different result gets new number

## 8. Open decisions (for explicit sign-off)

1. Document key granularity:
   - Keep `document:{source}:{chunk_id}`
   - Optional: include page if `chunk_id` is not globally unique in corpus
2. Tool identity field:
   - Use `tool_name` as `tool id` (default)
   - Optional: explicit stable tool identifier if introduced later
3. Session reset boundary:
   - Keep current chat session lifetime (Chainlit user session)
   - No cross-session persistence

## 9. Implementation sketch

1. Introduce session key helper(s) in citation model layer.
2. Move reference map/counter to `ChatOrchestrator` instance state.
3. Replace turn-local key usage with session key usage in both normal and
   fallback streaming paths.
4. Update unit tests for orchestrator numbering semantics.
5. Optionally re-enable visible side-panel numbering once session-stable IDs
   are in place.
