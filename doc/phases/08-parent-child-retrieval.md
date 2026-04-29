# Phase 8: Parent-Child Retrieval (Chunk Search, Parent Return)

## Motivation
Our current retrieval indexes and returns chunks. This is efficient for matching user queries, but chunk-sized context is often too small for high-quality answer synthesis.

We need a two-level retrieval model:
- Keep chunk-level retrieval precision.
- Return larger parent units to the LLM for richer context.

This phase introduces **Parent-Child Retrieval**:
- Children: existing chunks (with overlap) used for similarity search.
- Parents: larger semantic units loaded after chunk hits and returned by the retrieval tool.

## Goals
- Keep retrieval matching on chunks (no loss of recall/precision behavior from current chunking).
- Store and retrieve parent units for all sources.
- For PDFs: use pages as parents.
- For non-PDFs: split into meaningful larger parent sections.
- Allow child chunks to reference one or multiple parent units.
- Return parents (not raw child chunks) from retrieval tool output.
- Deduplicate parent results (same parent must not be returned multiple times).

## Non-Goals
- Replace or redesign the embedding model.
- Redesign ranking model beyond parent aggregation logic.
- Remove chunk overlap.
- Introduce UI redesign in this phase.

## Proposed Design

### 1. Data Model: Two-Level Documents
Introduce explicit document hierarchy in ingestion and retrieval storage.

Logical entities:
- `ParentDocument`
  - `parent_id`
  - `source_id`
  - `parent_index` (page number for PDFs, section index for others)
  - `title` (optional)
  - `content`
  - metadata (source path, type, etc.)
- `ChildChunk`
  - `chunk_id`
  - `source_id`
  - `content`
  - embedding vector
  - chunk metadata (offsets, overlap, etc.)
- `ChunkParentLink`
  - `chunk_id`
  - `parent_id`
  - supports many-to-many mapping

Design constraint:
- Retrieval index remains child-chunk-based.
- Parent content is stored and loadable by `parent_id`.

### 2. PDF Ingestion Strategy
For PDF sources:
- Extract and persist each page as one `ParentDocument`.
- Keep existing chunking behavior (size + overlap).
- Add parent links for each chunk:
  - default: chunk linked to the page where its anchor/majority text resides.
  - optional enhancement: page-spanning chunks can link to multiple pages if chunk text crosses page boundaries.

Important:
- Page-spanning chunks are desirable and supported if extraction pipeline can preserve boundary mapping reliably.
- If page-spanning mapping is unavailable initially, phase can start with strict per-page chunk assignment and add cross-page linking as a follow-up in the same phase backlog.

### 3. Non-PDF Ingestion Strategy
For non-PDF sources (markdown, txt, etc.):
- Build larger semantic parent units first.
- Then chunk within/over these units for embeddings.

Recommended parent split heuristics:
- Markdown: headings (`#`, `##`, `###`) as parent boundaries.
- Plain text: paragraph groups with soft size thresholds.
- Fallback: fixed-size parent windows with conservative boundaries.

Requirements:
- Parent units must be meaningfully larger than child chunks.
- Child chunks must keep overlap behavior.
- Every child chunk must reference at least one parent.

### 4. Retrieval Flow
Runtime retrieval remains a two-step process:
1. Similarity search on child chunks (`top_k_chunks`).
2. Resolve linked parent IDs.
3. Deduplicate parent IDs.
4. Load full parent content.
5. Return parent documents as retrieval tool output.

Dedup policy:
- If multiple hit chunks map to the same parent, include the parent once.
- Stable ordering should be preserved by first-hit rank.

Ranking aggregation:
- Parent score derived from linked chunk hits (for example max or weighted aggregate).
- Initial default: `parent_score = max(chunk_score)`.

### 5. Retrieval Tool Contract
Update retrieval tool output contract from chunk-oriented to parent-oriented payload.

Return shape (conceptual):
- `results`: list of parent documents
  - `parent_id`
  - `source`
  - `parent_index` (page/section)
  - `content` (full parent text)
  - `evidence_chunk_ids` (for traceability)
  - aggregated score

Behavioral requirement:
- The LLM receives parent content as context.
- Internal retrieval may remain chunk-based, but external tool output must be parent-based.

## Acceptance Criteria
- PDF ingestion stores pages as retrievable parents.
- Child chunks are still embedded and searched.
- Retrieval returns deduplicated parent results, not child chunks.
- Same page/parent is never duplicated in one retrieval response.
- Non-PDF sources produce meaningful parent units and child-parent links.
- At least one end-to-end test proves chunk hit -> parent load -> parent response.

## Migration Plan
1. Add parent entities and chunk-parent link schema.
2. Extend ingestion pipeline for PDF page parent persistence.
3. Add non-PDF parent splitter and child-parent mapping.
4. Update retrieval service to resolve and deduplicate parents after chunk search.
5. Change retrieval tool response contract to parent payload.
6. Adapt orchestrator consumers if needed for new payload shape.
7. Add unit + integration tests.
8. Reindex corpus.

## Test Plan

### Unit Tests
- PDF mapping:
  - chunk mapped to single page parent.
  - optional multi-page parent link handling.
- Non-PDF mapping:
  - heading-based parent splitting.
  - fallback parent splitting.
- Retrieval aggregation:
  - multiple chunks to one parent -> one result.
  - chunks to multiple parents -> stable parent ordering.

### Integration Tests
- PDF query retrieves chunk hits and returns full page parents.
- Non-PDF query retrieves parent sections, not raw chunks.
- Dedup verification when several hit chunks belong to same parent.

### Regression Checks
- No major recall regression versus chunk-only baseline.
- Context size increase is measurable and bounded.
- Latency impact is within acceptable limits.

## Risks and Mitigations
- Parent units too large can reduce effective context budget.
  - Mitigation: configurable parent max size and optional truncation strategy.
- Page-spanning chunk mapping complexity in PDFs.
  - Mitigation: start with strict page mapping; enable cross-page links once boundary metadata is reliable.
- Ranking drift when aggregating chunks into parents.
  - Mitigation: start with `max` score and benchmark alternatives.
- Duplicate/unstable responses from aggregation.
  - Mitigation: strict `parent_id` dedup + deterministic ordering by first ranked hit.

## Implementation Guardrails (Must-Haves)
1. Keep chunk embeddings and chunk search as retrieval backbone.
2. Parent return is mandatory in retrieval tool output.
3. No duplicate parents in a single retrieval response.
4. Every returned parent must be traceable to at least one hit chunk.
5. Parent-child mapping must be persisted during ingestion, not reconstructed ad hoc at query time.
6. Changes must preserve strict typing and protocol boundaries (no direct infrastructure coupling in core orchestration).

## Handoff: Expected Code Touchpoints
- Ingestion pipeline and source splitting:
  - `src/ingest/pipeline.py`
  - `src/ingest/infrastructure/document_store/_qdrant.py`
- Retrieval infrastructure:
  - `src/chatbot/infrastructure/retrieval/_qdrant.py`
  - `src/chatbot/infrastructure/retrieval/_config.py`
- Tool contract and adapter:
  - `src/chatbot/tools/retrieval/tool.py`
  - `src/chatbot/app/protocols.py`
- Orchestrator integration (if payload handling changes):
  - `src/chatbot/app/orchestrator.py`
- Tests:
  - `tests/unit/test_retrieval_and_ingestion.py`
  - `tests/integration/test_rag.py`

## Open Decisions to Confirm Before Implementation
- Parent max length target (tokens/chars) per source type.
- Whether page-spanning chunks are required for first implementation or can ship in a second increment.
- Parent score aggregation strategy (`max`, weighted sum, reciprocal rank fusion variant).
- Whether retrieval output should include optional child snippets in addition to full parent content for debug visibility.
