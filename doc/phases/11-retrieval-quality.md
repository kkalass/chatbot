# Phase 11: Retrieval Quality Overhaul

## Status
Draft — to be reviewed before implementation. This phase is exploratory and
contains explicit open questions. Trade-off decisions in the "Decisions Needed"
section must be resolved with the human owner before any implementation work
begins.

## Motivation
Anecdotal inspection of retrieved top-5 chunks and Phoenix experiment results
both indicate that retrieval quality is a primary bottleneck for end-to-end
answer quality.

Concrete observations:

- **Anecdotal (qualitative).** Retrieved chunks frequently contain unhelpful
  fragments: lists of authors, copyright pages, table-of-contents entries,
  bibliography lines, single captions like `Figure 6`, or headers/footers.
  These chunks appear to match because the source document is topically
  relevant, not because the chunk text itself answers the query.
- **Experiment evidence (quantitative).** Phoenix experiment
  `RXhwZXJpbWVudDoxMA==` on dataset `RGF0YXNldDox` (see
  `http://localhost:6006/datasets/RGF0YXNldDox/compare?experimentId=RXhwZXJpbWVudDoxMA==`)
  shows many examples with `mean_document_relevance` ≈ 0.4, i.e. only 2 of the
  top-5 retrieved chunks were judged relevant.

The remainder of this document records the current state, hypothesises why
quality is poor, examines design alternatives (including non-vector
approaches), and proposes concrete next steps. It deliberately does not
prescribe a single solution path — that decision is left to the
"Decisions Needed" section.

## Current State (as of writing)

### Ingestion pipeline
Source: [src/ingest/pipeline.py](../../src/ingest/pipeline.py),
[src/ingest/config.py](../../src/ingest/config.py).

- **Supported formats.** `.txt` (TextFileToDocument), `.md`
  (MarkdownToDocument), `.pdf` (custom `_PdfPageConverter` extracting one
  Haystack `Document` per non-empty PDF page).
- **Splitting.** A single `DocumentSplitter(split_by="word", split_length=200,
  split_overlap=20)` is applied uniformly to every format after conversion.
  PDF pages are themselves split further into 200-word word-windows.
- **Embeddings.** Each child chunk's `content` is embedded with
  `OllamaDocumentEmbedder` using `nomic-embed-text` (768 dims). Only the chunk
  text is embedded — neither title, source filename, nor surrounding context
  contributes to the vector.
- **Metadata.** Sidecar `*.meta.json` is merged into chunk metadata
  (`title`, `author`, `publication_date`, `source_url`, …) and the source path
  is set as `source`. PDF chunks additionally carry `page` / `total_pages`.
- **Storage.** Chunks are written to Qdrant via the OVERWRITE policy.

### Retrieval pipeline
Source: [src/chatbot/infrastructure/retrieval/_qdrant.py](../../src/chatbot/infrastructure/retrieval/_qdrant.py),
[src/settings/__init__.py](../../src/settings/__init__.py).

- **Query path.** Pure dense vector search via `QdrantEmbeddingRetriever`.
  Query is embedded with the same `nomic-embed-text` model and matched against
  child-chunk vectors.
- **Defaults.** `retrieval_top_k = 5`, `retrieval_score_threshold = 0.5`.
- **Output.** `SourceChunk` objects (chunk content + provenance metadata) are
  returned to the orchestrator and surfaced to the LLM and citation layer.

### Phase 8 status
[doc/phases/08-parent-child-retrieval.md](08-parent-child-retrieval.md)
specifies a parent-child retrieval design (chunk search, parent return) that is
relevant to several hypotheses below. The current code does **not** yet
implement parent linkage; chunks are returned directly. Phase 11 should be
sequenced and scoped against Phase 8 — see "Relationship to Phase 8".

## Hypotheses for Poor Retrieval Quality

This section enumerates plausible causes. Each is a hypothesis to be
validated, not an established fact.

### H1. Chunks are too large and structurally arbitrary
- 200 words, naive word-window splitting, no awareness of headings,
  paragraphs, sentences, or page boundaries beyond PDF page extraction.
- A 200-word window typically mixes several distinct topics. The chunk
  vector is then an *average* of those topics; for any single user query
  it is a worse match than a focused 50-100 word chunk built around one
  semantic unit (paragraph, list item, definition) would be.
- A coarse window also makes the boilerplate problem worse: a chunk that
  spans the last 100 words of body text plus the first 100 words of a
  references list inherits the topic signal of the body text and so ranks
  high, but the LLM then sees half a page of citations.
- Observed short chunks (visibly < 200 words) are the residual tail when
  the underlying source unit (PDF page, markdown section) is shorter than
  200 words; the splitter emits them as standalone chunks with no upstream
  merging. These are a symptom, not the core problem.
- Direction: smaller, structure-aware chunks (paragraph or sentence-group
  with a hard size budget). Phase 8's parent-child design then provides the
  surrounding context to the LLM without paying for it in match quality.

### H2. Embedding text lacks contextual anchors
- Only the raw chunk text is embedded. Neither document title, section path,
  nor source filename participates in the vector. A chunk containing only
  `Figure 6` has essentially no semantic signal to differentiate it from any
  other figure caption — yet it can still rank in the top-5 if the surrounding
  document is broadly on-topic, because *some* term overlap is enough in dense
  space.
- Common mitigations: prepend `title: …\nsection: …\n` to the embedding text
  (without changing displayed content), or store separate `embedding_text`
  vs. `display_text` fields.

### H3. Embedding model may be a poor fit
- `nomic-embed-text` is a strong general-purpose English model. Our corpus is
  mixed German/English (e.g. `bibb_ki_berufsbildung.pdf`,
  `weizenbaum_dp_41.pdf` are German; `executive_order_14110.txt`,
  `ai_knowledge_worker.md` are English).
- For mixed-language corpora, multilingual embedding models (e.g.
  `bge-m3`, `multilingual-e5-large`, `paraphrase-multilingual-mpnet`) are
  typically a better default. Cross-lingual query/document matching is also
  relevant if German users ask English-context questions or vice-versa.

### H4. No lexical/keyword fallback
- Pure dense retrieval is known to underperform on:
  - exact-name lookups (acronyms, person names, model identifiers, statute
    numbers like `EO 14110`),
  - rare technical terms not well represented in the embedding model's
    pre-training distribution,
  - numeric facts (years, percentages, page numbers).
- Hybrid retrieval (BM25 + dense, fused via Reciprocal Rank Fusion or a
  weighted combination) consistently improves recall for these cases without
  losing the semantic strengths of dense.

### H5. No chunk-quality filter at ingestion
- Boilerplate (cover pages, copyright notices, references), short/sparse
  chunks (`Figure 6`), and pure-numeric tables are indexed identically to
  prose chunks. Even a simple length and alphanumeric-density filter would
  remove a class of useless top-5 hits.

### H6. No re-ranking
- A cross-encoder reranker on the top-N candidates (e.g. top-25 dense → rerank
  → top-5) is one of the highest-ROI single steps for retrieval quality.
  Currently absent.

### H7. Threshold and `top_k` are not tuned per-query
- Static `score_threshold = 0.5` against raw cosine scores is a fragile
  signal: dense scores are not calibrated, and the meaningful operating point
  is corpus- and model-dependent. The threshold is set before any quality
  evaluation has been run on it.

### H8. Multi-modal content is silently ignored
- PDFs contain figures, diagrams, and tables that are dropped (only
  `page.extract_text()` is used). Image files (`.png`, `.jpg`) in the corpus
  are not ingested at all.
- For a corpus that includes research figures and tables, this is a
  meaningful information loss — tracked separately in
  [Phase 12](12-multi-modal-ingestion.md).

### H9. "What does document X say?" queries have no good target
- Summary/overview queries ("Was steht in `bibb_ki_berufsbildung.pdf`?",
  "Summarise EO 14110") are not answerable by retrieving 5 leaf chunks of
  200 words each — the relevant signal is *the document as a whole*.
- Currently no document-level representation exists in the index. The
  retriever can only return chunks, so summary questions either return a
  random sample of chunks (poor) or a cover-page / TOC chunk (also poor).
- Industry pattern: at ingestion, generate a short LLM summary per
  document (and optionally per page/section), index it as its own document
  with a clear `kind="summary"` marker, and let it compete in retrieval.
  Hits on summary documents are routed to either:
  - return the summary directly (cheap, good for one-shot overviews), or
  - expand into the full document / page parents (when paired with
    Phase 8).
- Summaries are also natural targets for the BM25 side once hybrid is in
  place (titles + summaries dominate keyword recall for "document X"
  style queries).

## Alternatives Considered

This section evaluates the major architectural directions on the table.

### A. Stay vector-only, improve chunking + embedding
Improvements without leaving the current architecture:

- **Better chunking.** Section/heading-aware for markdown, sentence- or
  paragraph-aware splitter (with size budget) for txt and PDF page contents.
  Configurable target chunk size with a minimum length floor.
- **Embedding-text augmentation.** Prepend title and section path to the
  text passed to the embedder, while keeping the raw text as the
  `display_text` / chunk content used for citation and LLM context.
- **Embedding model swap.** Pilot a multilingual model
  (`bge-m3`, `multilingual-e5-large`) and re-run the eval set.
- **Chunk filter.** Drop chunks below a length floor or below an
  alphanumeric-density floor at ingestion time.
- **Reranker.** Add a cross-encoder rerank stage over `top_k_initial`
  candidates, return final `top_k`.

Pros: minimal architectural change, no new infrastructure, addresses the
clearest observed failure modes (H1, H2, H3, H5, H6).

Cons: still vulnerable to dense's known weakness on exact-term and
named-entity queries (H4).

### B. Pure keyword / full-text search (BM25, e.g. Elasticsearch / OpenSearch)
- **Strengths.** Excellent on exact terms, named entities, statute numbers,
  acronyms. Mature ecosystem, well-understood relevance tuning, language
  analyzers per language.
- **Weaknesses.** Poor on paraphrase, synonymy, and cross-lingual matching.
  A user asking "Welche Risiken bringt generative KI für den Arbeitsmarkt?"
  will not match an English chunk talking about "labor-market disruption from
  generative AI".
- **i18n consideration.** BM25 quality depends on per-language analyzers
  (stemming, stop-word lists). A mixed-language corpus needs either
  per-language indices or an analyzer that handles all corpus languages
  acceptably.

  Optional mitigation worth considering for the BM25 side: index *both*
  the original chunk text and an LLM-generated translation into a single
  canonical analyzer language (e.g. English), tagged with a `lang="orig"`
  vs. `lang="translation"` field. The LLM still receives the original
  source text on retrieval (translations are an index-only artefact,
  never surfaced to the user or to the answer model), so citation
  fidelity is preserved. Numbers, named entities, acronyms, and statute
  identifiers are unchanged by good translation, so BM25 lexical signal
  on those terms is not degraded. The cost is ingestion-time LLM calls,
  index size roughly 2x, and de-duplication logic so the same passage
  doesn't surface twice. To be evaluated empirically — see N3g below.

For our goal ("Answer questions based on static, multi-modal content"),
pure keyword search is insufficient as the sole retrieval strategy.

### C. Hybrid (dense + BM25)
- Combine A and B: run both retrievers in parallel, fuse with Reciprocal Rank
  Fusion (RRF) or learned weights.
- Captures both semantic similarity and exact-term matches.
- Industry-standard default for production RAG. Qdrant supports both dense
  and sparse (BM25-style) vectors in the same collection (Qdrant ≥ 1.10),
  so we can keep a single store and avoid introducing Elasticsearch /
  OpenSearch as a second piece of infrastructure. Haystack provides
  `QdrantSparseEmbeddingRetriever` plus `DocumentJoiner` for RRF/weighted
  fusion.
- **Decided direction** in combination with A's chunking and embedding
  improvements (see Decisions section below).
- For i18n: BM25 side benefits from per-language analyzers (or the optional
  dual-index translation strategy from B); the dense side remains
  language-agnostic with a multilingual model. Source text is never
  normalised destructively — translations, if used, are additive
  index-only artefacts.

### D. Parent-child retrieval (Phase 8)
Already specified in [Phase 8](08-parent-child-retrieval.md). Orthogonal to
A/B/C: it changes what is *returned* to the LLM (parent units) while keeping
search on small, focused units. Directly mitigates the "fragment
problem" — even if a `Figure 6` chunk is retrieved, the LLM receives the full
page or section, which contains the surrounding prose.

Sequencing decision (see Decisions): Phase 11 first, Phase 8 after.
Rationale: we want to validate match-quality wins (chunking, embedding,
hybrid, rerank, summaries) on a clean, chunk-only return path before
layering parent expansion on top. Otherwise it becomes hard to attribute
improvements between "better matching" and "more context".

### E. Multi-modal ingestion (images, PDF figures)
Moved to [Phase 12](12-multi-modal-ingestion.md). Out of scope for Phase 11.

### F. Document-level summaries as first-class index entries
- Generate one short summary per source at ingestion (and optionally per
  PDF page / markdown section). Index summaries as their own documents
  with `kind="summary"` metadata.
- Directly addresses H9 ("what does document X say?" queries) and provides
  high-recall targets for both dense and BM25 retrieval.
- Composes cleanly with Phase 8: a summary hit can be expanded into the
  full parent document on the return path.
- Cost: ingestion-time LLM calls, modest index growth, and a routing rule
  in the retriever (or in the tool layer) so summary hits don't crowd out
  leaf-chunk hits for fact-level queries.
- **In scope for Phase 11.**

## i18n Considerations (Summary)

- Our corpus is mixed German/English. A retrieval design that assumes English
  will systematically degrade on the German subset.
- **Do not** translate or normalize source content to a single canonical
  language at ingestion. This destroys citation fidelity, hurts named-entity
  matching, and is brittle.
- **Dense side:** use a multilingual embedding model that supports both
  languages. Cross-lingual matching is a bonus.
- **Sparse / BM25 side (if introduced):** use language-aware analyzers.
  Detect chunk language at ingestion (cheap with `langdetect` or fasttext)
  and route to the correct analyzer / index.
- Query language detection at runtime is also useful for analyzer selection
  on the sparse side; the dense side does not need it.

## Relationship to Phase 8 and Phase 12

Phase 8 (parent-child retrieval) and Phase 11 (retrieval quality) are
tightly coupled but **explicitly sequenced**:

- Phase 11 first: chunking, embedding model, hybrid, rerank, summaries.
  Return path stays chunk-based.
- Phase 8 next: change the return path to parent units, with Phase 11's
  better child chunks driving recall.
- This ordering keeps Phase 11 ablations clean (one variable at a time)
  and avoids conflating "better matching" with "more context".

Phase 12 (multi-modal ingestion) is independent of Phase 11's text
retrieval changes but benefits from being layered on top of them: it
produces *additional* text chunks (image descriptions) that flow through
the same chunking, embedding, hybrid, and rerank stack. Recommended
sequence: Phase 11 → Phase 12 → Phase 8. See Decisions Needed.

## Decisions (Resolved)

The following decisions have been made jointly with the human owner and
are locked for the planning of this phase. Changes to these require
explicit re-approval.

1. **Sequencing.** Phase 11 first, then Phase 8. Phase 12 (multi-modal)
   sequenced after Phase 11 and before Phase 8 (see open question below).
   Phase 11 keeps the chunk-based return contract so improvements can be
   attributed cleanly.
2. **Hybrid retrieval.** Commit to dense + sparse hybrid in Qdrant
   (Qdrant native sparse vectors + RRF fusion via Haystack's
   `DocumentJoiner`). No second store (no Elasticsearch).
3. **Embedding model swap.** Approved. Pilot a multilingual model and
   pick the best on our eval set. **Ollama-hosted candidates:**
   - `bge-m3` — 1024-dim, multilingual, strong on dense + supports sparse
     and ColBERT-style multi-vector in its native form (Ollama serves the
     dense vector). Recommended primary candidate.
   - `mxbai-embed-large` — 1024-dim, English-strong, secondary
     candidate.
   - `granite-embedding` (IBM) — 278M / 30M variants, multilingual.
   Reindex required; `embedding_dim` setting will change.
4. **Reranker.** Moved to a dedicated optional follow-up phase (Phase
   11b / Phase 13 TBD). Rationale: the "LLM as reranker" alternative
   (scoring top-N with the chat model) is not meaningful — if the LLM is
   asked to evaluate all candidates it may as well answer the original
   question from them directly, which is plain RAG, not reranking. The
   only reranker worth introducing is a true cross-encoder (e.g.
   `BAAI/bge-reranker-v2-m3` via `sentence-transformers`), but that adds
   a new runtime process outside Ollama, introduces latency, and its
   benefit is hard to judge before H1–H6 improvements are measured.
   **What is in scope for Phase 11:** define a `Reranker` Protocol in the
   retrieval infrastructure so that a reranker can be plugged in without
   touching the orchestrator or tool layer. Concrete implementation and
   eval in the follow-up phase.
5. **Multi-modal scope.** Confirmed out of scope for Phase 11. Tracked
   as [Phase 12](12-multi-modal-ingestion.md).
6. **Quality target.** No fixed numeric target up front. Approach: lock
   the current Phoenix `mean_document_relevance` baseline (N1), then
   iterate. Decision on "done" deferred until we see the curve — once
   improvements plateau or hit clearly diminishing returns we declare
   the phase complete and document the achieved number. Confirmed.

## Confirmed Sub-Decisions

- **Phase ordering: 11 → 12 → 8.** Confirmed. Phase 12's image-description
  chunks are additive ingestion artefacts that flow through Phase 11's
  improved stack; Phase 8 then wraps the whole retrieval pipeline in
  parent-return semantics, treating images as natural parents.
- **Quality target: iterative.** Confirmed. Lock the current baseline
  after N1, iterate, and declare "done" when the improvement curve
  plateaus or hits diminishing returns. Final achieved number documented
  at phase close.

## Next Steps

The following are **investigative and planning** steps. None of them change
production code; they produce the evidence needed to make the Decisions Needed
above.

### N1. Quantify the current baseline
- Re-run the Phoenix experiment with the existing corpus and record per-query
  `mean_document_relevance`, recall@5, and latency as the locked baseline.
- Bucket failing queries by failure mode (exact-name miss, paraphrase miss,
  short-fragment hit, off-topic hit). This bucketing directly informs which
  hypotheses (H1–H8) matter most for our corpus.

### N2. Inspect the current index
- Dump chunk-length distribution (words, characters) from Qdrant.
- Dump the top-5 chunks for each failing eval query and tag their failure
  mode by hand on a sample (≈ 20 queries).
- Confirm or refute H1 (short chunks) and H5 (boilerplate chunks)
  empirically.

### N3. Cheap, isolated experiments (no code merge required)
Run each as an isolated branch / experiment so its effect can be attributed:

- **N3a.** Swap embedding model to a multilingual model (`bge-m3` or
  `multilingual-e5-large`), reindex, re-run eval. Targets H3.
- **N3b.** Augment embedding text with `title` + `source` prefix, keep
  display text unchanged, reindex, re-run eval. Targets H2.
- **N3c.** Add a minimum-length and alphanumeric-density filter at
  ingestion, reindex, re-run eval. Targets H5.
- **N3d.** Heading-aware splitter for markdown + sentence-aware splitter
  with size budget for txt/PDF. Targets H1.
- **N3e.** Define the `Reranker` Protocol and a stub no-op implementation;
  no concrete reranker model in Phase 11. The protocol boundary ensures
  the follow-up phase can wire in a `bge-reranker-v2-m3`
  (`sentence-transformers`) without touching orchestrator or tool code.
  Targets H6 (preparation only).
- **N3f.** Add BM25 sparse vectors in Qdrant + RRF fusion with dense.
  Targets H4.
- **N3g.** Generate per-document and per-page/section summaries at
  ingestion, index with `kind="summary"` metadata, evaluate impact on
  "what does document X say?" queries (extend eval set as needed).
  Targets H9.
- **N3h.** *Optional, only if N3f hybrid still underperforms on German
  queries:* dual-index BM25 with original + LLM-translated text, evaluate
  whether the translation side adds recall without polluting precision.
  Source text returned to the LLM is always the original.

### N4. Decide and implement
After N1–N3 produce evidence, return to Decisions Needed, agree on the
implementation plan, and only then write a concrete delivery section
(scope, acceptance criteria, migration, tests) to be appended to this
document.

## Out of Scope for Phase 11

- Image / figure / multi-modal ingestion → [Phase 12](12-multi-modal-ingestion.md).
- Parent-return semantics → [Phase 8](08-parent-child-retrieval.md).

Phases 7, 7.1, 9, 10 are already implemented and are independent of this
work. Phase 11 must not regress their behaviour but does not interact with
them semantically.

## Open Questions / Resolved

- ~~Are there known queries where the current retrieval works very well?~~
  **Resolved:** No. Current retrieval quality is anecdotally
  indistinguishable from random for most queries. We have no natural
  regression anchors and need to define them deliberately as part of N1
  (e.g. queries where dense currently scores ≥ 0.8 — keep these as
  do-not-regress checks).
- ~~Do we have user-facing telemetry on which retrieved chunks were
  actually cited?~~ **Resolved (with gap).** Inspection of the trace
  attached by the human (`1342bdd84951eb5a68ec929b08d2a126`) and of
  [src/chatbot/observability/schema.py](../../src/chatbot/observability/schema.py)
  shows the current span hierarchy is
  `chat.ui.on_message → chat.orchestrator.step → chat.tool.search_documents
  → chat.retriever.qdrant.retrieve` plus model spans. The retriever span
  records `result_count` and `top_scores` but does **not** record which
  retrieved `chunk_id`s ended up cited in the final answer; citation
  parsing happens UI-side in `format_citation_marker`
  ([src/chatbot/ui/citation_view.py](../../src/chatbot/ui/citation_view.py))
  and is not reflected back into spans. **Action item for Phase 11:** add
  a per-step span attribute listing cited `chunk_id`s (set after the
  citation pass), so we can compute citation-rate-per-chunk as a
  real-world quality signal alongside Phoenix's LLM-judge metric. Small
  scope; decoupled from retrieval changes.
- ~~German-language eval dataset?~~ **Resolved.** The eval dataset
  ([eval/datasets/rag_questions.jsonl](../../eval/datasets/rag_questions.jsonl))
  is mixed: 6 English questions (covering EO 14110, IMF, GPTs-are-GPTs,
  knowledge-worker), 4 German questions (BIBB, IAB, Fraunhofer). The
  corpus mirrors this split. Multilingual gains are therefore visible in
  the eval, but the German subset is small (n=4) — the per-question
  metric will be noisy. **Action item:** as part of N1, add 6–10 more
  German questions covering the German PDFs to bring the German subset to
  ~10 items, so multilingual ablations are statistically meaningful.
  Cheap and decoupled from implementation.
