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
- **Experiment evidence (quantitative).** Frozen baseline for this phase is
  Phoenix experiment `RXhwZXJpbWVudDoxMQ==` on dataset `RGF0YXNldDox` (see
  `http://localhost:6006/datasets/RGF0YXNldDox/compare?experimentId=RXhwZXJpbWVudDoxMQ==`).
  Aggregate `mean_document_relevance` for that run is `0.386`, i.e. fewer than
  2 of the top-5 retrieved chunks were relevant on average.

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

## Baseline Frozen (N1) / Index Inspection (N2)

The first two planned next steps have been executed against the current
production-like corpus and the agreed reference experiment.

### N1. Frozen baseline
- **Reference experiment.** `RXhwZXJpbWVudDoxMQ==`
  (`chatbot-eval-f9f7d224`) on `rag-questions-v1`.
- **Aggregate metrics.**
  - `mean_document_relevance = 0.386`
  - average latency in Phoenix compare view: `18.8s`
- **Worst queries by mean document relevance.**
  - `0.00` — `Welche Chancen und Risiken sieht das Fraunhofer-Institut beim Einsatz von KI in Unternehmensprozessen?`
  - `0.10` — `How might generative AI affect employment and the future of work according to the IMF?`
  - `0.1667` — `Was empfiehlt das BIBB zum Umgang mit KI in der Berufsausbildung?`
  - `0.20` — `Which federal agencies are required to take action under Executive Order 14110?`
  - `0.28` — `What does the IMF report say about which jobs are most exposed to generative AI?`
- **Best queries by mean document relevance.**
  - `1.00` — `Welche Auswirkungen hat KI auf die berufliche Bildung in Deutschland?`
  - `0.6333` — `How do knowledge workers use AI tools to augment their work, and what skills become more important?`
  - `0.60` — `Wie verändert die Digitalisierung und KI die Tätigkeitsprofile in deutschen Berufen laut IAB?`
  - `0.48` — `What are GPTs in the context of economic history, and why might large language models qualify as general purpose technologies?`
  - `0.40` — `What are the main AI safety requirements established by Executive Order 14110?`

Interpretation:
- The baseline confirms the qualitative impression that retrieval quality is
  highly unstable across questions.
- German coverage is not uniformly bad; the issue is retrieval selectivity,
  not merely corpus language mismatch.
- Even the "better" queries are mostly below a quality level that would be
  acceptable for production grounding.

### N2. Current index findings
Direct inspection of the active Qdrant collection (`chatbot`) shows:

- `623` indexed chunks total.
- Word-count distribution:
  - min `4`
  - mean `143.57`
  - median `154`
  - max `295`
- `27` chunks have fewer than `50` words.
- `200` chunks have fewer than `100` words.
- Source distribution (top 8):
  - `corpus/executive_order_14110.txt` → `220`
  - `corpus/imf_gen_ai_future_of_work.pdf` → `128`
  - `corpus/gpts_are_gpts.pdf` → `83`
  - `corpus/iab_digitalisierung.pdf` → `72`
  - `corpus/bibb_ki_berufsbildung.pdf` → `56`
  - `corpus/fraunhofer_ki_prozesse.pdf` → `36`
  - `corpus/weizenbaum_dp_41.pdf` → `22`
  - `corpus/ai_knowledge_worker.md` → `6`
- Page provenance is present for `397` chunks and absent for `226` chunks
  (expected: PDFs have page metadata, txt/md do not).

Representative short-chunk samples confirm the failure modes from the initial
anecdotal review:
- `WORKING PAPER Figure 6` (`4` words, `corpus/gpts_are_gpts.pdf`)
- bibliography / title-page fragments from `corpus/bibb_ki_berufsbildung.pdf`
- table / axis fragments from `corpus/iab_digitalisierung.pdf`
- footer / filing boilerplate from `corpus/executive_order_14110.txt`
- appendix / classification fragments from `corpus/imf_gen_ai_future_of_work.pdf`

Interpretation:
- H1 and H5 are now supported by actual index data, not just anecdotal UI
  inspection.
- The dominant problem is not only that some chunks are too short; it is that
  the current splitter emits many low-signal residual chunks and allows
  structurally poor units (TOC entries, figure captions, filing footers,
  appendix tables) into the index unchanged.
- PDF page extraction preserves page provenance, but there is no structural
  protection against indexing low-information page tails or figure-only text.

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
   **Updated priority signal from N3f.4:** the N3f.4 experiment showed
   that the hybrid retriever already achieves high recall (9.4 relevant
   docs/q) — the limiting factor is precision, not recall.  A
   cross-encoder reranker is the natural next step and should be treated
   as **high priority** when retrieval quality work resumes.
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
- Done for the frozen baseline experiment `RXhwZXJpbWVudDoxMQ==`.
- Next remaining sub-step: bucket the failing queries by failure mode
  (exact-name miss, paraphrase miss, short-fragment hit, off-topic hit).

### N2. Inspect the current index
- Done for chunk-length distribution and spot-check sampling.
- H1 and H5 are empirically supported.
- Next remaining sub-step: dump the top-5 chunks for each failing eval query
  and tag their failure mode by hand on a sample (≈ 20 query-document pairs).

### N3. Completed experiment snapshot (current state)

> **⚠ Judge incompatibility notice.**  Early experiments (N3a–N3f, N3f-old)
> were originally evaluated with `llama3.1:8b` running locally via Ollama.
> Selected experiments (Baseline, N3a, N3f.3) were later replayed with
> `llama-v3p3-70b-instruct` (Fireworks AI) as judge.  Scores from these two
> judges are **not directly comparable** — the 70b model is more lenient and
> produces systematically higher absolute scores.  Only compare within the same
> judge column in the tables below.

The following experiments have been executed and compared against the frozen
baseline (`RXhwZXJpbWVudDoxMQ==`, `mean_document_relevance = 0.386`,
`llama3.1:8b` judge):

- **Baseline (replay with `llama-v3p3-70b-instruct` judge).**
  - Same chatbot traces as `RXhwZXJpbWVudDoxMQ==`, re-evaluated with the
    stronger judge to establish a comparable reference point.
  - Replay experiment: `RXhwZXJpbWVudDozNQ==`.
  - Result file: `eval/results/phase11-baseline-judge-llama33-70b-perdoc.jsonl`.
  - Aggregate (`llama-v3p3-70b-instruct` judge): mean `0.399`, median `0.350`,
    docs/q `18.0`.

- **N3a (embedding model swap only).**
  - Setup: `EMBEDDING_MODEL=bge-m3`, `EMBEDDING_DIM=1024`, low-signal filter
    disabled, structure-aware segmentation disabled.
  - Result file: `eval/results/phase11-n3a-bge-m3-only-perdoc.jsonl`.
  - Aggregate (`llama3.1:8b` judge): mean `0.6182`, median `0.650`, docs/q
    `13.3` (`+0.232` vs baseline).
  - Aggregate (`llama-v3p3-70b-instruct` judge): mean `0.730`, median `0.750`.
    Result file: `eval/results/phase11-n3a-judge-llama33-70b-v3-perdoc.jsonl`.
  - Interpretation: strong positive signal; multilingual embedding swap is the
    first change that clearly improves retrieval quality in this repo.

- **N3b (embedding text augmentation: title/source prefix).**
  - Setup: `EMBEDDING_MODEL=bge-m3`, `EMBEDDING_DIM=1024`,
    `INGESTION_EMBEDDING_CONTEXT_PREFIX=true`, low-signal filter disabled,
    structure-aware segmentation disabled.
  - Experiment: `RXhwZXJpbWVudDoxOA==`
    (`http://127.0.0.1:6006/datasets/RGF0YXNldDox/compare?experimentId=RXhwZXJpbWVudDoxOA==`).
  - Result file: `eval/results/phase11-n3b-bge-m3-plus-context-prefix-perdoc.jsonl`.
  - Aggregate (`llama3.1:8b` judge): mean `0.4793`, median `0.463`, docs/q `9.9`
    (`+0.093` vs baseline, `-0.139` vs N3a — same judge).
  - Interpretation: in this setup, prefix augmentation clearly underperforms
    plain `bge-m3`; keep it disabled by default.

- **N3c (low-signal chunk filter only).**
  - Setup: `INGESTION_MIN_CHUNK_WORDS=40`, `INGESTION_MIN_ALPHA_RATIO=0.55`.
  - Result file: `eval/results/phase11-n3c-low-signal-filter-rerun-perdoc.jsonl`.
  - Aggregate (`llama3.1:8b` judge): mean `0.3724`, median `0.295`, docs/q `16.0`
    (`-0.014` vs baseline — same judge).
  - Interpretation: despite removing obvious junk chunks, this threshold pair
    slightly hurts retrieval relevance on the current dataset.

- **N3d (structure-aware segmentation).**
  - Setup: heading-aware markdown segmentation + paragraph grouping before
    split; tested behind `INGESTION_STRUCTURE_AWARE_SEGMENTATION=true`.
  - Result file: `eval/results/phase11-n3d-structure-aware-splitting-perdoc.jsonl`.
  - Aggregate (`llama3.1:8b` judge): mean `0.3360`, median `0.263`, docs/q `12.5`
    (`-0.050` vs baseline, `-0.036` vs N3c — same judge).
  - Interpretation: this first segmentation variant regresses quality and
    should not be the default path.

- **N3f-old (hybrid with home-grown TF sparse — INVALIDATED).**
  - Result file: `eval/results/phase11-n3f-hybrid-dense-sparse-rrf-perdoc.jsonl`.
  - Aggregate (`llama3.1:8b` judge): `0.5268` — **invalidated.** Sparse embedder
    used TF + hash tokenisation without IDF; regex ASCII tokeniser silently
    dropped all German tokens with umlauts. Not comparable to any
    fastembed-based result.

- **N3f (hybrid dense+sparse with fastembed `Qdrant/bm25` + RRF — invalidated, superseded by N3f.3/N3f.4).**
  - Setup: `EMBEDDING_MODEL=bge-m3`, `EMBEDDING_DIM=1024`,
    `INGESTION_ENABLE_SPARSE_VECTORS=true`, `RETRIEVAL_MODE=hybrid`.
    Sparse vectors produced by fastembed `Qdrant/bm25` (pre-trained multilingual
    IDF, Unicode tokenisation).  Dense + sparse fused via RRF.
  - Experiment: `RXhwZXJpbWVudDoyMA==`
    (`http://127.0.0.1:6006/datasets/RGF0YXNldDox/compare?experimentId=RXhwZXJpbWVudDoyMA==`).
  - Result file: `eval/results/phase11-n3f-fastembed-bm25-hybrid-perdoc.jsonl`.
  - Aggregate (`llama3.1:8b` judge): `0.5967` — **invalidated.**  The retriever
    used `top_k * 2` as the candidate pool size for both dense and sparse
    retrievers, giving hybrid an unfair advantage over dense-only N3a (which
    uses exactly `top_k`). Fixed: both retrievers now use `top_k`; superseded
    by N3f.3 and N3f.4.

- **N3f.3 (hybrid + split-query: separate `query_dense` / `query_sparse` fields).**
  - Setup: as N3f above plus `query_dense` and `query_sparse` tool-schema fields;
    the LLM formulates a semantic paraphrase for dense and a keyword term list
    for BM25 separately.  `DocumentJoiner` bounded to `top_k` candidates.
  - Experiments: `llama3.1:8b` judge `RXhwZXJpbWVudDoyMw==`;
    `llama-v3p3-70b-instruct` replay `RXhwZXJpbWVudDozNA==`.
  - Result files: `eval/results/phase11-n3f3-split-query-perdoc.jsonl`
    (`llama3.1:8b`), `eval/results/phase11-n3f3-judge-llama33-70b-perdoc.jsonl`
    (`llama-v3p3-70b-instruct`).
  - Aggregate (`llama3.1:8b` judge): mean `0.609`, median `0.536`, docs/q `6.3`
    (`-0.009` vs N3a same judge, within noise).
  - Aggregate (`llama-v3p3-70b-instruct` judge): mean `0.744`, median `1.000`,
    docs/q `6.3` (`+0.014` vs N3a same judge — marginal positive).
  - Interpretation: split-query yields fewer candidates per query (6.3 vs 13.3
    for N3a), reducing noise in the LLM context.  Quality delta vs N3a is small
    but non-negative.  The llama33-70b judge shows a clearer (though still
    modest) improvement.  The halved docs/q is the stronger practical signal:
    the LLM context is cleaner without losing relevant documents.

- **N3f.4 (full RRF pool: `retrieval_pool_size=None`).**
  - Setup: as N3f.3 (split-query: separate `query_dense` / `query_sparse`
    fields); `DocumentJoiner` returns the complete fused list without a
    `top_k` cut (`retrieval_pool_size=None`).  Both individual retrievers
    still fetch exactly `top_k` candidates; the joiner merges them without
    truncating.
  - Experiment: `RXhwZXJpbWVudDozNg==` (`phase11-n3f4-hybrid-full-pool`).
  - Result file: `eval/results/phase11-n3f4-hybrid-full-pool-perdoc.jsonl`.
  - Aggregate (`llama-v3p3-70b-instruct` judge): mean `0.678`, median `0.686`,
    docs/q `17.9` (relevant docs/q `9.4` at score > 0).
  - Within-judge comparison vs N3f.3 (`llama-v3p3-70b-instruct`): mean
    `0.678` vs `0.744` — N3f.4 is **worse** despite surfacing 9.4 relevant
    docs/q vs 3.7 for N3f.3.  The full pool contains more relevant documents
    in absolute terms, but dilutes the per-query mean because irrelevant
    documents (score `0`) make up the larger share of the pool (8.5 of 17.9
    total, vs 2.6 of 6.3 for N3f.3).  The `top_k` cut in N3f.3 acts as a
    precision filter: it discards both noise *and* some relevant documents,
    but the noise-to-signal ratio in the retained set is better.
  - **Reranking signal:** N3f.4's recall (9.4 relevant docs/q) demonstrates
    that the hybrid retriever *finds* the right documents — the bottleneck is
    precision, not recall.  A cross-encoder reranker over the N3f.4 pool would
    be able to surface those 9.4 relevant documents in a tight top-k, which a
    blind `top_k` cut cannot.  This is strong empirical motivation to
    prioritise reranking (N3e / Phase 13) as the next retrieval quality step.

#### Judge comparison table

The `llama-v3p3-70b-instruct` replays re-evaluate saved experiment traces with
a stronger judge (Fireworks AI).  **Do not compare scores across judge columns**
— the 70b model is more lenient and produces systematically higher absolute
scores.  Only the within-column delta (e.g. N3a vs Baseline, same judge) is
meaningful.

| Config | llama3.1:8b mean | llama3.1:8b median | llama-v3p3-70b mean | llama-v3p3-70b median | docs/q | rel docs/q (>0) |
|---|---|---|---|---|---|---|
| Baseline | `0.386`† | — | `0.399` | `0.350` | 18.0 | — |
| N3a — bge-m3 dense | `0.618` | `0.650` | `0.730` | `0.750` | 13.3 | — |
| N3f.3 — split-query hybrid | `0.609` | `0.536` | `0.744` | `1.000` | 6.3 | 3.7 |
| N3f.4 — full RRF pool | — | — | `0.678` | `0.686` | 17.9 | 9.4 |

† Baseline `llama3.1:8b` = `0.386` from Phoenix experiment
`RXhwZXJpbWVudDoxMQ==`; `llama-v3p3-70b-instruct` replay
`RXhwZXJpbWVudDozNQ==`.

For reference — experiments that showed regressions (`llama3.1:8b` judge only;
no `llama-v3p3-70b-instruct` replay run; deltas are within-judge):

| Config | llama3.1:8b mean | vs baseline | vs N3a |
|---|---|---|---|
| N3b — title/source prefix | `0.479` | `+0.093` | `-0.139` |
| N3c — chunk filter 40w/0.55α | `0.372` | `-0.014` | `-0.246` |
| N3d — structure-aware segmentation | `0.336` | `-0.050` | `-0.282` |

Current decision from measured evidence:
- **N3a** (`bge-m3` embedding swap) is the confirmed primary improvement and the
  basis for all subsequent experiments.  Δ = `+0.232` vs baseline
  (`llama3.1:8b` judge), `+0.331` (`llama-v3p3-70b-instruct` judge — same-judge
  comparison only).
- **N3f.3** (split-query hybrid) is at worst neutral vs N3a and halves LLM
  context noise (docs/q 6.3 vs 13.3).  Considered a net positive.
- **N3f.4** (full RRF pool, `llama-v3p3-70b-instruct` judge) is complete.
  Mean `0.678` vs N3f.3's `0.744` (same judge) — **worse** despite surfacing
  9.4 relevant docs/q (vs 3.7 for N3f.3).  The `top_k` cut in N3f.3 acts as
  an effective precision filter: the noise-to-signal ratio of the uncut pool
  drags the per-query mean down.  N3f.4 is discarded in favour of N3f.3.
- **N3b** is disabled by default — prefix augmentation hurts vs N3a on this
  corpus/model combination.
- **N3c / N3d** are discarded in their current parameterisations; both
  regress quality vs baseline.  No further tuning planned for Phase 11.

### N3. Cheap, isolated experiments (no code merge required)
Run each as an isolated branch / experiment so its effect can be attributed:

- **N3a.** Swap embedding model to a multilingual model (`bge-m3` or
  `multilingual-e5-large`), reindex, re-run eval. Targets H3.
  - Status: done for `bge-m3` with strong gain (`+0.2322` vs baseline).
- **N3b.** Augment embedding text with `title` + `source` prefix, keep
  display text unchanged, reindex, re-run eval. Targets H2.
  - Status: done (tested on top of `bge-m3`), regression vs N3a
    (`0.4793`, i.e. `-0.1389` vs N3a).
- **N3c.** Add a minimum-length and alphanumeric-density filter at
  ingestion, reindex, re-run eval. Targets H5.
  - Status: done (tested thresholds `40` / `0.55`), slight regression on this
    dataset (`-0.0136` vs baseline).
- **N3d.** Heading-aware splitter for markdown + sentence-aware splitter
  with size budget for txt/PDF. Targets H1.
  - Status: done for first variant, regression (`-0.0500` vs baseline).
- **N3e.** Define the `Reranker` Protocol and a stub no-op implementation;
  no concrete reranker model in Phase 11. The protocol boundary ensures
  the follow-up phase can wire in a `bge-reranker-v2-m3`
  (`sentence-transformers`) without touching orchestrator or tool code.
  Targets H6 (preparation only).
- **N3f.** Add BM25 sparse vectors in Qdrant + RRF fusion with dense.
  Targets H4.
  - Status: implemented with fastembed `Qdrant/bm25`.  Previous result
    (`0.5967`) invalidated: retriever used `top_k * 2` pool, giving hybrid an
    unfair advantage.  Fixed; re-evaluation required.
  - Note on reranking: RRF cannot resolve the case where dense and sparse
    produce fully disjoint candidate lists — it assigns rank-based scores but
    has no signal for absolute relevance across the two pools.  A
    cross-encoder reranker over the merged pool (before the final `top_k` cut)
    directly addresses this.  With two retrievers each fetching `top_k`, the
    merged pool already contains up to `2 * top_k` candidates — exactly the
    right input size for the reranker, no multiplier needed (N3e).
  - Sub-experiments (run after valid N3f baseline is established):
    - **N3f.2** — Tool description adapted for hybrid mode: instruct the LLM
      to pass proper nouns, acronyms, and statute identifiers (e.g. "BIBB",
      "Executive Order 14110") verbatim rather than paraphrasing them.
      Hypothesis: the current description ("embedding based vector search")
      biases the LLM toward paraphrasing away the exact terms that sparse
      matching needs.
    - **N3f.3** — Two query fields in the tool schema: `query_dense` (LLM
      formulates a semantic paraphrase for dense) and `query_sparse` (LLM
      formulates a keyword-oriented term list for BM25).  Each field is routed
      to its respective retriever.  Measures whether explicit query splitting
      beats feeding the same query to both sides.
      - Status: **done**.  `_SearchInput` extended with optional
        `query_dense` and `query_sparse`; `Retriever` Protocol and both
        implementations updated; tool description updated to guide the LLM to
        provide all three fields.
      - Results: see snapshot above.  Mean `0.609` (orig judge), `0.744`
        (llama33-70b).  Docs/q `6.3` vs `13.3` for N3a.  Considered net
        positive; see decision in snapshot section.
    - **N3f.4** — Full RRF pool (`retrieval_pool_size=None`): both dense and
      sparse retrievers each fetch `top_k` candidates; the `DocumentJoiner`
      returns the complete fused list without a cut (no `top_k` limit on the
      joiner output), so the LLM receives all candidates ranked by RRF score.
      Hypothesis: limiting the joiner to `top_k` or `2*top_k` discards
      candidates that would be relevant for RRF — passing the full pool lets
      the LLM see the complete signal from both retrievers.  Uses split-query
      like N3f.3 (`query_dense` / `query_sparse` in tool schema).
      - Status: **complete, discarded** (experiment `phase11-n3f4-hybrid-full-pool`
        `RXhwZXJpbWVudDozNg==`).  Mean `0.678` vs N3f.3's `0.744` (same judge);
        discarded in favour of N3f.3.
      - Setting: `retrieval_pool_size: int | None = None` (default, no limit).
    - **N3f.5** — Sparse query = unmodified original user message (bypasses
      the LLM query formulation step for the BM25 side); dense query = LLM
      formulated as today.  Simplest way to ensure exact named-entity terms
      reach BM25 without extra LLM calls.
    - **N3f.6** — Combine N3f.3 and N3f.5: three inputs to BM25 (LLM
      keyword-optimised term + LLM dense query + original user message).
      Highest recall potential; evaluate precision impact.
- **N3g.** Generate per-document and per-page/section summaries at
  ingestion, index with `kind="summary"` metadata, evaluate impact on
  "what does document X say?" queries (extend eval set as needed).
  Targets H9.
- **N3h.** *Optional, only if N3f hybrid still underperforms on German
  queries:* dual-index BM25 with original + LLM-translated text, evaluate
  whether the translation side adds recall without polluting precision.
  Source text returned to the LLM is always the original.
- **N3i.** *Alternative infrastructure experiment:* replace Qdrant sparse
  vectors with Elasticsearch/OpenSearch for the BM25 side.
  Motivation: ES/OS provides corpus-specific IDF (computed from the actual
  index), per-language analyzers (German stemming, stop-words, umlaut
  normalisation), and mature relevance tuning tooling — capabilities that
  fastembed's pre-trained IDF approximates but does not fully replicate.
  Implementation approach: build `ingest_es` and `retrieval_es` adapters
  alongside the existing Qdrant adapters (do not remove Qdrant); wire them
  behind the same `Protocol` boundaries; compare eval results.
  Cost: new Docker service (elasticsearch or opensearch), Haystack ES
  integration, hybrid fusion logic between the ES BM25 retriever and the
  Qdrant dense retriever.  Worthwhile only if fastembed N3f results still
  disappoint after N3f.2–N3f.5 are explored.

### N4. Decide and implement
After N1–N3 produce evidence, return to Decisions Needed, agree on the
implementation plan, and only then write a concrete delivery section
(scope, acceptance criteria, migration, tests) to be appended to this
document.

## Experiment Decision Summary

### Keep and ship (confirmed positive signal)

- **N3a — bge-m3 multilingual embedding.**  Strongest single improvement
  (`+0.232` orig, `+0.331` llama33-70b vs baseline).  Default for all further
  experiments and for the final configuration.
- **N3f — hybrid dense+sparse (Qdrant fastembed bm25 + RRF).**  Infrastructure
  is implemented and stable.  Signal is positive after the `top_k*2` bug fix;
  N3f.3 and N3f.4 are the active evaluation variants.
- **N3f.3 — split-query (separate `query_dense` / `query_sparse` tool fields).**
  Marginal quality gain; halves LLM context noise (docs/q 6.3 vs 13.3).
  Implemented; default for hybrid mode going forward unless N3f.4 demonstrates
  a clear advantage.

### Discard — evaluated, net negative or superseded

- **N3f.4 — full RRF pool (`retrieval_pool_size=None`).**  Evaluated with
  `llama-v3p3-70b-instruct` judge: mean `0.678` vs N3f.3's `0.744` (same
  judge) — worse.  9.4 relevant docs/q vs 3.7 for N3f.3 shows recall is
  strong, but the noise-to-signal ratio of the uncut pool drags precision
  down.  Discarded in favour of N3f.3.  The recall data is strong motivation
  for reranking (N3e / Phase 13).

### Postpone — not pursued in Phase 11

- **N3f.5 — original user message as sparse query.**  Not evaluated; lower
  priority given N3f.3 already handles this well enough.  Can be revisited
  in Phase 13 together with reranking.
- **N3f.6 — three-way BM25 input.**  Same; deferred to Phase 13.

### Discard — empirically tested, net negative on this dataset

- **N3b — title/source embedding prefix.** Regression vs N3a (`-0.139`).
  Keep the setting flag for future experiments but do not enable by default.
- **N3c — chunk filter (min 40 words / alpha ratio 0.55).** Slight regression
  vs baseline (`-0.014`).  Thresholds are too aggressive; not worth tuning
  further in Phase 11.
- **N3d — structure-aware segmentation (first variant).** Regression vs
  baseline (`-0.050`).  Heading-aware splitting alone does not improve
  retrieval without a complementary parent-return mechanism (Phase 8).

### Postpone indefinitely / out of scope for Phase 11

- **N3e — Reranker Protocol + concrete cross-encoder.**  Protocol boundary
  defined; concrete implementation (`bge-reranker-v2-m3`) deferred to
  Phase 13.  Adding a second runtime process (outside Ollama) is not
  warranted until the embedding + hybrid baseline is stable.
  **High-priority next step:** N3f.4 demonstrated that recall is already
  strong (9.4 relevant docs/q from the full RRF pool); a cross-encoder
  reranker over that pool is the highest-ROI remaining improvement for
  retrieval precision.  Should be the first experiment when Phase 13
  starts.
- **N3g — Per-document / per-page LLM summaries.**  High potential for H9
  queries but requires ingestion-time LLM calls and routing logic.  Defer
  to a dedicated sub-phase after Phase 11 hybrid is shipped.
- **N3h — Dual-index BM25 with LLM-translated text.**  Only relevant if
  hybrid fastembed still underperforms on German queries after N3f.3/N3f.4.
  Current data does not indicate this is the bottleneck; park until needed.
- **N3i — Elasticsearch / OpenSearch as BM25 backend.**  Infrastructure
  complexity (new Docker service, new adapters) not justified while fastembed
  variants are unexplored.  Keep as a last resort if fastembed BM25 quality
  ceiling is reached.

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
  knowledge-worker), and the German subset has now been expanded beyond the
  original 4 questions (BIBB, IAB, Fraunhofer, Weizenbaum). The
  corpus mirrors this split. Multilingual gains are therefore visible in
  the eval, but we still need to monitor balance between English and German
  queries in future revisions so multilingual ablations remain statistically
  meaningful.
