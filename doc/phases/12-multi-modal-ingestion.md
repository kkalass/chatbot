# Phase 12: Multi-Modal Ingestion (Images and PDF Figures)

## Status
Ready for implementation. Sequenced after [Phase 11](11-retrieval-quality.md)
and before [Phase 8](08-parent-child-retrieval.md).

## Motivation

Today's ingestion silently drops all non-text content:

- PDF ingestion uses `page.extract_text()` only
  ([src/ingest/pipeline.py](../../src/ingest/pipeline.py),
  `_PdfPageConverter`). Figures, diagrams, charts, and tables embedded as
  raster or vector graphics inside the PDF contribute zero signal to
  retrieval.
- Standalone image files (`.png`, `.jpg`, `.jpeg`, `.webp`) are not in the
  list of supported suffixes (`_FORMAT_HANDLERS`) and are skipped at
  ingestion time.

For our corpus this is a meaningful information loss \u2014 several source
PDFs (e.g. IAB, Fraunhofer, IMF) carry key claims primarily in figures and
tables. A user asking about those claims gets nothing back, even though
the source document literally contains the answer.

## Goals

- Index a useful textual representation of each non-trivial image / figure
  / table found in the corpus, so that semantic queries about the visual
  content can match them.
- Preserve the link from the textual representation back to the original
  visual artefact, so the UI / citation layer can surface the image to the
  user.
- Keep ingestion deterministic and re-runnable: image \u2192 description
  generation must be cacheable (content-hash keyed) so reindexing doesn't
  re-spend LLM tokens for unchanged images.
- Compose with Phase 11's improvements (multilingual embedding, hybrid
  retrieval, reranker) without a parallel pipeline.
- Compose with Phase 8's parent-child design when it lands: an image is a
  natural "parent" whose "children" are the description chunks.

## Non-Goals

- Native multi-modal embeddings (CLIP-style image-vector search). Possible
  follow-up; not required to deliver the user-facing improvement.
- OCR of scanned PDFs (separate problem; today's PDFs are mostly text-PDFs
  with embedded figures, not scans). Re-evaluate if corpus changes.
- A multi-modal *answer model* that takes images as input. Phase 12
  improves *retrieval* by indexing image *descriptions*; the answer model
  remains text-only.
- UI redesign. The UI may need a small change to render an image when a
  citation points at one (deferred to its own small task).

## Proposed Design

### 1. Source taxonomy

- **Standalone images** in the corpus: any file with suffix
  `.png`, `.jpg`, `.jpeg`, `.webp`. Sidecar `*.meta.json` already
  supported by `load_sidecar_meta` is reused unchanged.
- **PDF-embedded images and tables**: extracted per page during PDF
  ingestion. Each extracted image is treated as if it were a standalone
  image, with provenance metadata pointing at the parent PDF + page.

### 2. Image \u2192 text description

For each image (standalone or PDF-extracted):

1. Compute a content hash (SHA-256 of normalised image bytes).
2. Look up the hash in a local description cache (filesystem or sqlite,
   stored under a configurable cache dir, e.g. `.cache/image_descriptions/`).
3. On cache miss, call a vision-capable model to produce a structured
   description.
4. Persist the description in the cache keyed by hash.
5. Index the description as a Haystack `Document` with metadata linking
   back to the image.

**Description prompt contract** (to be refined during implementation):

- Output is plain text, not JSON.
- Three short paragraphs:
  - *What it is* (chart type, diagram type, photo, screenshot, table).
  - *What it shows* (axes, categories, key entities, dominant data
    points, headings).
  - *What it claims or implies* (the takeaway a careful reader would
    extract).
- Include numeric values verbatim where present (axis labels, table
  cells, percentages). These are the highest-value retrieval signal.
- Language: match the surrounding document's language when known
  (sidecar meta or detected); fall back to English.

### 3. Vision model selection

Implementation decision:

- Primary baseline: `qwen2.5vl:7b` on Ollama.
- Provider remains swappable via `VisionDescriber` so we can move to an
  OpenAI-compatible cloud vision model if quality gates are not met.

Alternative candidates (kept for reference):

- `llama3.2-vision` (11B / 90B) \u2014 Meta's vision-instruct model on
  Ollama. Good general descriptions, English-strong. Acceptable on
  Apple Silicon at 11B.
- `qwen2.5vl` \u2014 strong on charts, diagrams and multilingual content.
  Selected initial family for our mixed DE/EN corpus (`qwen2.5vl:7b`).
- `granite3.2-vision` \u2014 IBM, lighter weight.

Wrap behind a `VisionDescriber` Protocol (`describe(image_bytes,
hint_meta) -> str`) so the model is swappable and testable with a fake.

### 4. PDF figure extraction

`pypdf` exposes per-page image streams via `Page.images`. For each PDF
page already converted to a parent text document by `_PdfPageConverter`:

- Iterate `page.images`, dedupe by content hash (PDFs frequently embed
  the same logo on every page \u2014 cache hit handles this naturally), and
  filter trivially small images (e.g. < 64\u00d764 pixels) and decorative
  images (configurable allowlist of size / aspect-ratio rules).
- Generate the description and emit a Haystack `Document` per surviving
  image, with metadata:
  - `source`: parent PDF path
  - `page`: PDF page number (1-based, string \u2014 matches existing convention)
  - `image_index`: index within page
  - `image_hash`: content hash for dedupe and cache
  - `image_path`: stable on-disk path under
    `.cache/extracted_images/<pdf_hash>/<page>_<idx>.png` so the UI
    can surface it
  - `kind`: `"image_description"` (distinct from `"text"` to allow
    retrieval-side weighting / filtering later)
  - sidecar `title`, `author`, `publication_date`, `source_url` inherited
    from the parent PDF's sidecar meta unchanged

Tables: out-of-scope for v1 in their own right, but a chart that is
*rendered as an image* in a PDF is captured by this same extraction
path. True text-tables (PDF text streams arranged in a tabular layout)
remain a known gap and are tracked as a follow-up.

### 5. Indexing

The image-description `Document`s flow through the **same** chunking,
embedding, hybrid-retrieval, and reranker stages as text documents. They
are not a parallel pipeline. This is the whole reason for sequencing
Phase 12 after Phase 11.

Practical consequences:

- The chunker may emit one or several chunks per description (descriptions
  are short \u2014 typically one chunk).
- The dense embedder embeds the description text using the multilingual
  embedding model from Phase 11.
- The sparse (BM25) side benefits especially here: numeric values and
  named entities lifted from charts and tables become exact-match
  retrievable.
- The reranker sees image-description chunks alongside text chunks and
  scores them on the same query.

### 6. Retrieval and citation

- Retrieval returns `SourceChunk`s as today; image-description chunks
  carry `image_path` and `kind="image_description"` in metadata.
- The orchestrator / tool layer is unchanged on the matching side.
- The UI / citation layer is extended (small, additive change) to render
  the linked image when a cited chunk's metadata carries an
  `image_path`. The user sees both the description text *and* the actual
  figure.

When Phase 8 lands, the image becomes the natural parent of its
description chunk(s): the citation contract returns the image (with the
description as evidence) instead of just the description text.

## Architecture Touchpoints

- [src/ingest/pipeline.py](../../src/ingest/pipeline.py): add
  image-suffix `_FormatHandler`s and a per-page image extraction step
  inside `_PdfPageConverter`.
- New module `src/ingest/infrastructure/vision/` with a `VisionDescriber`
  Protocol and an Ollama-backed implementation, mirroring the existing
  `embeddings_document/` structure.
- New module `src/ingest/infrastructure/image_cache/` (or simpler: a
  filesystem-keyed helper) for content-hash cached descriptions and
  extracted-image storage.
- [src/settings/__init__.py](../../src/settings/__init__.py): add
  vision-model settings (`vision_model`, `vision_base_url`,
  `vision_provider`, `image_cache_dir`, image filter thresholds). All
  optional with sensible defaults.
- [src/chatbot/ui/citation_view.py](../../src/chatbot/ui/citation_view.py):
  small extension to render an image when a cited chunk carries
  `image_path`.

## Acceptance Criteria

- Standalone `.png`/`.jpg` files in the corpus produce at least one
  retrievable description chunk each.
- PDF ingestion emits image-description chunks for non-trivial embedded
  images, deduped by content hash across pages.
- Description generation is cached: a second `reindex` run with no
  corpus changes makes zero vision-model calls.
- A query targeting a known figure (e.g. \"What does Figure 3 of the IMF
  report show?\") retrieves the corresponding image-description chunk in
  the top-5.
- The UI renders the cited image alongside the citation text.
- Phase 11 retrieval quality on text-only queries does not regress (the
  image-description chunks must compete fairly, not pollute, results).

## Risks and Mitigations

- **Vision model hallucination on sparse charts.** Descriptions can
  invent numbers. Mitigation: prompt explicitly for verbatim values and
  uncertainty markers; spot-check during eval.
- **Cost / latency at ingestion.** Large PDFs with many figures multiply
  vision-model calls. Mitigation: content-hash cache, configurable size
  and aspect-ratio filters, ability to disable image ingestion per
  source via sidecar meta.
- **Decorative images poisoning the index.** Logos and section
  decorations generate uninformative descriptions. Mitigation: filter
  rules (min size, min entropy) and a length floor on descriptions
  before indexing. Re-uses Phase 11's chunk-quality filter (H5).
- **Description language mismatch.** Mitigation: pass document language
  hint from sidecar meta into the prompt; multilingual embedding model
  from Phase 11 absorbs residual mismatch.
- **PDF image extraction edge cases.** Vector graphics, masks,
  non-standard color spaces. Mitigation: best-effort with `pypdf`
  `Page.images`; log and skip on failure rather than abort the page.

## Test Plan

### Unit tests
- Image hash determinism and dedupe across identical images on different
  PDF pages.
- Cache hit on second invocation with same hash \u2014 no vision call.
- Filter rules drop too-small / too-low-entropy images.
- Standalone-image ingestion produces a `Document` with the documented
  metadata shape.
- PDF page with N images + body text emits 1 text parent + N
  image-description docs (or fewer after filtering / dedupe).
- `VisionDescriber` Protocol can be substituted with a fake in tests.

### Integration tests
- End-to-end ingest of a fixture PDF containing one figure: figure is
  extracted, described, indexed, and a targeted query retrieves it.
- Fixture standalone PNG: indexed and retrievable.
- Reindex with no corpus change makes zero vision calls (assert via
  fake describer call counter).

### Regression
- Phase 11 text-retrieval eval scores within agreed tolerance of the
  text-only baseline (image chunks must not crowd out text chunks).

## Sequencing

Recommended: **after Phase 11, before Phase 8.** Rationale:

- Phase 12 is purely additive on the ingestion side and uses Phase 11's
  improved chunking + embedding + hybrid + rerank without modification.
- Doing Phase 12 before Phase 8 means Phase 8's parent-child design can
  treat \"image\" as a first-class parent type from day one rather than
  retrofitting.

## Out of Scope

- Native image embedding (CLIP / SigLIP) for image-vector retrieval.
- OCR of scanned PDFs.
- Layout-aware extraction of true text-tables from PDFs (separate phase
  if it becomes important).
- Multi-modal answer model.
