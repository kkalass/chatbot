# Phase 13 — Architecture Review

Status: **Draft / discussion**. No code changes yet.

## Findings (current `src/` tree)

### Misleading names
- `src/chatbot/config.py` and `src/ingest/application/config.py` are not configuration —
  they are `Settings → InfraConfig` mappers, i.e. composition glue.
- `src/chatbot/app/tracing.py` (app-type → span-attribute serialization) clashes
  in name with `src/chatbot/observability/tracing.py` (OTEL setup). Both are
  observability concerns at different layers.
- `src/chatbot/app/protocols.py` is a 600-line catch-all: `ChatMessage`, `Tool`,
  `ChatModel`, `Retriever`, `Citation`, `CredentialStore`, `I18nMessage`,
  `ModelProfile`, `ProcessEvent`, …
- `src/chatbot/app/protocols_citeable_tool.py` is the second catch-all (naming drift).
- `src/ingest/application/` does not match the documented target name `app/`.

### Layering violations
- `src/ingest/domain/image_description_service.py` imports `infrastructure/{image_cache,vision}` —
  violates "domain stays pure". It is a composite of infrastructure adapters, not a
  pure type module.
- `IngestionPipeline.__init__` receives `ImageDescriptionService` only to pass it into
  `_build_format_handlers` which forwards it to converter constructors. The pipeline
  never calls any method on the service itself. `_build_format_handlers` is therefore
  composition logic that belongs in the composition root, not inside the pipeline.
  Once moved, `IngestionPipeline` has zero dependency on `ImageDescriptionService`.
- `src/chatbot/infrastructure/retrieval/_qdrant_hybrid.py` imports
  `src.ingest.infrastructure.embeddings_sparse` — cross-feature dependency
  between `chatbot` and `ingest`.
- `src/chatbot/ui/logging_config.py` is imported from the ingest CLI — a UI
  module reused from a non-UI entry point.
- `src/chatbot/infrastructure/chat/_model_profile.py` is listed here for completeness:
  on first read it appears to violate layering because `adjust_prompts` looks like
  app policy. On closer inspection the profiles encode *model-family quirks* (how a
  specific model family handles tool calls, thinking tokens, etc.) — that is adapter
  knowledge, not orchestration policy. It stays in `infrastructure/chat/`; only the
  *selection* function (`build_chat_model_profile`) moves to the composition root.

### Composition fragmented
- Composition logic is spread over: `_build_*` in `src/chatbot/ui/app.py`,
  `_build_*` in `src/ingest/cli.py`, both `config.py` mappers, plus
  `infrastructure/chat/__init__.py::build_chat_model_profile()` (which selects
  a profile from the model name — a composition decision).

### Inconsistent public-surface convention
- Some packages re-export a clear API via `__init__.py`
  (e.g. `infrastructure/chat/`, `tools/vacation_days/`).
- Others have empty `__init__.py` and consumers import private modules
  directly (e.g. `app/citation/_parser`).
- Underscore convention is uneven: `_shared.py` vs `_input_model.py` vs `_qdrant.py`
  vs plain-named internals.

### Stray files outside sub-packages
- `src/ingest/infrastructure/embeddings_sparse.py` is a single file while its
  sibling `embeddings_document/` is a sub-package.

---

## Concept: structure principles

Two axes, applied consistently in both feature packages:

1. **Vertical layering** inside a feature (`chatbot`, `ingest`):
   - `contracts/` — pure types, value objects, Protocols. No framework imports
     beyond stdlib / `pydantic` / `dataclasses`.
   - `app/` — use-case orchestration. Imports `contracts/`. Consumes Protocols;
     does not import provider code directly.
   - `infrastructure/` — all concrete Protocol implementations. Sub-packages
     within the same feature may import each other (e.g. a composite adapter
     importing two primitive adapters). Each sub-package: `Config` (frozen
     dataclass), `build_*` factory, private `_impl.py`.
     - `infrastructure/tools/` *(chatbot only)* — implements `Tool` Protocol;
       may depend on other `infrastructure/` sub-packages (e.g. `retrieval/`).
     - `infrastructure/converters/` *(ingest only)* — implements `FileConverter`;
       may depend on `infrastructure/image_description/` etc.
   - `ui/` or `cli/` — entry point + composition root. All adapter and
     service instantiation lives here, not inside `app/`.

2. **Horizontal cross-cutting** above the features (`src/shared/`):
   - `shared/settings/` — `pydantic-settings` Settings + `get_settings()`.
   - `shared/observability/` — OTEL setup, OpenInference helpers, structlog setup.
   - `shared/qdrant/` — shared Qdrant infrastructure: `DocumentStoreConfig`,
     `build_document_store`, and `embeddings_sparse/` (BM25). Both features
     connect to the same Qdrant collection; the connection config and the sparse
     model are genuinely shared.

   Grouping all three under `shared/` is honest: `settings/` and
   `observability/` are just as cross-feature as `embeddings_sparse`. Separate
   top-level names would imply a distinction that does not exist.

> The term **"domain"** is dropped. It promised purity that the existing module
> did not deliver, and there is no equivalent on the chatbot side. `contracts/`
> is honest about what lives there: typed boundaries, nothing else.

> No separate **`services/`** layer. `ImageDescriptionService` appeared to need
> one because both `converters/` and `app/pipeline.py` imported it. On
> inspection `pipeline.py` only forwarded the service into converter constructors
> and never called it directly — meaning `_build_format_handlers` is composition
> logic. Moving it to `cli/composition.py` removes `pipeline.py`'s dependency on
> `ImageDescriptionService` entirely. The service is then a pure
> infrastructure composite (cache + vision adapter + filter config) and lives in
> `ingest/infrastructure/image_description/`.

### Rules (testable)
1. `contracts/` imports only stdlib / `pydantic` / `dataclasses`.
2. `app/` imports `contracts/` and Protocols from `infrastructure/`. Never
   instantiates infrastructure directly; that is composition's job.
3. `infrastructure/` imports `contracts/` and other `infrastructure/`
   sub-packages **within the same feature**. Never `app/`, never another feature.
4. Composition glue (`_build_*`, settings → config mappers, converter
   construction) lives only in `ui/composition.py` or `cli/composition.py`.
   No `config.py` files for mapping logic.
5. Public API per `__init__.py` in every sub-package. Internal modules use a
   `_` prefix consistently.
6. No feature imports another feature. Cross-feature code lives in
   `src/shared/` (`shared/settings/`, `shared/observability/`, `shared/qdrant/`).

---

## Target tree

```
src/
├── shared/                         # all cross-feature / cross-cutting code
│   ├── settings/                   # was src/settings/ (unchanged)
│   │   └── __init__.py             # Settings + get_settings()
│   ├── observability/              # was chatbot/observability/ + logging_config.py
│   │   ├── tracing.py              # OTEL setup
│   │   ├── openinference.py        # OpenInference helpers
│   │   └── logging.py              # structlog setup (was chatbot/ui/logging_config.py)
│   └── qdrant/                     # shared Qdrant infrastructure
│       ├── _config.py              # DocumentStoreConfig (host, port, collection,
│       │                           # embedding_dim, similarity) — frozen dataclass
│       ├── _document_store.py      # build_document_store(config, recreate_index)
│       ├── embeddings_sparse/      # BM25 sparse embedder
│       │                           # (was ingest/infrastructure/embeddings_sparse.py)
│       └── __init__.py
│
├── chatbot/
│   ├── contracts/                  # was app/protocols.py + protocols_citeable_tool.py + prompts.py, split by topic
│   │   ├── chat.py                 # ChatMessage, ToolCallInfo, ChatStreamItem, ChatModel, ModelProfile, ThinkingContent
│   │   ├── tools.py                # Tool, ToolSchema, JsonObject
│   │   ├── citation.py             # Citation family, CiteableTool, CitableUnit, RawCitation, markers
│   │   ├── retrieval.py            # Retriever, SourceChunk
│   │   ├── credentials.py          # CredentialStore, UsernamePasswordCredentials, AuthRequiredException
│   │   ├── process.py              # ProcessEvent, ToolEvent, AuthRequiredEvent, ToolCallStarted/Finished
│   │   ├── i18n.py                 # I18nMessage
│   │   └── prompts.py              # Prompts dataclass only (despite the name: it is a type,
│   │                               # not a prompt string) — in contracts because it appears in
│   │                               # ModelProfile.adjust_prompts() signature; Protocol and its
│   │                               # param type must be co-located
│   │
│   ├── app/                        # use-case orchestration
│   │   ├── orchestrator.py
│   │   ├── chat_prompts.py         # DEFAULT_PROMPTS — app-level policy: what the chatbot does
│   │   │                           # by default; model profiles modify it, not define it
│   │   ├── citation/               # CitationModel, parser, messages
│   │   └── credential_store.py     # InMemoryCredentialStore (default impl)
│   │
│   ├── infrastructure/             # all concrete Protocol implementations
│   │   ├── chat/                   # Ollama, OpenAI-compatible, TextToolCallParsingWrapper
│   │   │                           # _model_profile.py stays here: profiles encode model-family
│   │   │                           # quirks — infra knowledge, not app policy
│   │   ├── embeddings_text/
│   │   ├── observability/          # chatbot-specific OTEL helpers
│   │   │                           # _attrs.py: summarize_messages helper (was app/tracing.py)
│   │   │                           # _spans.py: chatbot-specific span names (was observability/schema.py)
│   │   │                           # placed under infrastructure/ because they are stateless
│   │   │                           # OTEL-shape helpers — no orchestration policy. App layer
│   │   │                           # imports the public API only.
│   │   ├── retrieval/              # QdrantHybridRetriever; receives QdrantDocumentStore
│   │   │                           # injected from composition root (no store_* fields)
│   │   └── tools/                  # implements Tool Protocol; may depend on retrieval/ etc.
│   │       ├── _input_model.py
│   │       ├── retrieval/
│   │       └── vacation_days/
│   │
│   └── ui/                         # Chainlit lifecycle + rendering
│       ├── app.py                  # thin: handlers; delegates to composition
│       ├── composition.py          # _build_*, settings → config mappers (replaces chatbot/config.py)
│       ├── citation_view.py
│       └── i18n_messages.py
│
└── ingest/
    ├── contracts/                  # pure types and constants
    │   ├── images.py               # IMAGE_SUFFIXES, IMAGE_KIND_DESCRIPTION,
    │   │                           # ImageFilterConfig, ImageDescriptionResult
    │   └── converters.py           # FileConverter Protocol, ImageDescriptionPayload
    │                               # (was inside ingest/converters/_shared.py)
    │
    ├── app/                        # renamed from application/
    │   ├── pipeline.py             # IngestionPipeline + IngestionConfig
    │   │                           # constructor receives pre-built format handlers;
    │   │                           # no ImageDescriptionService dependency
    │   └── vision_prompts.py       # build_image_description_prompt (was application/vision_prompts.py)
    │                               # app-level policy: what we want extracted from images;
    │                               # vision adapter receives the built prompt injected, not imported
    │
    ├── infrastructure/
    │   ├── embeddings_document/    # document (write-side) embedder
    │   ├── image_cache/
    │   ├── image_description/      # ImageDescriptionService composite
    │   │                           # (was ingest/domain/image_description_service.py)
    │   │                           # combines image_cache + vision + ImageFilterConfig
    │   ├── vision/                 # vision LLM adapter; receives prompt string as constructor param
    │   └── converters/             # implements FileConverter Protocol; depends on image_description/
    │       ├── image.py
    │       └── pdf.py
    │   # note: document_store/ and embeddings_sparse/ removed — both moved
    │   # to src/shared/qdrant/ because chatbot also constructs the same store
    │
    └── cli/                        # was cli.py
        ├── __main__.py             # argparse + main
        └── composition.py          # _build_*, settings → config mappers,
                                    # _build_format_handlers (replaces application/config.py)
```

---

## Refactoring map (what each move solves)

| Today | Move | Solves |
|---|---|---|
| `chatbot/config.py` | → `chatbot/ui/composition.py` (delete `config.py`) | name lies; composition fragmented |
| `ingest/application/config.py` | → `ingest/cli/composition.py` (delete) | same |
| `app/protocols.py` (600 LOC) | split by topic → `chatbot/contracts/*` | catch-all module |
| `app/prompts.py::Prompts` | → `chatbot/contracts/prompts.py` | param type of ModelProfile Protocol |
| `app/prompts.py::DEFAULT_PROMPTS` | → `chatbot/app/chat_prompts.py` | app-level policy; symmetric with `ingest/app/vision_prompts.py` |
| `app/protocols_citeable_tool.py` | merge into `chatbot/contracts/citation.py` | naming drift |
| `app/tracing.py` | → `chatbot/infrastructure/observability/` | name clash with OTEL setup; belongs in infrastructure |
| `src/settings/` | → `src/shared/settings/` | honest: settings is also cross-feature |
| `chatbot/observability/` | → `src/shared/observability/` | cross-cutting; CLI also needs it |
| `chatbot/observability/schema.py` | → `chatbot/app/observability_spans.py` | chatbot-specific spans don't belong in cross-cutting layer |
| `chatbot/ui/logging_config.py` | → `src/shared/observability/logging.py` | UI module reused from CLI |
| `infrastructure/chat/_model_profile.py` | stays in `infrastructure/chat/` | profiles are model-family adapters — infra knowledge |
| `infrastructure/chat/__init__.py::build_chat_model_profile` | → `chatbot/ui/composition.py` | profile *selection* by model name is composition |
| `ingest/domain/image_description_service.py` | → `ingest/infrastructure/image_description/` | infra composite; pipeline passes it through only |
| `IngestionPipeline.__init__` (accepts `image_service`) | remove param; accept pre-built format handlers | composition belongs in composition root |
| `_build_format_handlers` in `pipeline.py` | → `ingest/cli/composition.py` | converter construction is composition |
| `ingest/domain/__init__.py` (constants) | → `ingest/contracts/images.py` | "domain" misnamed |
| `ingest/application/vision_prompts.py` | → `ingest/app/vision_prompts.py` | app-level policy (what to extract), not infra detail (how to call); adapter receives prompt injected |
| `ingest/converters/_shared.py` (FileConverter Protocol etc.) | → `ingest/contracts/converters.py` | typed boundary belongs with contracts |
| `ingest/infrastructure/embeddings_sparse.py` | → `src/shared/qdrant/embeddings_sparse/` | cross-feature import violation + stray single file; both solved by moving to shared |
| `ingest/infrastructure/document_store/` | → `src/shared/qdrant/` | both features build same QdrantDocumentStore; duplicate config fields |
| `chatbot/infrastructure/retrieval/RetrieverConfig.store_*` | remove fields; inject `QdrantDocumentStore` | eliminates config duplication; retriever no longer responsible for store construction |
| `ingest/application/` | → `ingest/app/` | matches naming convention |
| `ingest/cli.py` | → `ingest/cli/{__main__.py, composition.py}` | mixes entry-point with composition |
| `chatbot/tools/` | → `chatbot/infrastructure/tools/` | not a separate layer; implements `Tool` Protocol like other infra adapters implement theirs |
| `ingest/converters/` | → `ingest/infrastructure/converters/` | same: `FileConverter` impls belong inside `infrastructure/`, not as a sibling layer |

---

## Decisions

1. **`ingest/contracts/converters.py`**: `FileConverter` Protocol and
   `ImageDescriptionPayload` both live in this one module. No split into
   `protocols.py` / `payloads.py`.
2. **`IngestionPipeline` constructor signature**: receives a
   `Sequence[FormatHandler]` injected by the composition root.
   `FormatHandler` (named-tuple or frozen dataclass) stays in `app/pipeline.py`
   — it is not infrastructure-specific.
3. **`DocumentStore` type**: imported directly from haystack at the use site
   (`ingest/app/pipeline.py`). No re-export, no custom Protocol wrapper.
4. **Rollout**: no migration shims, no backwards-compatible intermediate state.
   Each step lands clean. Order:
   1. ✅ Create `src/shared/`; move `settings/` → `shared/settings/`, extract
      `shared/observability/` — purely additive, update all import sites.
   2. ✅ Move `embeddings_sparse` + `document_store/` → `shared/qdrant/`;
      remove `store_*` fields from `RetrieverConfig`; inject store from
      composition root.
   3. ✅ Rename `ingest/application/` → `ingest/app/`.
   4. ✅ Replace both `config.py` files with `composition.py`; move
      `_build_format_handlers` there; inject format handlers into pipeline.
   5. ✅ Move `image_description_service` →
      `ingest/infrastructure/image_description.py`.
   6. ✅ Split `app/protocols.py` into `chatbot/contracts/`.
   7. ✅ Move `chatbot/tools/` → `chatbot/infrastructure/tools/`;
      `ingest/converters/` → `ingest/infrastructure/converters/`;
      extract `FileConverter` Protocol + `ImageDescriptionPayload` + image
      constants to `ingest/contracts/`.
   8. ✅ Convert `ingest/cli.py` → `ingest/cli/{__main__.py, composition.py}`.

   Steps 1–5 are largely mechanical. Step 6 touches the most import sites.

   **Status: complete.** All 8 steps landed clean (no shims). Final validation:
   `ruff` clean, `pyright` 0 errors / 0 warnings / 0 informations,
   `pytest` 237 passed / 4 skipped.
