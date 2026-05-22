# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Settings-driven factory functions for the ingest subsystem.

Bridges :class:`~src.shared.settings.Settings` to ingest infrastructure so
that the CLI composition root stays thin and the wiring logic is not
bound to a specific entry point.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Literal, assert_never

import structlog
from haystack.components.converters import MarkdownToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.document_stores.types import DocumentStore

from src.ingest.app import (
    FormatHandler,
    IngestionConfig,
    IngestionPipeline,
    build_image_description_prompt,
)
from src.ingest.contracts.converters import FileConverter
from src.ingest.contracts.images import IMAGE_SUFFIXES
from src.ingest.infrastructure.converters import (
    ImageFileConverter,
    PdfPageConverter,
)
from src.ingest.infrastructure.embeddings_document import (
    DocumentEmbedder,
    build_ollama_document_embedder,
)
from src.ingest.infrastructure.image_cache import ExtractedImageStore, ImageDescriptionCache
from src.ingest.infrastructure.image_description import (
    ImageDescriptionService,
    ImageFilterConfig,
)
from src.ingest.infrastructure.vision import VisionDescriber, build_ollama_vision_describer
from src.shared.qdrant import build_qdrant_document_store
from src.shared.qdrant.embeddings_sparse import SparseDocumentEmbedder
from src.shared.settings import Settings

logger = structlog.get_logger(__name__)


def _build_word_splitter(config: IngestionConfig) -> DocumentSplitter:
    return DocumentSplitter(
        split_by="word",
        split_length=config.split_length,
        split_overlap=config.split_overlap,
    )


def build_format_handlers(
    *,
    image_service: ImageDescriptionService | None,
    extracted_image_store: ExtractedImageStore | None,
) -> tuple[FormatHandler, ...]:
    """Assemble the per-suffix dispatch handlers for one pipeline instance.

    Image suffixes (``.png``, ``.jpg``, ``.jpeg``, ``.webp``) are registered
    only when *image_service* is supplied; otherwise standalone image files
    are treated as unsupported and skipped at discovery time.
    """

    def _make_pdf_converter() -> FileConverter:
        return PdfPageConverter(
            image_service=image_service,
            extracted_image_store=extracted_image_store,
        )

    handlers: list[FormatHandler] = [
        FormatHandler(
            suffix=".txt",
            converter_factory=TextFileToDocument,
            splitter_factory=_build_word_splitter,
        ),
        FormatHandler(
            suffix=".md",
            converter_factory=MarkdownToDocument,
            splitter_factory=_build_word_splitter,
        ),
        FormatHandler(
            suffix=".pdf",
            converter_factory=_make_pdf_converter,
            splitter_factory=_build_word_splitter,
        ),
    ]
    if image_service is not None:
        bound_service = image_service

        def _make_image_converter() -> FileConverter:
            return ImageFileConverter(bound_service)

        for suffix in sorted(IMAGE_SUFFIXES):
            handlers.append(
                FormatHandler(
                    suffix=suffix,
                    converter_factory=_make_image_converter,
                    splitter_factory=_build_word_splitter,
                )
            )
    return tuple(handlers)


def _build_vision_describer(settings: Settings) -> VisionDescriber:
    """Build the vision describer and its dependencies from settings."""
    provider: Literal["ollama"] = settings.vision_provider  # type: ignore[assignment]  # validated by settings pattern constraint
    match provider:
        case "ollama":
            return build_ollama_vision_describer(
                model=settings.vision_model,
                url=settings.vision_base_url,
                prompt_builder=build_image_description_prompt,
            )
        case _:
            assert_never(provider)


def _build_image_service(
    settings: Settings,
) -> tuple[
    ImageDescriptionService | None,
    ExtractedImageStore | None,
]:
    """Build the image-description service when vision ingestion is enabled.

    Returns a ``(service, extracted_image_store)`` pair; both are ``None`` when
    vision is disabled, so the caller registers no image handlers.
    """
    if not settings.vision_ingestion_enabled:
        logger.info("cli.reindex.vision_disabled")
        return None, None
    describer = _build_vision_describer(settings)
    cache = ImageDescriptionCache(Path(settings.image_cache_dir))
    extracted_image_store = ExtractedImageStore(Path(settings.extracted_image_dir))
    image_service = ImageDescriptionService(
        describer=describer,
        cache=cache,
        filter_config=ImageFilterConfig(
            min_dimension=settings.image_min_dimension,
            min_description_length=settings.image_min_description_length,
        ),
    )
    logger.info(
        "cli.reindex.vision_enabled",
        model=settings.vision_model,
        cache_dir=settings.image_cache_dir,
        extracted_dir=settings.extracted_image_dir,
    )
    return image_service, extracted_image_store


def build_ingestion_config(settings: Settings) -> IngestionConfig:
    """Map settings to :class:`~src.ingest.app.IngestionConfig`."""
    return IngestionConfig(
        split_length=settings.split_length,
        split_overlap=settings.split_overlap,
        batch_size=settings.ingest_file_batch_size,
    )


def _build_document_embedder(settings: Settings) -> DocumentEmbedder:
    """Construct the document embedder prescribed by ``config.provider``."""
    provider: Literal["ollama"] = settings.embedding_model_provider  # type: ignore[assignment]  # validated by settings pattern constraint
    match provider:
        case "ollama":
            return build_ollama_document_embedder(
                model=settings.embedding_model,
                url=settings.embedding_base_url,
            )
        case _:
            assert_never(provider)


def build_ingestion_pipeline(
    settings: Settings,
    *,
    document_store_factory: Callable[[], DocumentStore] | None = None,
) -> IngestionPipeline:
    """Compose a fully-wired :class:`IngestionPipeline` for one CLI run.

    *document_store_factory* lets ``cmd_reset`` reuse the freshly recreated
    store so that the wipe and re-ingest land on the same instance; if omitted,
    a new store is built from settings.
    """
    document_store: DocumentStore = (
        document_store_factory()
        if document_store_factory
        else build_qdrant_document_store(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection=settings.qdrant_collection,
            embedding_dim=settings.embedding_dim,
        )
    )
    embedder = _build_document_embedder(settings)
    sparse_embedder = SparseDocumentEmbedder()
    logger.info("cli.reindex.sparse_vectors_enabled")
    image_service, extracted_image_store = _build_image_service(settings)
    handlers = build_format_handlers(
        image_service=image_service,
        extracted_image_store=extracted_image_store,
    )
    return IngestionPipeline(
        config=build_ingestion_config(settings),
        document_store=document_store,
        embedder=embedder,
        format_handlers=handlers,
        sparse_embedder=sparse_embedder,
    )
