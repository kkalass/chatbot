# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Settings → ingest infrastructure config converters.

Pure data transformation — no I/O, no object construction.
"""

from src.ingest.infrastructure.document_store import DocumentStoreConfig
from src.ingest.infrastructure.embeddings_document import DocumentEmbedderConfig
from src.ingest.pipeline import IngestionConfig
from src.settings import Settings


def build_document_store_config(settings: Settings) -> DocumentStoreConfig:
    """Map settings to :class:`~src.ingest.infrastructure.document_store.DocumentStoreConfig`."""
    return DocumentStoreConfig(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection=settings.qdrant_collection,
        embedding_dim=settings.embedding_dim,
        backend="qdrant",
    )


def build_document_embedder_config(settings: Settings) -> DocumentEmbedderConfig:
    """Map settings to :class:`~src.ingest.infrastructure.embeddings_document.DocumentEmbedderConfig`."""
    return DocumentEmbedderConfig(
        url=settings.ollama_base_url,
        embedding_model=settings.embedding_model,
        provider="ollama",
    )


def build_ingestion_config(settings: Settings) -> IngestionConfig:
    """Map settings to :class:`~src.ingest.pipeline.IngestionConfig`."""
    return IngestionConfig(
        split_length=settings.split_length,
        split_overlap=settings.split_overlap,
    )
