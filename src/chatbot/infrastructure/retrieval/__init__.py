# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public retrieval infrastructure API."""

from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from src.chatbot.infrastructure.embeddings_text import TextEmbedder

from ._qdrant_hybrid import QdrantHybridRetriever


def build_qdrant_retriever(
    *,
    top_k: int,
    llm_top_k: int | None = None,
    text_embedder: TextEmbedder,
    document_store: QdrantDocumentStore,
) -> QdrantHybridRetriever:
    """Construct a Qdrant hybrid (dense + sparse) retriever.

    The document store is injected from the composition root so that the
    retriever does not own connection details.
    """
    return QdrantHybridRetriever(
        top_k=top_k,
        llm_top_k=llm_top_k,
        document_store=document_store,
        text_embedder=text_embedder,
    )


__all__ = ["build_qdrant_retriever"]
