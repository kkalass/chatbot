# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Qdrant-backed Haystack ``DocumentStore`` factory."""

from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


def build_qdrant_document_store(
    *,
    host: str,
    port: int,
    collection: str,
    embedding_dim: int,
    similarity: str = "cosine",
    recreate_index: bool = False,
) -> QdrantDocumentStore:
    """Build the configured document store instance.

    The concrete return type is intentional: the chatbot's hybrid retriever
    needs the Qdrant-specific store (the haystack Qdrant retrievers require
    it), while the ingest pipeline only relies on the abstract
    :class:`~haystack.document_stores.types.DocumentStore` interface and
    accepts the concrete type via structural compatibility.
    """
    return QdrantDocumentStore(
        host=host,
        port=port,
        index=collection,
        embedding_dim=embedding_dim,
        use_sparse_embeddings=True,
        similarity=similarity,
        recreate_index=recreate_index,
    )
