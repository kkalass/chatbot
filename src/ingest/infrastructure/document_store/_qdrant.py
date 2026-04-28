"""Qdrant document store implementation helpers."""

from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


def build_qdrant_document_store(
    *,
    host: str,
    port: int,
    collection: str,
    embedding_dim: int,
    similarity: str,
    recreate_index: bool,
) -> QdrantDocumentStore:
    """Build a Qdrant-backed Haystack document store."""
    return QdrantDocumentStore(
        host=host,
        port=port,
        index=collection,
        embedding_dim=embedding_dim,
        similarity=similarity,
        recreate_index=recreate_index,
    )
