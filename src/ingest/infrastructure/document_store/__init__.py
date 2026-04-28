"""Public document-store infrastructure API."""

from dataclasses import dataclass
from typing import Literal, assert_never

from haystack.document_stores.types import DocumentStore

from ._qdrant import build_qdrant_document_store


@dataclass(frozen=True)
class DocumentStoreConfig:
    """Construction-time config for the application document store."""

    host: str
    port: int
    collection: str
    embedding_dim: int
    backend: Literal["qdrant"] = "qdrant"
    similarity: str = "cosine"


def build_document_store(
    config: DocumentStoreConfig, *, recreate_index: bool = False
) -> DocumentStore:
    """Build and return the configured document store instance."""
    match config.backend:
        case "qdrant":
            return build_qdrant_document_store(
                host=config.host,
                port=config.port,
                collection=config.collection,
                embedding_dim=config.embedding_dim,
                similarity=config.similarity,
                recreate_index=recreate_index,
            )
        case _:
            assert_never(config.backend)


__all__ = ["DocumentStore", "DocumentStoreConfig", "build_document_store"]
