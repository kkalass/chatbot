"""Unit tests for document store factory wiring."""

import pytest
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from src.ingest.infrastructure.document_store import DocumentStoreConfig, build_document_store


class TestBuildDocumentStore:
    def test_builds_qdrant_store_from_config(self) -> None:
        config = DocumentStoreConfig(
            backend="qdrant",
            host="localhost",
            port=6333,
            collection="unit_test_collection",
            embedding_dim=768,
        )

        store = build_document_store(config)

        assert isinstance(store, QdrantDocumentStore)

    def test_raises_for_unsupported_backend(self) -> None:
        config = DocumentStoreConfig(
            backend="qdrant",
            host="localhost",
            port=6333,
            collection="unit_test_collection",
            embedding_dim=768,
        )
        # Runtime guard: invalid backend values from untyped inputs must fail fast.
        object.__setattr__(config, "backend", "elasticsearch")

        with pytest.raises((AssertionError, ValueError)):
            build_document_store(config)
