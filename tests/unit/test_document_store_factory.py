# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for document store factory wiring."""

from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from src.shared.qdrant import build_qdrant_document_store


class TestBuildDocumentStore:
    def test_builds_qdrant_store_from_kwargs(self) -> None:
        store = build_qdrant_document_store(
            host="localhost",
            port=6333,
            collection="unit_test_collection",
            embedding_dim=768,
        )

        assert isinstance(store, QdrantDocumentStore)
