# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for settings-to-config mapping helpers."""

from src.chatbot.config import build_retriever_config
from src.ingest.config import (
    build_document_embedder_config,
    build_document_store_config,
    build_ingestion_config,
)
from src.settings import Settings


class TestConfigMappers:
    def test_build_document_store_config_maps_expected_fields(self) -> None:
        settings = Settings(
            qdrant_host="127.0.0.1",
            qdrant_port=7000,
            qdrant_collection="my_collection",
            embedding_dim=384,
        )

        config = build_document_store_config(settings)

        assert config.backend == "qdrant"
        assert config.host == "127.0.0.1"
        assert config.port == 7000
        assert config.collection == "my_collection"
        assert config.embedding_dim == 384

    def test_build_document_embedder_config_maps_expected_fields(self) -> None:
        settings = Settings(
            embedding_base_url="http://localhost:9999",
            embedding_model="bge-m3",
        )

        config = build_document_embedder_config(settings)

        assert config.provider == "ollama"
        assert config.url == "http://localhost:9999"
        assert config.embedding_model == "bge-m3"

    def test_build_ingestion_config_maps_expected_fields(self) -> None:
        settings = Settings(
            split_length=180,
            split_overlap=15,
        )

        config = build_ingestion_config(settings)

        assert config.split_length == 180
        assert config.split_overlap == 15

    def test_build_retriever_config_maps_expected_fields(self) -> None:
        settings = Settings(
            retrieval_top_k=7,
            qdrant_host="127.0.0.1",
            qdrant_port=6333,
            qdrant_collection="rag",
            embedding_dim=1024,
        )

        config = build_retriever_config(settings)

        assert config.top_k == 7
        assert config.store_collection == "rag"
