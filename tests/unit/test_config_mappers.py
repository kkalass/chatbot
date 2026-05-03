# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for settings-to-config mapping helpers."""

from src.ingest.config import build_document_store_config
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
