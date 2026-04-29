"""Unit tests for settings-to-config mapping helpers."""

from src.chatbot.config import build_chat_runtime_flags
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

    def test_build_chat_runtime_flags_maps_phase7_toggles(self) -> None:
        settings = Settings(
            inline_quotes_enabled=False,
            citation_round_trip_enabled=True,
        )

        flags = build_chat_runtime_flags(settings)

        assert flags.inline_quotes_enabled is False
        assert flags.citation_round_trip_enabled is True

    def test_build_chat_runtime_flags_inline_only_defaults(self) -> None:
        """WP6: production default — inline enabled, round-trip disabled."""
        settings = Settings(
            inline_quotes_enabled=True,
            citation_round_trip_enabled=False,
        )

        flags = build_chat_runtime_flags(settings)

        assert flags.inline_quotes_enabled is True
        assert flags.citation_round_trip_enabled is False

    def test_build_chat_runtime_flags_both_disabled(self) -> None:
        settings = Settings(
            inline_quotes_enabled=False,
            citation_round_trip_enabled=False,
        )

        flags = build_chat_runtime_flags(settings)

        assert flags.inline_quotes_enabled is False
        assert flags.citation_round_trip_enabled is False
