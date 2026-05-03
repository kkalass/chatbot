"""Unit tests for settings-to-config mapping helpers."""

from src.chatbot.config import build_chat_model_config
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


class TestBuildChatModelConfig:
    def test_qwen_coder_model_enables_parse_text_tool_calls(self) -> None:
        settings = Settings(chat_model="qwen2.5-coder:14b")
        config = build_chat_model_config(settings)
        assert config.parse_text_tool_calls is True

    def test_qwen_non_coder_model_disables_parse_text_tool_calls(self) -> None:
        settings = Settings(chat_model="qwen3.5:9b")
        config = build_chat_model_config(settings)
        assert config.parse_text_tool_calls is False

    def test_other_model_disables_parse_text_tool_calls(self) -> None:
        settings = Settings(chat_model="llama3.1:8b")
        config = build_chat_model_config(settings)
        assert config.parse_text_tool_calls is False
