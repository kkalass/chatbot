"""Unit tests for src/config/settings.py."""

from src.config.settings import Settings, get_settings


class TestSettingsDefaults:
    def test_default_values_are_valid(self) -> None:
        settings = Settings()
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.chat_model == "llama3.2"
        assert settings.embedding_model == "nomic-embed-text"
        assert settings.qdrant_host == "localhost"
        assert settings.qdrant_port == 6333
        assert settings.qdrant_collection == "chatbot"
        assert settings.corpus_path == "corpus"
        assert settings.retrieval_top_k == 5
        assert settings.retrieval_score_threshold == 0.5
        assert settings.log_format == "console"

    def test_get_settings_returns_settings_instance(self) -> None:
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_env_override(self, monkeypatch: object) -> None:
        import pytest

        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("CHAT_MODEL", "mistral")
            mp.setenv("RETRIEVAL_TOP_K", "10")
            settings = Settings()
            assert settings.chat_model == "mistral"
            assert settings.retrieval_top_k == 10

    def test_log_format_validation_rejects_invalid(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(log_format="invalid")
