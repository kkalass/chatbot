"""Unit tests for src/config/settings.py."""

from src.settings import Settings, get_settings


class TestSettingsDefaults:
    def test_default_values_are_valid(self) -> None:
        import pytest

        with pytest.MonkeyPatch().context() as mp:
            for key in (
                "OLLAMA_BASE_URL",
                "CHAT_MODEL",
                "EMBEDDING_MODEL",
                "QDRANT_HOST",
                "QDRANT_PORT",
                "QDRANT_COLLECTION",
                "CORPUS_PATH",
                "RETRIEVAL_TOP_K",
                "RETRIEVAL_SCORE_THRESHOLD",
                "LOG_FORMAT",
                "PHOENIX_PROJECT_NAME",
                "OTEL_DEPLOYMENT_ENVIRONMENT",
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                "OTEL_AUTO_INSTRUMENT_HAYSTACK",
                "MODEL_TEMPERATURE",
                "MODEL_SEED",
                "EVAL_ENVIRONMENT",
                "EVAL_NAME",
                "EVAL_RUN_ID",
                "EVAL_CANDIDATE_ID",
                "EVAL_PROMPT_VERSION_ANSWER",
                "EVAL_PROMPT_VERSION_CITATION",
                "EVAL_RETRIEVAL_VERSION",
                "EVAL_CORPUS_VERSION",
                "EVAL_DATASET_VERSION",
            ):
                mp.delenv(key, raising=False)
            settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
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
        assert settings.phoenix_project_name == "chatbot-local"
        assert settings.otel_deployment_environment == "development"
        assert settings.otel_exporter_otlp_endpoint == "http://localhost:6006/v1/traces"
        assert settings.otel_auto_instrument_haystack is True
        assert settings.model_temperature == 0.0
        assert settings.model_seed == 42
        assert settings.eval_environment == "local"
        assert settings.eval_name is None
        assert settings.eval_run_id is None
        assert settings.eval_candidate_id is None
        assert settings.eval_prompt_version_answer is None
        assert settings.eval_prompt_version_citation is None
        assert settings.eval_retrieval_version is None
        assert settings.eval_corpus_version is None
        assert settings.eval_dataset_version is None

    def test_get_settings_returns_settings_instance(self) -> None:
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_env_override(self, monkeypatch: object) -> None:
        import pytest

        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("CHAT_MODEL", "mistral")
            mp.setenv("RETRIEVAL_TOP_K", "10")
            mp.setenv("PHOENIX_PROJECT_NAME", "chatbot-ci")
            mp.setenv("MODEL_TEMPERATURE", "0.2")
            mp.setenv("MODEL_SEED", "42")
            mp.setenv("EVAL_NAME", "eval-2026-04")
            mp.setenv("EVAL_CANDIDATE_ID", "mistral__ans-7__cit-3__ret-1")
            settings = Settings()
            assert settings.chat_model == "mistral"
            assert settings.retrieval_top_k == 10
            assert settings.phoenix_project_name == "chatbot-ci"
            assert settings.model_temperature == 0.2
            assert settings.model_seed == 42
            assert settings.eval_name == "eval-2026-04"
            assert settings.eval_candidate_id == "mistral__ans-7__cit-3__ret-1"

    def test_log_format_validation_rejects_invalid(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(log_format="invalid")
