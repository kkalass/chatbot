# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for src/config/settings.py."""

from src.shared.settings import Settings, get_settings


class TestSettingsDefaults:
    def test_default_values_are_valid(self) -> None:
        import pytest

        with pytest.MonkeyPatch().context() as mp:
            for key in (
                "CHAT_BASE_URL",
                "EMBEDDING_BASE_URL",
                "CHAT_MODEL",
                "EMBEDDING_MODEL",
                "QDRANT_HOST",
                "QDRANT_PORT",
                "QDRANT_COLLECTION",
                "CORPUS_PATH",
                "RETRIEVAL_TOP_K",
                "LOG_FORMAT",
                "PHOENIX_PROJECT_NAME",
                "OTEL_DEPLOYMENT_ENVIRONMENT",
                "OTEL_PHOENIX_OTLP_ENDPOINT",
                "OTEL_EXPORT_PHOENIX",
                "OTEL_EXPORT_JAEGER",
                "OTEL_JAEGER_OTLP_ENDPOINT",
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
                "EVAL_JUDGE_INITIAL_PER_SECOND_REQUEST_RATE",
            ):
                mp.delenv(key, raising=False)
            settings = Settings(_env_file="")  # pyright: ignore[reportCallIssue]
        assert settings.chat_base_url == "http://localhost:11434"
        assert settings.embedding_base_url == "http://localhost:11434"
        assert settings.chat_model == "qwen3.5:9b"
        assert settings.embedding_model == "bge-m3"
        assert settings.embedding_dim == 1024
        assert settings.qdrant_host == "localhost"
        assert settings.qdrant_port == 6333
        assert settings.qdrant_collection == "chatbot"
        assert settings.corpus_path == "corpus"
        assert settings.retrieval_top_k == 5
        assert settings.log_format == "console"
        assert settings.phoenix_project_name == "chatbot-local"
        assert settings.otel_deployment_environment == "development"
        assert settings.otel_phoenix_otlp_endpoint == "http://localhost:6006/v1/traces"
        assert settings.otel_export_phoenix is True
        assert settings.otel_export_jaeger is True
        assert settings.otel_jaeger_otlp_endpoint == "http://localhost:4318/v1/traces"
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
        assert settings.eval_judge_initial_per_second_request_rate == 1.5

    def test_get_settings_returns_settings_instance(self) -> None:
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_env_override(self, monkeypatch: object) -> None:
        import pytest

        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("CHAT_MODEL", "mistral")
            mp.setenv("RETRIEVAL_TOP_K", "10")
            mp.setenv("PHOENIX_PROJECT_NAME", "chatbot-ci")
            mp.setenv("OTEL_PHOENIX_OTLP_ENDPOINT", "http://localhost:6007/v1/traces")
            mp.setenv("OTEL_EXPORT_PHOENIX", "false")
            mp.setenv("OTEL_EXPORT_JAEGER", "true")
            mp.setenv("OTEL_JAEGER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
            mp.setenv("MODEL_TEMPERATURE", "0.2")
            mp.setenv("MODEL_SEED", "42")
            mp.setenv("EVAL_NAME", "eval-2026-04")
            mp.setenv("EVAL_CANDIDATE_ID", "mistral__ans-7__cit-3__ret-1")
            mp.setenv("EVAL_JUDGE_INITIAL_PER_SECOND_REQUEST_RATE", "0.8")
            settings = Settings()
            assert settings.chat_model == "mistral"
            assert settings.retrieval_top_k == 10
            assert settings.phoenix_project_name == "chatbot-ci"
            assert settings.otel_phoenix_otlp_endpoint == "http://localhost:6007/v1/traces"
            assert settings.otel_export_phoenix is False
            assert settings.otel_export_jaeger is True
            assert settings.otel_jaeger_otlp_endpoint == "http://localhost:4318/v1/traces"
            assert settings.model_temperature == 0.2
            assert settings.model_seed == 42
            assert settings.eval_name == "eval-2026-04"
            assert settings.eval_candidate_id == "mistral__ans-7__cit-3__ret-1"
            assert settings.eval_judge_initial_per_second_request_rate == 0.8

    def test_log_format_validation_rejects_invalid(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(log_format="invalid")
