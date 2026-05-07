# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Application-wide settings loaded from environment variables.

All runtime configuration (model names, service endpoints, retrieval parameters)
is centralised here. ``pydantic-settings`` validates and coerces values at startup,
so misconfigured environments fail fast with a clear error rather than silently at
call-time.

Per-user credentials for the external-service simulation are explicitly *not*
sourced from environment variables — they are session-scoped runtime state managed
by the orchestrator (see architecture doc, Auth-Protected Tool Sequence).
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated, immutable application configuration sourced from environment variables.

    Load once at startup via :func:`get_settings` and pass as a constructor
    argument to components that need it.  Never read ``os.environ`` directly
    in application code.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Chat model ---
    chat_model: str = Field(
        default="qwen3.5:9b",
        description="Model identifier used for chat generation.",
    )
    chat_model_provider: str = Field(
        default="ollama",
        description=(
            "Chat model backend: 'ollama' for a local Ollama server, "
            "'openai_compatible' for any OpenAI-API-compatible provider (e.g. Groq, Together AI)."
        ),
        pattern="^(ollama|openai_compatible)$",
    )
    chat_base_url: str = Field(
        default="http://localhost:11434",
        description=(
            "Base URL for the chat model provider. "
            "For 'ollama' this is the Ollama server URL. "
            "For 'openai_compatible' set this to the provider endpoint "
            "(e.g. https://api.groq.com/openai/v1)."
        ),
    )
    chat_api_key: str | None = Field(
        default=None,
        description="API key for the chat model provider (required for openai_compatible).",
    )

    # --- Embedding model ---
    embedding_model: str = Field(
        default="bge-m3",
        description="Ollama model used to produce embeddings.",
    )
    embedding_model_provider: str = Field(
        default="ollama",
        description="Embedding model backend. Currently only 'ollama' is supported.",
        pattern="^(ollama)$",
    )
    embedding_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the Ollama server used for embeddings.",
    )

    # --- Qdrant ---
    qdrant_host: str = Field(
        default="localhost",
        description="Qdrant server hostname.",
    )
    qdrant_port: int = Field(
        default=6333,
        description="Qdrant server port.",
    )
    qdrant_collection: str = Field(
        default="chatbot",
        description="Name of the Qdrant collection used for document chunks.",
    )

    # --- Corpus ---
    corpus_path: str = Field(
        default="corpus",
        description="Path (relative or absolute) to the directory containing source documents.",
    )

    # --- Ingestion ---
    embedding_dim: int = Field(
        default=1024,
        description="Embedding vector dimension. Must match the output dimension of EMBEDDING_MODEL.",
    )
    split_length: int = Field(
        default=200,
        description="Number of words per ingestion chunk.",
    )
    split_overlap: int = Field(
        default=20,
        description="Word overlap between adjacent ingestion chunks.",
    )

    # --- Vision / Multi-modal ingestion ---
    vision_ingestion_enabled: bool = Field(
        default=True,
        description=(
            "When True, ingest image content (standalone image files and PDF-embedded images) "
            "by generating textual descriptions via a vision model. When False, image files "
            "are skipped and PDF image extraction is bypassed."
        ),
    )
    vision_model: str = Field(
        default="qwen2.5vl:7b",
        description="Vision-capable model identifier used for image description generation.",
    )
    vision_provider: str = Field(
        default="ollama",
        description="Vision model backend. Currently only 'ollama' is supported.",
        pattern="^(ollama)$",
    )
    vision_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the Ollama server used for vision-model calls.",
    )
    image_cache_dir: str = Field(
        default=".cache/image_descriptions",
        description=(
            "Directory used to cache vision-model descriptions keyed by image content hash. "
            "Reindexing unchanged images costs zero vision-model calls."
        ),
    )
    extracted_image_dir: str = Field(
        default=".cache/extracted_images",
        description=(
            "Directory under which PDF-embedded images are persisted (one subdirectory per "
            "source PDF). The on-disk path is the citation's surfaceable image path."
        ),
    )
    image_min_dimension: int = Field(
        default=64,
        ge=1,
        description=(
            "Filter threshold for trivially small images: images whose width or height is "
            "below this value (in pixels) are dropped before vision-model calls."
        ),
    )
    image_min_description_length: int = Field(
        default=40,
        ge=0,
        description=(
            "Length floor (in characters) on generated image descriptions. Shorter "
            "descriptions are dropped — they typically come from decorative images "
            "(logos, dividers) and would pollute retrieval."
        ),
    )

    # --- Retrieval ---
    retrieval_top_k: int = Field(
        default=5,
        description="Number of top chunks to retrieve per query.",
    )
    retrieval_llm_top_k: int | None = Field(
        default=None,
        description=(
            "Maximum number of documents passed to the LLM after RRF fusion. "
            "None (default) uses retrieval_top_k — N3f.3 precision-optimised behaviour. "
            "Set higher to surface more candidates at the cost of LLM context noise."
        ),
    )

    # --- Logging ---
    log_format: str = Field(
        default="console",
        description="Log renderer: 'console' for human-readable dev output, 'json' for structured production output.",
        pattern="^(console|json)$",
    )

    # --- Tracing (OpenTelemetry) ---
    otel_enabled: bool = Field(
        default=False,
        description="Enable OpenTelemetry tracing export.",
    )
    otel_service_name: str = Field(
        default="chatbot",
        description="OpenTelemetry service.name resource attribute.",
    )
    phoenix_project_name: str = Field(
        default="chatbot-local",
        description="Phoenix project name used to group local GenAI traces.",
    )
    otel_deployment_environment: str = Field(
        default="development",
        description="OpenTelemetry deployment environment resource attribute.",
    )
    otel_phoenix_otlp_endpoint: str = Field(
        default="http://localhost:6006/v1/traces",
        description="Phoenix OTLP/HTTP trace endpoint.",
    )
    otel_export_phoenix: bool = Field(
        default=True,
        description="Export traces to Phoenix when tracing is enabled.",
    )
    otel_export_jaeger: bool = Field(
        default=True,
        description="Export traces to Jaeger when tracing is enabled.",
    )
    otel_jaeger_otlp_endpoint: str = Field(
        default="http://localhost:4318/v1/traces",
        description="Jaeger OTLP endpoint. OTLP/HTTP (e.g. http://localhost:4318/v1/traces) is the default; non-/v1/traces endpoints use gRPC mode.",
    )
    otel_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Root span sampling ratio in [0.0, 1.0].",
    )
    otel_console_export: bool = Field(
        default=False,
        description="Additionally export spans to stdout for local debugging.",
    )
    otel_auto_instrument_haystack: bool = Field(
        default=True,
        description="Enable OpenInference auto-instrumentation for Haystack components.",
    )

    # --- Model Sampling ---
    model_temperature: float = Field(
        default=0.0,
        ge=0.0,
        description="Chat-model temperature used for generation and trace metadata. 0.0 is most deterministic/focused; 1.0 is more diverse/creative and less deterministic.",
    )
    model_seed: int = Field(
        default=42,
        description="Chat-model seed used for reproducible generations and trace metadata.",
    )

    # --- Evaluation Metadata ---
    eval_environment: str = Field(
        default="local",
        description="Logical evaluation environment label used for trace filtering.",
    )
    eval_name: str | None = Field(
        default=None,
        description="Evaluation cycle name (for example: rag-prompt-tuning-2026-04).",
    )
    eval_run_id: str | None = Field(
        default=None,
        description="Run identifier grouping related trace samples. If unset, a process-level UUID is generated automatically.",
    )
    eval_candidate_id: str | None = Field(
        default=None,
        description="Candidate identifier used to compare prompt/model variants.",
    )
    eval_prompt_version_answer: str | None = Field(
        default=None,
        description="Version label for the answer-generation prompt.",
    )
    eval_prompt_version_citation: str | None = Field(
        default=None,
        description="Version label for the citation-pass prompt.",
    )
    eval_retrieval_version: str | None = Field(
        default=None,
        description="Version label for retrieval configuration and strategy.",
    )
    eval_corpus_version: str | None = Field(
        default=None,
        description="Corpus snapshot/version identifier used for this run.",
    )
    eval_dataset_version: str | None = Field(
        default=None,
        description="Dataset snapshot/version identifier for experiment runs.",
    )

    # --- Evaluation LLM Judge ---
    eval_judge_model: str = Field(
        default="llama3.1:8b",
        description="Model identifier for the LLM judge used in eval evaluators (e.g. FaithfulnessEvaluator).",
    )
    eval_judge_provider: str = Field(
        default="ollama",
        description=(
            "LLM judge backend: 'ollama' for a local Ollama server, "
            "'openai_compatible' for any OpenAI-API-compatible provider (e.g. Groq, OpenAI)."
        ),
        pattern="^(ollama|openai_compatible)$",
    )
    eval_judge_base_url: str | None = Field(
        default=None,
        description=(
            "Base URL for the LLM judge provider. "
            "Defaults to http://localhost:11434 for 'ollama'. "
            "For 'openai_compatible' set this to the provider endpoint "
            "(e.g. https://api.groq.com/openai/v1)."
        ),
    )
    eval_judge_api_key: str | None = Field(
        default=None,
        description="API key for the LLM judge provider (required for openai_compatible).",
    )
    eval_judge_initial_per_second_request_rate: float = Field(
        default=1.5,
        gt=0.0,
        description=(
            "Initial request rate for Phoenix eval LLM judge calls. "
            "Lower values reduce rate-limit errors on constrained providers."
        ),
    )


def get_settings() -> Settings:
    """Construct and return a validated :class:`Settings` instance.

    Reads ``.env`` (if present) then environment variables.  Raises
    ``pydantic_settings.ValidationError`` on missing required fields or
    type mismatches, providing a fast-fail at startup.
    """
    return Settings()
