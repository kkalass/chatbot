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

    # --- Ollama ---
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama server.",
    )
    chat_model: str = Field(
        default="llama3.2",
        description="Ollama model used for chat generation.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama model used to produce embeddings.",
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
        default=768,
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

    # --- Retrieval ---
    retrieval_top_k: int = Field(
        default=5,
        description="Number of top chunks to retrieve per query.",
    )
    retrieval_score_threshold: float = Field(
        default=0.5,
        description="Minimum similarity score for a chunk to be included in the context.",
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

    # --- Phase 7 rollout flags ---
    inline_quotes_enabled: bool = Field(
        default=True,
        description="Enable inline quote stream items in the chat pipeline.",
    )
    citation_round_trip_enabled: bool = Field(
        default=True,
        description="Enable legacy citation round-trip pass while inline-quote rollout is in progress.",
    )


def get_settings() -> Settings:
    """Construct and return a validated :class:`Settings` instance.

    Reads ``.env`` (if present) then environment variables.  Raises
    ``pydantic_settings.ValidationError`` on missing required fields or
    type mismatches, providing a fast-fail at startup.
    """
    return Settings()
