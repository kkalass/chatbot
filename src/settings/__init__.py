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


def get_settings() -> Settings:
    """Construct and return a validated :class:`Settings` instance.

    Reads ``.env`` (if present) then environment variables.  Raises
    ``pydantic_settings.ValidationError`` on missing required fields or
    type mismatches, providing a fast-fail at startup.
    """
    return Settings()
