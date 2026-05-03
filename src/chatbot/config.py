"""Settings → chatbot infrastructure config converters.

Pure data transformation — no I/O, no object construction.
"""

from src.chatbot.infrastructure.chat import ChatModelConfig
from src.chatbot.infrastructure.embeddings_text import TextEmbedderConfig
from src.chatbot.infrastructure.retrieval import RetrieverConfig
from src.settings import Settings


def build_text_embedder_config(settings: Settings) -> TextEmbedderConfig:
    """Map settings to :class:`~src.chatbot.infrastructure.embeddings_text.TextEmbedderConfig`."""
    return TextEmbedderConfig(
        url=settings.ollama_base_url,
        embedding_model=settings.embedding_model,
        provider="ollama",
    )


def build_retriever_config(settings: Settings) -> RetrieverConfig:
    """Map settings to :class:`~src.chatbot.infrastructure.retrieval.RetrieverConfig`."""
    return RetrieverConfig(
        top_k=settings.retrieval_top_k,
        score_threshold=settings.retrieval_score_threshold,
        store_host=settings.qdrant_host,
        store_port=settings.qdrant_port,
        store_collection=settings.qdrant_collection,
        embedding_dim=settings.embedding_dim,
        store_backend="qdrant",
    )


def build_chat_model_config(settings: Settings) -> ChatModelConfig:
    """Map settings to :class:`~src.chatbot.infrastructure.chat.ChatModelConfig`."""
    model_lower = settings.chat_model.lower()
    parse_text_tool_calls = "qwen" in model_lower and "coder" in model_lower
    return ChatModelConfig(
        base_url=settings.ollama_base_url,
        model=settings.chat_model,
        temperature=settings.model_temperature,
        seed=settings.model_seed,
        provider="ollama",
        parse_text_tool_calls=parse_text_tool_calls,
    )
