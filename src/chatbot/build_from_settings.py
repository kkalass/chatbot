# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Settings-driven factory functions for the chatbot subsystem.

Bridges :class:`~src.shared.settings.Settings` to chatbot infrastructure so
that multiple entry points (Chainlit UI, eval runner) can build the same
subsystem components without duplicating settings-field mapping.

Provider dispatch and model-profile selection live here — they are
composition decisions ("which concrete implementation for this config"),
not infrastructure knowledge.
"""

from typing import Literal, assert_never

from src.chatbot.contracts.chat import ChatModel, ModelProfile
from src.chatbot.infrastructure.chat import (
    DefaultChatModelProfile,
    QwenCoderModelProfile,
    SmallModelProfile,
    TextToolCallParsingWrapper,
    build_ollama_chat_model,
    build_openai_compatible_chat_model,
)
from src.chatbot.infrastructure.embeddings_text import TextEmbedderConfig, build_text_embedder
from src.chatbot.infrastructure.retrieval import build_qdrant_retriever
from src.chatbot.infrastructure.tools.retrieval import RetrievalTool
from src.shared.qdrant import build_qdrant_document_store
from src.shared.settings import Settings


def build_model_profile(model_name: str) -> ModelProfile:
    """Select a model profile based on the model name.

    Profiles encode model-family quirks (tool-call format, thinking tokens,
    prompt adjustments).  Add a case here when a new model family needs
    non-default behaviour.
    """
    name = model_name.lower()
    match name:
        case n if "llama" in n:
            return SmallModelProfile()
        case n if "qwen" in n and "coder" in n:
            return QwenCoderModelProfile()
        case n if "qwen" in n:
            return SmallModelProfile()
        case _:
            return DefaultChatModelProfile()


def build_chat_model_with_profile(settings: Settings) -> tuple[ChatModel, ModelProfile]:
    """Build a chat model and its model profile from settings.

    Combines provider dispatch and profile selection so entry points never
    need to coordinate the two steps manually.
    """
    profile = build_model_profile(settings.chat_model)
    provider: Literal["ollama", "openai_compatible"] = settings.chat_model_provider  # type: ignore[assignment]  # validated by settings pattern constraint
    match provider:
        case "ollama":
            model: ChatModel = build_ollama_chat_model(
                base_url=settings.chat_base_url,
                model=settings.chat_model,
                temperature=settings.model_temperature,
                seed=settings.model_seed,
            )
        case "openai_compatible":
            model = build_openai_compatible_chat_model(
                base_url=settings.chat_base_url,
                model=settings.chat_model,
                api_key=settings.chat_api_key,
                temperature=settings.model_temperature,
                seed=settings.model_seed,
            )
        case _:
            assert_never(provider)
    if profile.parse_text_tool_calls:
        model = TextToolCallParsingWrapper(model)
    return model, profile


def build_retrieval_tool(settings: Settings) -> RetrievalTool:
    """Build the full retrieval stack (text embedder + document store + retriever) from settings.

    Note that the settings currently only have qdrant-specific fields so we hardcoded
    creating a Qdrant-based retriever; if we add more retriever types in the future, we will
    do a match on the provider backend type here similar to the other builders.
    """
    text_embedder = build_text_embedder(
        TextEmbedderConfig(
            url=settings.embedding_base_url,
            embedding_model=settings.embedding_model,
            provider=settings.embedding_model_provider,  # type: ignore[arg-type]  # validated by settings pattern constraint
        )
    )
    retriever = build_qdrant_retriever(
        top_k=settings.retrieval_top_k,
        llm_top_k=settings.retrieval_llm_top_k,
        text_embedder=text_embedder,
        document_store=build_qdrant_document_store(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection=settings.qdrant_collection,
            embedding_dim=settings.embedding_dim,
        ),
    )
    return RetrievalTool(retriever=retriever)
