# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public retrieval infrastructure API."""

from typing import assert_never

from src.chatbot.app.protocols import Retriever
from src.chatbot.infrastructure.embeddings_text import TextEmbedder

from ._config import RetrieverConfig
from ._qdrant import build_qdrant_retriever


def build_retriever(
    config: RetrieverConfig,
    text_embedder: TextEmbedder,
) -> Retriever:
    """Construct the retriever."""
    match config.store_backend:
        case "qdrant":
            return build_qdrant_retriever(config=config, text_embedder=text_embedder)
        case _:
            assert_never(config.store_backend)


__all__ = ["RetrieverConfig", "build_retriever"]
