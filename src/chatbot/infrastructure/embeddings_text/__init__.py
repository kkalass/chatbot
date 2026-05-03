# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public embedding infrastructure API."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol, assert_never

from ._ollama import build_ollama_text_embedder


@dataclass(frozen=True)
class TextEmbedderConfig:
    """Construction-time config for a query (retrieval-time) text embedder."""

    url: str
    embedding_model: str
    provider: Literal["ollama"] = "ollama"


class TextEmbedder(Protocol):
    """Structural boundary for query embedding components."""

    def run(self, text: str) -> dict[str, Any]:
        """Embed text and return a mapping with key ``embedding``."""
        ...


def build_text_embedder(config: TextEmbedderConfig) -> TextEmbedder:
    """Construct the query text embedder prescribed by ``config.provider``."""
    match config.provider:
        case "ollama":
            return build_ollama_text_embedder(
                model=config.embedding_model,
                url=config.url,
            )
        case _:
            assert_never(config.provider)


__all__ = [
    "TextEmbedder",
    "TextEmbedderConfig",
    "build_text_embedder",
]
