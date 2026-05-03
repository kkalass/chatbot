# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public embedding infrastructure API."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol, assert_never, runtime_checkable

from haystack.dataclasses import Document

from ._ollama import build_ollama_document_embedder


@dataclass(frozen=True)
class DocumentEmbedderConfig:
    """Construction-time config for a document (ingestion-time) embedder."""

    url: str
    embedding_model: str
    provider: Literal["ollama"] = "ollama"


@runtime_checkable
class DocumentEmbedder(Protocol):
    """Structural boundary for document embedding components."""

    def run(self, documents: list[Document]) -> dict[str, Any]:
        """Embed documents and return a mapping with key ``documents``."""
        ...


def build_document_embedder(config: DocumentEmbedderConfig) -> DocumentEmbedder:
    """Construct the document embedder prescribed by ``config.provider``."""
    match config.provider:
        case "ollama":
            return build_ollama_document_embedder(
                model=config.embedding_model,
                url=config.url,
            )
        case _:
            assert_never(config.provider)


__all__ = [
    "DocumentEmbedder",
    "DocumentEmbedderConfig",
    "build_document_embedder",
]
