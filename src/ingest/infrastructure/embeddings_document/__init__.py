# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public embedding infrastructure API."""

from typing import Any, Protocol, runtime_checkable

from haystack.dataclasses import Document

from ._ollama import build_ollama_document_embedder


@runtime_checkable
class DocumentEmbedder(Protocol):
    """Structural boundary for document embedding components."""

    def run(self, documents: list[Document]) -> dict[str, Any]:
        """Embed documents and return a mapping with key ``documents``."""
        ...


__all__ = [
    "DocumentEmbedder",
    "build_ollama_document_embedder",
]
