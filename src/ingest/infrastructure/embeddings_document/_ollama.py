# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ollama embedder implementation helpers."""

from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder


def build_ollama_document_embedder(*, model: str, url: str) -> OllamaDocumentEmbedder:
    """Build an Ollama document embedder."""
    return OllamaDocumentEmbedder(model=model, url=url)
