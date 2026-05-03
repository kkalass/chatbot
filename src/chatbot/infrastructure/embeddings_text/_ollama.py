# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ollama embedder implementation helpers."""

from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder


def build_ollama_text_embedder(*, model: str, url: str) -> OllamaTextEmbedder:
    """Build an Ollama text embedder."""
    return OllamaTextEmbedder(model=model, url=url)
