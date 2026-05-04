# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Citation layer: prompt augmentation, marker parsing, and validation.

This package owns the citation layer implementation. Its public API is
deliberately narrow — only the types that are *produced* by this package
and consumed by callers outside it:

- :class:`CitationLayer` — the main chat-model decorator
- :data:`CitationLayerMessage` and its four concrete variants, which the
  orchestrator and tests inspect

Value objects (:class:`~src.chatbot.app.protocols.Citation`,
:class:`~src.chatbot.app.protocols.RawCitation`, etc.) and the
:class:`~src.chatbot.app.protocols_citeable_tool.CiteableTool` protocol live
in their respective ``protocols*`` modules; import them from there directly.
"""

from src.chatbot.app.citation.layer import CitationLayer
from src.chatbot.app.citation.messages import (
    CitationLayerAssistantMessage,
    CitationLayerMessage,
    CitationLayerSystemMessage,
    CitationLayerToolMessage,
    CitationLayerUserMessage,
)

__all__ = [
    "CitationLayer",
    "CitationLayerAssistantMessage",
    "CitationLayerMessage",
    "CitationLayerSystemMessage",
    "CitationLayerToolMessage",
    "CitationLayerUserMessage",
]
