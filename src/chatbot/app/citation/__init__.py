# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Citation: prompt augmentation, marker parsing, and validation.

This package owns the citation model implementation. Its public API is
deliberately narrow — only the types that are *produced* by this package
and consumed by callers outside it:

- :class:`CitationModel` — the main chat-model decorator
- :data:`CitationMessage` and its four concrete variants, which the
  orchestrator and tests inspect

Value objects (:class:`~src.chatbot.app.protocols.Citation`,
:class:`~src.chatbot.app.protocols.RawCitation`, etc.) and the
:class:`~src.chatbot.app.protocols_citeable_tool.CiteableTool` protocol live
in their respective ``protocols*`` modules; import them from there directly.
"""

from src.chatbot.app.citation.citation_model import CitationModel
from src.chatbot.app.citation.messages import (
    CitationAssistantMessage,
    CitationMessage,
    CitationSystemMessage,
    CitationToolMessage,
    CitationUserMessage,
)

__all__ = [
    "CitationModel",
    "CitationAssistantMessage",
    "CitationMessage",
    "CitationSystemMessage",
    "CitationToolMessage",
    "CitationUserMessage",
]
