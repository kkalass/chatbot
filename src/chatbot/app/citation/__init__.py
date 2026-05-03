"""Citation layer: prompt augmentation, marker parsing, and validation.

This package owns the entire citation concern. The orchestrator depends on
:class:`~src.chatbot.app.citation.layer.CitationLayer` (a decorator around a
:class:`~src.chatbot.app.protocols.ChatModel`) and never touches markers,
parsers, or per-tool citation logic directly.

Public API:
    - :class:`CitationLayer`
    - :class:`CiteableTool`, :class:`CiteInstructions`
    - :class:`CitationContext`
    - :class:`RawCitation`
    - :data:`Citation`, :class:`DocumentCitation`, :class:`ToolCitation`
    - :class:`NumberedCitation`, :class:`HallucinatedCitation`, :class:`UnsubstantiatedClaim`
    - :data:`CitationLayerMessage` and its variants
"""

from src.chatbot.app.citation.citeable_tool import CiteableTool, CiteInstructions
from src.chatbot.app.citation.context import CitationContext, build_citation_context
from src.chatbot.app.citation.layer import CitationLayer
from src.chatbot.app.citation.messages import (
    CitationLayerAssistantMessage,
    CitationLayerMessage,
    CitationLayerSystemMessage,
    CitationLayerToolMessage,
    CitationLayerUserMessage,
)
from src.chatbot.app.citation.models import (
    QUOTE_END_MARKER,
    QUOTE_START_MARKER,
    Citation,
    DocumentCitation,
    HallucinatedCitation,
    NumberedCitation,
    RawCitation,
    ToolCitation,
    UnsubstantiatedClaim,
    canonical_key,
)

__all__ = [
    "QUOTE_END_MARKER",
    "QUOTE_START_MARKER",
    "Citation",
    "CitationContext",
    "CitationLayer",
    "CitationLayerAssistantMessage",
    "CitationLayerMessage",
    "CitationLayerSystemMessage",
    "CitationLayerToolMessage",
    "CitationLayerUserMessage",
    "CiteInstructions",
    "CiteableTool",
    "DocumentCitation",
    "HallucinatedCitation",
    "NumberedCitation",
    "RawCitation",
    "ToolCitation",
    "UnsubstantiatedClaim",
    "build_citation_context",
    "canonical_key",
]
