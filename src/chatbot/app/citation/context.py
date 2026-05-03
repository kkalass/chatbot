# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only view of past tool results passed to ``CiteableTool.validate_and_enrich``.

The :class:`CitationContext` Protocol is the only coupling point between the
``tools`` package and the citation layer's history representation; tools never
import :mod:`src.chatbot.app.citation.messages` directly.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.chatbot.app.citation.messages import CitationLayerMessage, CitationLayerToolMessage
from src.chatbot.app.protocols import JsonObject


@runtime_checkable
class CitationContext(Protocol):
    """Read-only, narrow view of past tool results used during citation validation."""

    def tool_result_for(self, tool_call_id: str) -> JsonObject | None:
        """Return the raw JSON result for a specific prior tool call, or ``None``."""
        ...

    def tool_results_for(self, tool_name: str) -> Sequence[JsonObject]:
        """Return all prior tool-result JSON objects emitted by *tool_name*,
        in chronological (oldest-first) order."""
        ...


@dataclass(frozen=True)
class _DefaultCitationContext:
    """Thin adapter over the orchestrator's history of ``CitationLayerMessage``."""

    history: tuple[CitationLayerMessage, ...]

    def tool_result_for(self, tool_call_id: str) -> JsonObject | None:
        for msg in self.history:
            if isinstance(msg, CitationLayerToolMessage) and msg.tool_call_id == tool_call_id:
                return msg.result
        return None

    def tool_results_for(self, tool_name: str) -> Sequence[JsonObject]:
        return tuple(
            msg.result
            for msg in self.history
            if isinstance(msg, CitationLayerToolMessage) and msg.tool_name == tool_name
        )


def build_citation_context(history: Sequence[CitationLayerMessage]) -> CitationContext:
    """Build a default :class:`CitationContext` adapter over *history*."""
    return _DefaultCitationContext(history=tuple(history))
