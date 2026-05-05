# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Citation-layer history messages with pre-computed LLM-side content.

Each variant carries the LLM-ready ``llm_content`` string alongside any
citation metadata. :class:`CitationModel` is the factory for these
messages — see :mod:`src.chatbot.app.citation.citation_model`.
"""

from dataclasses import dataclass

from src.chatbot.app.protocols import (
    Citation,
    HallucinatedCitation,
    JsonObject,
    ToolCallInfo,
    UnsubstantiatedClaim,
)
from src.chatbot.app.protocols_citeable_tool import CitableUnit


@dataclass(frozen=True)
class CitationSystemMessage:
    """System turn. ``llm_content`` is the orchestrator-supplied profile-adjusted
    base prompt with citation instructions appended by the citation layer.
    """

    llm_content: str


@dataclass(frozen=True)
class CitationUserMessage:
    """User turn. ``llm_content`` is ``<reminder> + Prompts.user_message(text)``
    as produced by :meth:`CitationModel.make_user_message`.
    """

    llm_content: str


@dataclass(frozen=True)
class CitationAssistantMessage:
    """Assistant turn — the only variant that retains structured ``parts``.

    ``parts`` preserves the streaming order of text fragments, validated
    citations, and hallucinated citations so that ``raw_marker_text`` can be
    spliced back at the correct positions to reconstruct ``llm_content``.

    ``tool_calls`` carries any tool invocations the model emitted at the end
    of this turn (mutually exclusive with a final text-only assistant message).
    """

    parts: tuple[str | Citation | HallucinatedCitation | UnsubstantiatedClaim, ...]
    llm_content: str
    tool_calls: tuple[ToolCallInfo, ...] | None = None


@dataclass(frozen=True)
class CitationToolMessage:
    """Tool-result turn correlated to a prior assistant tool call.

    ``llm_content`` is pre-computed via the responsible
    :class:`~src.chatbot.app.citation.citeable_tool.CiteableTool`'s
    ``render_for_history`` (or the layer's generic wrapper when no such tool
    is registered).

    ``units`` carries the citable units embedded in ``llm_content`` so the
    citation layer can rebuild its global token → unit index for any
    historical assistant turn.
    """

    tool_call_id: str
    tool_name: str
    result: JsonObject
    llm_content: str
    units: tuple[CitableUnit, ...] = ()


type CitationMessage = (
    CitationSystemMessage | CitationUserMessage | CitationAssistantMessage | CitationToolMessage
)
