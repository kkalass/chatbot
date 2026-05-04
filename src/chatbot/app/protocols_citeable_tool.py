# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``CiteableTool`` Protocol — extends ``Tool`` with custom citation behaviour.

This module defines the public contract that tool implementations must satisfy
to participate in the citation system.  It lives next to
:mod:`~src.chatbot.app.protocols` (not inside ``citation/``) so that tool
implementors depend only on this module and never on citation internals.

Citation flow (single shape across all tools):

1. After execution, the citation layer asks the tool to render its result for
   the LLM-side history. The tool returns a :class:`ToolHistoryRendering`
   bundling the rendered text *and* the set of :class:`CitableUnit`\\s
   embedded in it. Each unit carries a ``citation_token`` that MUST appear
   verbatim in the rendered text (typically as a ``citation_token=""`` XML
   attribute) so the model can copy it into a marker payload.

2. The layer maintains a global token → ``(tool, CitableUnit)`` index across
   all prior tool results in the conversation. When the model emits a marker
   ``{"ref": "<token>"}``, the layer resolves the token, looks up the owning
   tool, and calls :meth:`CiteableTool.enrich` to materialise the typed
   :class:`~src.chatbot.app.protocols.Citation`. Token existence is the
   layer's invariant; semantic correctness (does the cited content actually
   support the claim?) is the model's responsibility and only checked
   downstream.

Plain :class:`~src.chatbot.app.protocols.Tool` implementations that do not
implement this Protocol are auto-handled by the layer with a generic
rendering and a generic :class:`~src.chatbot.app.protocols.ToolCitation`
enrichment — no per-tool code required for tools without specialised citation
needs.

Wire-format markers
-------------------
:data:`QUOTE_START_MARKER` and :data:`QUOTE_END_MARKER` are part of this
protocol module because ``CiteableTool`` implementors **must** know them: the
``prompt_fragment`` returned by :meth:`CiteableTool.cite_instructions` is
injected into the system prompt and should contain realistic marker examples
so the model understands the expected format.  Without the exact marker
strings the examples would be useless or misleading.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.chatbot.app.protocols import Citation, JsonObject, RawCitation, Tool

# ---------------------------------------------------------------------------
# Wire-format markers — the delimiter pair that wraps every citation marker
# in the model's output stream.  The parser and the citation layer use these
# to detect and extract JSON payloads.  Defined here (not in citation/) so
# that tool implementors can embed them in their ``cite_instructions``
# ``prompt_fragment`` without importing citation internals.
# ---------------------------------------------------------------------------

QUOTE_START_MARKER = "<°_quote_°>"
QUOTE_END_MARKER = "</°_quote_°>"


@dataclass(frozen=True)
class CiteInstructions:
    """Citation prompt fragment contributed by a ``CiteableTool``.

    The ``prompt_fragment`` is appended to the system prompt's citation
    section; it should describe what the tool's rendered output looks like
    and where the model finds the ``citation_token`` it must copy into a
    marker.  Use :data:`QUOTE_START_MARKER` and :data:`QUOTE_END_MARKER` in
    the examples so the model sees the exact delimiters it must emit.  The
    marker shape itself is universal (``{"ref": "<token>"}``) and is
    documented once by the citation layer — fragments must not redefine it.

    ``reminder_fragment``, if provided, is included in the per-turn user
    reminder for tool-specific attribution rules that are easy to forget
    (e.g. chunk-content verification for document retrieval). Keep it short.
    """

    prompt_fragment: str
    reminder_fragment: str | None = None


@dataclass(frozen=True)
class CitableUnit:
    """One citable item produced by a tool result.

    ``citation_token`` is the opaque string the model must copy into a marker
    ``ref`` to cite this unit. It MUST be embedded verbatim in the LLM-visible
    rendering returned by :meth:`CiteableTool.render_for_history`.

    ``payload`` is opaque to the citation layer; the owning tool consumes it
    in :meth:`CiteableTool.enrich` to build the typed
    :class:`~src.chatbot.app.protocols.Citation`. Use a domain value object
    here (e.g. ``SourceChunk``) — never the raw tool arguments or wire payload.
    """

    citation_token: str
    payload: object


@dataclass(frozen=True)
class ToolHistoryRendering:
    """LLM-visible rendering of a tool result plus its citable units.

    ``llm_content`` is the exact string the model will see for this tool
    result on subsequent turns. It MUST embed the ``citation_token`` of every
    unit in ``units`` (typically as a ``citation_token=""`` attribute on the
    enclosing element) so the model can locate and copy the token.
    """

    llm_content: str
    units: tuple[CitableUnit, ...]


@runtime_checkable
class CiteableTool(Tool, Protocol):
    """A :class:`~src.chatbot.app.protocols.Tool` whose results may be cited."""

    def cite_instructions(self) -> CiteInstructions:
        """Return the citation prompt fragment contributed by this tool."""
        ...

    def render_for_history(self, result: JsonObject) -> ToolHistoryRendering:
        """Render *result* for the LLM-side history and expose its citable units.

        Owns the LLM-side rendering of this tool's output (e.g. structured XML
        for retrieval chunks). Called once per tool result by
        :meth:`~src.chatbot.app.citation.citation_model.CitationModel.make_tool_message`.
        """
        ...

    def enrich(self, raw: RawCitation, unit: CitableUnit) -> Citation:
        """Materialise a typed :class:`~src.chatbot.app.protocols.Citation` from a resolved ``CitableUnit``.

        Called by the citation layer after it resolved ``raw.ref`` to a
        ``CitableUnit`` previously produced by this tool. The unit's
        ``payload`` is exactly the value the tool stored when emitting the
        unit. Implementations should not perform existence checks here — the
        layer guarantees that ``unit`` was produced by *this* tool for this
        conversation.
        """
        ...
