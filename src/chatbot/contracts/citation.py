# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Citation vocabulary and CiteableTool Protocol.

Single module so tool implementors depend on one citation contract surface
(rather than two: protocols + protocols_citeable_tool as in the prior split).
"""

from dataclasses import dataclass
from typing import Protocol, assert_never, runtime_checkable

from pydantic import BaseModel, ConfigDict

from src.chatbot.contracts.i18n import I18nMessage, JsonObject
from src.chatbot.contracts.tools import Tool

# ---------------------------------------------------------------------------
# Wire-format markers — the delimiter pair that wraps every citation marker
# in the model's output stream.  Defined in this contract module (not in
# citation/) so that tool implementors can embed them in their
# ``cite_instructions`` ``prompt_fragment`` without importing citation internals.
# ---------------------------------------------------------------------------

QUOTE_START_MARKER = "<°_quote_°>"
QUOTE_END_MARKER = "</°_quote_°>"


class RawCitation(BaseModel):
    """Marker payload emitted by the model.

    ``ref`` is the citation token of a previously-emitted citable unit; it is
    required for all regular citations. For unsubstantiated claims the model
    emits ``{"kind": "unsubstantiated"}`` without a ``ref``.
    """

    model_config = ConfigDict(frozen=True)

    ref: str | None = None
    kind: str | None = None
    raw_marker_text: str = ""


@dataclass(frozen=True)
class DocumentCitation:
    """Validated citation backed by a retrieved document chunk."""

    raw_marker_text: str
    citation_token: str
    source: str
    chunk_id: str
    content: str
    score: float
    title: str | None = None
    author: str | None = None
    publication_date: str | None = None
    source_url: str | None = None
    page: str | None = None
    kind: str = "text"
    image_path: str | None = None


@dataclass(frozen=True)
class ToolCitation:
    """Validated citation backed by a non-document tool result."""

    raw_marker_text: str
    citation_token: str
    tool_name: str
    result: JsonObject
    display_name: I18nMessage | None = None  # resolved by UI via translation map


type Citation = DocumentCitation | ToolCitation


@dataclass(frozen=True)
class HallucinatedCitation:
    """A ``RawCitation`` that failed validation by the responsible tool.

    The UI decides how (or whether) to surface it. ``raw_marker_text`` is also
    spliced back into the LLM-side history so that the model sees its own
    output verbatim on subsequent turns.
    """

    raw: RawCitation
    reason: str

    @property
    def raw_marker_text(self) -> str:
        return self.raw.raw_marker_text


@dataclass(frozen=True)
class UnsubstantiatedClaim:
    """A ``RawCitation`` with ``kind="unsubstantiated"`` — the model explicitly
    signals that no tool output supports the preceding claim.

    This is *not* a validation failure: it is correct, transparent model
    behaviour. The UI renders it as ``_(unbelegt)_`` inline at the marker
    position.
    """

    raw: RawCitation

    @property
    def raw_marker_text(self) -> str:
        return self.raw.raw_marker_text


@dataclass(frozen=True)
class NumberedCitation:
    """A ``Citation`` with a stable per-session reference number assigned by the
    orchestrator (``[N]`` in the rendered text). Reference numbers are reused
    when the same canonical key appears more than once within or across turns.
    """

    reference_number: int
    citation: Citation


def canonical_key(citation: Citation) -> str:
    """Stable session-scoped key for citation deduplication and reference reuse."""
    match citation:
        case DocumentCitation():
            return f"document:{citation.citation_token}"
        case ToolCitation():
            return f"tool:{citation.citation_token}"
        case _:
            assert_never(citation)


# ---------------------------------------------------------------------------
# CiteableTool Protocol (was: src.chatbot.app.protocols_citeable_tool)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CiteInstructions:
    """Citation prompt fragment contributed by a ``CiteableTool``.

    The ``prompt_fragment`` is appended to the system prompt's citation
    section; it should describe what the tool's rendered output looks like
    and where the model finds the ``citation_token`` it must copy into a
    marker.  Use :data:`QUOTE_START_MARKER` and :data:`QUOTE_END_MARKER` in
    the examples so the model sees the exact delimiters it must emit.

    ``reminder_fragment``, if provided, is included in the per-turn user
    reminder for tool-specific attribution rules that are easy to forget
    (e.g. chunk-content verification for document retrieval).
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
    in :meth:`CiteableTool.enrich` to build the typed :data:`Citation`.
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
    """A :class:`~src.chatbot.contracts.tools.Tool` whose results may be cited."""

    def cite_instructions(self) -> CiteInstructions:
        """Return the citation prompt fragment contributed by this tool."""
        ...

    def render_for_history(self, result: JsonObject) -> ToolHistoryRendering:
        """Render *result* for the LLM-side history and expose its citable units."""
        ...

    def enrich(self, raw: RawCitation, unit: CitableUnit) -> Citation:
        """Materialise a typed :data:`Citation` from a resolved ``CitableUnit``."""
        ...
