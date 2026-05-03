"""Citation vocabulary: marker tokens, raw payloads, validated citations.

The :class:`CitationLayer` defines the citation format vocabulary as paired
typed subtypes — a ``RawCitation`` produced by the marker parser and a
corresponding ``Citation`` produced by ``CiteableTool.validate_and_enrich``.
"""

from dataclasses import dataclass
from typing import Literal, assert_never

from pydantic import BaseModel, ConfigDict

from src.chatbot.app.protocols import JsonObject

QUOTE_START_MARKER = "<°_quote_°>"
QUOTE_END_MARKER = "</°_quote_°>"


# ---------------------------------------------------------------------------
# RawCitation hierarchy — model output, parsed from marker JSON.
# ``raw_marker_text`` is filled by the parser with the complete marker block
# (start token + JSON payload + end token) so the orchestrator can splice it
# back into the LLM-side history exactly as the model emitted it.
# ---------------------------------------------------------------------------


class _RawCitationBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_marker_text: str = ""


class DocumentRawCitation(_RawCitationBase):
    """Marker payload for a claim grounded in a document-returning tool.

    Used by :class:`~src.chatbot.tools.retrieval.tool.RetrievalTool` and any
    other ``CiteableTool`` that returns retrievable chunks identified by
    ``(source, chunk_id)``.
    """

    kind: Literal["document"] = "document"
    tool_call_id: str
    source: str
    chunk_id: str
    quote_text: str | None = None
    claim: str | None = None


class ToolRawCitation(_RawCitationBase):
    """Marker payload for a claim grounded in a non-document tool result.

    The model is only required to copy ``tool_call_id`` verbatim; the
    authoritative tool name is resolved from history by the citation layer.
    """

    kind: Literal["tool_call"] = "tool_call"
    tool_call_id: str


type RawCitation = DocumentRawCitation | ToolRawCitation


# ---------------------------------------------------------------------------
# Citation hierarchy — validated and enriched by the responsible CiteableTool.
# Carries ``raw_marker_text`` from the originating ``RawCitation`` so the
# orchestrator can reconstruct the assistant's LLM-side text by splicing.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentCitation:
    """Validated citation backed by a retrieved document chunk."""

    raw_marker_text: str
    tool_call_id: str
    source: str
    chunk_id: str
    content: str
    score: float
    title: str | None = None
    author: str | None = None
    publication_date: str | None = None
    source_url: str | None = None
    page: str | None = None
    quote_text: str | None = None


@dataclass(frozen=True)
class ToolCitation:
    """Validated citation backed by a non-document tool result."""

    raw_marker_text: str
    tool_call_id: str
    tool_name: str
    result: JsonObject


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
class NumberedCitation:
    """A ``Citation`` with a stable per-turn reference number assigned by the
    orchestrator (``[N]`` in the rendered text). Reference numbers are reused
    when the same canonical key appears more than once in a turn.
    """

    reference_number: int
    citation: Citation


def canonical_key(citation: Citation) -> str:
    """Stable structural key for citation deduplication and reference reuse.

    Keys are derived deterministically from the validated citation so equality
    is exact. The orchestrator uses this key to assign a single
    ``reference_number`` per distinct cited evidence point per turn.
    """
    match citation:
        case DocumentCitation():
            return f"document:{citation.tool_call_id}:{citation.source}:{citation.chunk_id}"
        case ToolCitation():
            return f"tool:{citation.tool_call_id}"
        case _:
            assert_never(citation)
