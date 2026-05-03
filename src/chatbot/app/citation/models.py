"""Citation vocabulary: marker tokens, raw payloads, validated citations.

The :class:`CitationLayer` defines the citation format vocabulary as a single
:class:`RawCitation` produced by the marker parser and typed
:class:`Citation` subtypes produced by ``CiteableTool.validate_and_enrich``.
"""

from dataclasses import dataclass
from typing import assert_never

from pydantic import BaseModel, ConfigDict

from src.chatbot.app.protocols import JsonObject

QUOTE_START_MARKER = "<°_quote_°>"
QUOTE_END_MARKER = "</°_quote_°>"


# ---------------------------------------------------------------------------
# RawCitation — model output, parsed from marker JSON.
# ``raw_marker_text`` is filled by the parser with the complete marker block
# (start token + JSON payload + end token) so the orchestrator can splice it
# back into the LLM-side history exactly as the model emitted it.
# ---------------------------------------------------------------------------


class RawCitation(BaseModel):
    """Marker payload emitted by the model.

    ``tool_call_id`` is required for all regular citations. For unsubstantiated
    claims the model emits ``{"kind": "unsubstantiated"}`` without a
    ``tool_call_id``; the parser injects a sentinel ``tool_call_id=""`` so that
    ``model_validate`` succeeds, and ``_validate`` short-circuits on ``kind``
    before consulting the tool-call lookup.

    ``chunk_id`` is present for document-level citations only.

    The :class:`~src.chatbot.app.citation.layer.CitationLayer` routes
    validation to the ``CiteableTool`` registered for the cited
    ``tool_call_id``, falling back to a generic
    :class:`~src.chatbot.app.citation.models.ToolCitation` for tools without
    custom citation logic. Unknown marker fields are silently ignored.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    tool_call_id: str
    chunk_id: str | None = None
    kind: str | None = None
    raw_marker_text: str = ""


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
class UnsubstantiatedClaim:
    """A ``RawCitation`` with ``kind="unsubstantiated"`` — the model explicitly
    signals that no tool output supports the preceding claim.

    This is *not* a validation failure: it is correct, transparent model
    behaviour. The UI renders it as ``_(unbelegt)_`` inline at the marker
    position. ``raw_marker_text`` is spliced back into the LLM-side history
    so the model sees its own signal on subsequent turns.
    """

    raw: RawCitation

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
            return f"document:{citation.tool_call_id}:{citation.chunk_id}"
        case ToolCitation():
            return f"tool:{citation.tool_call_id}"
        case _:
            assert_never(citation)
