"""``CiteableTool`` Protocol — extends ``Tool`` with custom citation behavior.

The :class:`CitationLayer` already supports default ``tool_call`` citations for
all tools. Implement this Protocol only when a tool needs specialized prompt
instructions, history rendering, or validation/enrichment logic (for example
document-level citations with ``chunk_id``).
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.chatbot.app.citation.context import CitationContext
from src.chatbot.app.citation.models import Citation, RawCitation
from src.chatbot.app.protocols import JsonObject, Tool


@dataclass(frozen=True)
class CiteInstructions:
    """Citation prompt fragment contributed by a ``CiteableTool``.

    The fragment must be self-contained: it documents which
    ``RawCitation`` subtype the model should emit when citing this tool, what
    the JSON schema inside the marker block looks like, and which fields of
    the tool's result payload feed into that schema. The
    :class:`CitationLayer` concatenates fragments from all registered
    ``CiteableTool``s into the citation section of the system prompt.

    ``reminder_fragment``, if provided, is included in the per-turn user
    reminder so the model is nudged on every turn with tool-specific
    attribution rules that are easy to forget (e.g. chunk-content
    verification for document retrieval). Keep it short — one to three
    sentences.
    """

    prompt_fragment: str
    reminder_fragment: str | None = None


@runtime_checkable
class CiteableTool(Tool, Protocol):
    """A :class:`~src.chatbot.app.protocols.Tool` whose results may be cited."""

    def cite_instructions(self) -> CiteInstructions:
        """Return the citation prompt fragment contributed by this tool."""
        ...

    def format_for_history(self, result: JsonObject) -> str:
        """Render a tool result as the string the model will see in subsequent turns.

        Owns the LLM-side rendering of this tool's output (e.g. structured XML
        for retrieval chunks, or compact JSON for scalar tool results). Called
        once per tool result by :meth:`CitationLayer.make_tool_message`.
        """
        ...

    def validate_and_enrich(
        self,
        raw: RawCitation,
        context: CitationContext,
    ) -> Citation | None:
        """Validate a marker payload and enrich it into a typed :class:`Citation`.

        Returns ``None`` when the payload cannot be reconciled with this tool's
        prior results; the citation layer will then surface a
        :class:`~src.chatbot.app.citation.models.HallucinatedCitation` to the
        orchestrator.
        """
        ...
