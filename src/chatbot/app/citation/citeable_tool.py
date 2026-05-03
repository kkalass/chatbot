"""``CiteableTool`` Protocol — extends ``Tool`` with citation responsibilities.

Tools that may be cited by the model implement this Protocol. The
:class:`CitationLayer` calls these methods to assemble the citation prompt
section, format tool results for the model history, and validate model-emitted
:class:`~src.chatbot.app.citation.models.RawCitation` payloads.
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
    """

    prompt_fragment: str


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
