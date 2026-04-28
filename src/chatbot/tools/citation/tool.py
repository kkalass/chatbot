"""LLM-callable tool that validates source citations against retrieved chunks.

The ``cite_sources`` tool is invoked during the citation pass after the main
agentic loop.  It cross-references claimed ``(source, chunk_id)`` pairs against the
``search_documents`` results present in conversation history, emits a
:class:`~src.chatbot.app.protocols.SourceCitationEvent` for validated sources, and
returns a structured result so the model has acknowledgement of what was accepted.
"""

import structlog
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from pydantic import BaseModel, ValidationError

from src.chatbot.app.citation_support import collect_search_chunks
from src.chatbot.app.protocols import (
    JsonObject,
    SourceChunk,
    SourceCitationEvent,
    ToolContext,
    ToolEvent,
    ToolSchema,
)
from src.chatbot.observability import to_attribute_text
from src.chatbot.observability.schema import SPAN_CHAT_TOOL_CITE_SOURCES
from src.chatbot.tools._input_model import ToolInputModel

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

_TOOL_NAME = "cite_sources"


class _CitationClaim(BaseModel):
    source: str
    chunk_id: str


class _CitationInput(ToolInputModel):
    citations: list[_CitationClaim]


class CitationTool:
    """LLM-callable tool that validates claimed source citations.

    During the citation pass the model calls this tool with the list of
    ``(source, chunk_id)`` pairs it referenced. The tool cross-references those
    pairs against all ``search_documents`` results in history, confirming which citations are
    grounded in actual retrieved evidence.

    Args:
        None — stateless; all inputs come from the model call arguments and
        the injected :class:`~src.chatbot.app.protocols.ToolContext`.
    """

    def __init__(self) -> None:
        self.schema = ToolSchema(
            name=_TOOL_NAME,
            description="""Declare which exact retrieved chunks you actually used in your answer.

Call this tool after answering with information from search_documents results.
Provide arguments as a JSON object with a citations array, for example:
{"citations":[{"source":"corpus/executive_order_14110.txt","chunk_id":"6ac85fb662c1fe7507171392be9e89350244bae72bc5f554c4fa44165288bad1"}]}

Important:
- use source+chunk_id pairs exactly as returned by search_documents
- citations must be a JSON array value
- citations must not be a JSON-encoded string
- citations must not be a schema object""",
            parameters_schema=_CitationInput.model_json_schema(mode="validation"),  # type: ignore[arg-type]
        )

    async def execute(
        self, args: JsonObject, context: ToolContext
    ) -> tuple[JsonObject, list[ToolEvent]]:
        """Validate *args["citations"]* against ``search_documents`` results in context history."""
        with tracer.start_as_current_span(SPAN_CHAT_TOOL_CITE_SOURCES) as span:
            span.set_attribute("chat.tool.arguments", to_attribute_text(args))
            try:
                citation_input = _CitationInput.model_validate(args)
            except ValidationError as exc:
                error_msg = f"Invalid arguments: {exc}"
                span.set_attribute("chat.tool.error", True)
                span.set_attribute("chat.tool.error_message", error_msg)
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, error_msg)
                return {"error": error_msg}, []

            available = collect_search_chunks(context.history)
            claimed = citation_input.citations

            validated_chunks: list[SourceChunk] = []
            validated: list[dict[str, str]] = []
            unvalidated: list[dict[str, str]] = []
            seen: set[tuple[str, str]] = set()

            for citation in claimed:
                key = (citation.source, citation.chunk_id)
                if key in seen:
                    continue
                seen.add(key)

                if key in available:
                    validated_chunks.append(available[key])
                    validated.append(
                        {
                            "source": citation.source,
                            "chunk_id": citation.chunk_id,
                        }
                    )
                else:
                    unvalidated.append(
                        {
                            "source": citation.source,
                            "chunk_id": citation.chunk_id,
                        }
                    )

            logger.info(
                "citation_tool.executed",
                claimed=len(claimed),
                validated=len(validated_chunks),
                unvalidated=len(unvalidated),
            )

            span.set_attribute("chat.tool.claimed", len(claimed))
            span.set_attribute("chat.tool.validated", len(validated_chunks))
            span.set_attribute("chat.tool.unvalidated", len(unvalidated))
            span.set_attribute("chat.tool.validated_pairs", to_attribute_text(validated))
            span.set_attribute("chat.tool.unvalidated_pairs", to_attribute_text(unvalidated))

            events: list[ToolEvent] = (
                [SourceCitationEvent(validated=tuple(validated_chunks))] if validated_chunks else []
            )
            return {
                "validated": validated,
                "unvalidated": unvalidated,
            }, events
