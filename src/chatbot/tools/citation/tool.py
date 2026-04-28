"""LLM-callable tool that validates source citations against retrieved chunks.

The ``cite_sources`` tool is invoked during the citation pass after the main
agentic loop.  It cross-references claimed ``(source, chunk_id)`` pairs against the
``search_documents`` results present in conversation history, emits a
:class:`~src.chatbot.app.protocols.SourceCitationEvent` for validated sources, and
returns a structured result so the model has acknowledgement of what was accepted.
"""

from typing import cast

import structlog
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from pydantic import BaseModel, ValidationError

from src.chatbot.app.protocols import (
    ChatMessage,
    JsonObject,
    SourceChunk,
    SourceCitationEvent,
    ToolContext,
    ToolEvent,
    ToolSchema,
)
from src.chatbot.observability import to_attribute_text
from src.chatbot.observability.schema import SPAN_CHAT_TOOL_CITE_SOURCES

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

_TOOL_NAME = "cite_sources"
_SEARCH_TOOL_NAME = "search_documents"


class _CitationClaim(BaseModel):
    source: str
    chunk_id: str


class _CitationInput(BaseModel):
    citations: list[_CitationClaim]


def _collect_search_chunks(history: tuple[ChatMessage, ...]) -> dict[tuple[str, str], SourceChunk]:
    """Build a map from ``(source, chunk_id)`` to :class:`SourceChunk`.

    Matches tool result messages by correlating assistant ``tool_calls``
    entries named ``search_documents`` with their corresponding ``role="tool"``
    responses via ``call_id`` / ``tool_call_id``.
    """
    search_call_ids: set[str] = set()
    for msg in history:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.name == _SEARCH_TOOL_NAME:
                    search_call_ids.add(tc.call_id)

    chunks: dict[tuple[str, str], SourceChunk] = {}
    for msg in history:
        if msg.role != "tool":
            continue
        if msg.tool_call_id is None or msg.tool_call_id not in search_call_ids:
            continue
        if not isinstance(msg.content, dict):
            continue
        raw_chunks = msg.content.get("chunks")
        if not isinstance(raw_chunks, list):
            continue
        for item in cast(list[object], raw_chunks):
            if not isinstance(item, dict):
                continue
            item_dict = cast(dict[str, object], item)
            source = item_dict.get("source")
            if not isinstance(source, str):
                continue
            chunk = _json_to_source_chunk(item_dict)
            key = (source, chunk.chunk_id)
            if key not in chunks or chunk.score > chunks[key].score:
                chunks[key] = chunk

    return chunks


def _json_to_source_chunk(data: dict[str, object]) -> SourceChunk:
    """Reconstruct a :class:`SourceChunk` from the JSON stored in tool result history."""

    def _opt_str(key: str) -> str | None:
        v = data.get(key)
        return str(v) if v is not None else None

    score_raw = data.get("score")
    score = float(score_raw) if isinstance(score_raw, (int, float)) else 0.0  # type: ignore[arg-type]

    return SourceChunk(
        content=str(data.get("content", "")),
        source=str(data.get("source", "")),
        score=score,
        chunk_id=str(data.get("chunk_id", "")),
        title=_opt_str("title"),
        author=_opt_str("author"),
        publication_date=_opt_str("publication_date"),
        source_url=_opt_str("source_url"),
    )


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
            description=(
                "Declare which exact retrieved chunks you referenced in your answer. "
                "Call this tool after providing an answer that used search_documents results. "
                "Provide citations as source+chunk_id pairs exactly as returned by search_documents."
            ),
            parameters_schema=_CitationInput.model_json_schema(mode="validation"),  # type: ignore[arg-type]
        )

    async def execute(
        self, args: JsonObject, context: ToolContext
    ) -> tuple[JsonObject, list[ToolEvent]]:
        """Validate *args["citations"]* against ``search_documents`` results in context history."""
        with tracer.start_as_current_span(SPAN_CHAT_TOOL_CITE_SOURCES) as span:
            try:
                citation_input = _CitationInput.model_validate(args)
            except ValidationError as exc:
                error_msg = f"Invalid arguments: {exc}"
                span.set_attribute("chat.tool.error", True)
                span.set_attribute("chat.tool.error_message", error_msg)
                span.set_attribute("chat.tool.arguments", to_attribute_text(args))
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, error_msg)
                return {"error": error_msg}, []

            available = _collect_search_chunks(context.history)
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
