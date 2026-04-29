"""LLM-callable tool that wraps the retrieval layer.

Modelling retrieval as a tool lets the LLM decide *when* to search and *how*
to formulate the query, enabling multi-hop retrieval and query reformulation
based on full conversation context — capabilities not possible with eager
pre-retrieval.
"""

import structlog
from openinference.semconv.trace import OpenInferenceMimeTypeValues, OpenInferenceSpanKindValues
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from pydantic import ValidationError

from src.chatbot.app.protocols import (
    JsonObject,
    Retriever,
    SourceChunk,
    ToolContext,
    ToolEvent,
    ToolSchema,
)
from src.chatbot.observability.openinference import (
    build_input_attributes,
    build_output_attributes,
    build_span_kind_attributes,
    build_tool_execution_attributes,
)
from src.chatbot.observability.schema import SPAN_CHAT_TOOL_SEARCH_DOCUMENTS
from src.chatbot.tools._input_model import ToolInputModel

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

_TOOL_NAME = "search_documents"


class _SearchInput(ToolInputModel):
    query: str


def _trace_request(*, span: trace.Span, args: JsonObject) -> None:
    span.set_attributes(build_span_kind_attributes(OpenInferenceSpanKindValues.TOOL))
    span.set_attributes(build_input_attributes(args, mime_type=OpenInferenceMimeTypeValues.JSON))


def _trace_error(
    *,
    span: trace.Span,
    args: JsonObject,
    exc: Exception,
    error_msg: str,
    error_result: JsonObject,
) -> None:
    span.set_attributes(build_tool_execution_attributes(tool_name=_TOOL_NAME, parameters=args))
    span.set_attributes(
        build_output_attributes(error_result, mime_type=OpenInferenceMimeTypeValues.JSON)
    )
    span.set_attribute("chat.tool.error", True)
    span.set_attribute("chat.tool.error_message", error_msg)
    span.record_exception(exc)
    span.set_status(StatusCode.ERROR, error_msg)


def _trace_response(*, span: trace.Span, validated_args: JsonObject, result: JsonObject) -> None:
    span.set_attributes(
        build_tool_execution_attributes(tool_name=_TOOL_NAME, parameters=validated_args)
    )
    span.set_attributes(build_output_attributes(result, mime_type=OpenInferenceMimeTypeValues.JSON))


class RetrievalTool:
    """LLM-callable tool wrapping the document retrieval layer.

    The LLM decides when and with what query to invoke this tool.  Retrieved
    chunks are returned as structured JSON so the model can ground its answer
    in the actual content and cite sources.

    Args:
        retriever: Retrieval backend satisfying the :class:`~src.chatbot.app.protocols.Retriever` Protocol.
    """

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever
        self.schema = ToolSchema(
            name=_TOOL_NAME,
            description="""Search the document corpus for information relevant to a query.

Call this tool when the user's request may be answered from the uploaded documents.
Returns relevant text chunks with source paths, chunk IDs, content, and similarity scores.""",
            parameters_schema=_SearchInput.model_json_schema(mode="validation"),  # type: ignore[arg-type]
        )

    async def execute(
        self, args: JsonObject, context: ToolContext
    ) -> tuple[JsonObject, list[ToolEvent]]:
        """Retrieve chunks for *args[\"query\"]* and return them as structured JSON."""
        with tracer.start_as_current_span(SPAN_CHAT_TOOL_SEARCH_DOCUMENTS) as span:
            _trace_request(span=span, args=args)
            try:
                search_input = _SearchInput.model_validate(args)
            except ValidationError as exc:
                error_msg = f"Invalid arguments: {exc}"
                error_result: JsonObject = {"error": error_msg}
                _trace_error(
                    span=span, args=args, exc=exc, error_msg=error_msg, error_result=error_result
                )
                return error_result, []

            sources: list[SourceChunk] = await self._retriever.retrieve(search_input.query)
            logger.info(
                "retrieval_tool.executed",
                query=search_input.query,
                chunks=len(sources),
            )
            result: JsonObject = (
                {
                    "chunks": [
                        {
                            "source": chunk.source,
                            "chunk_id": chunk.chunk_id,
                            "content": chunk.content,
                            "score": chunk.score,
                            "title": chunk.title,
                            "author": chunk.author,
                            "publication_date": chunk.publication_date,
                            "source_url": chunk.source_url,
                            "page": chunk.page,
                        }
                        for chunk in sources
                    ]
                }
                if sources
                else {"chunks": [], "message": "No relevant documents found."}
            )
            _trace_response(
                span=span, validated_args=search_input.model_dump(mode="json"), result=result
            )
            return result, []
