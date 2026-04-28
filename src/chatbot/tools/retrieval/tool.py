"""LLM-callable tool that wraps the retrieval layer.

Modelling retrieval as a tool lets the LLM decide *when* to search and *how*
to formulate the query, enabling multi-hop retrieval and query reformulation
based on full conversation context — capabilities not possible with eager
pre-retrieval.
"""

import structlog
from pydantic import BaseModel, ValidationError

from src.chatbot.app.protocols import (
    JsonObject,
    Retriever,
    SourceChunk,
    ToolContext,
    ToolEvent,
    ToolSchema,
)

logger = structlog.get_logger(__name__)

_TOOL_NAME = "search_documents"


class _SearchInput(BaseModel):
    query: str


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
            description=(
                "Search the document corpus for information relevant to a query. "
                "Returns relevant text excerpts with source paths, chunk IDs, and similarity scores. "
                "Call this tool when the user asks about topics that may be covered in the "
                "uploaded documents."
            ),
            parameters_schema=_SearchInput.model_json_schema(mode="validation"),  # type: ignore[arg-type]
        )

    async def execute(
        self, args: JsonObject, context: ToolContext
    ) -> tuple[JsonObject, list[ToolEvent]]:
        """Retrieve chunks for *args[\"query\"]* and return them as structured JSON."""
        try:
            search_input = _SearchInput.model_validate(args)
        except ValidationError as exc:
            return {"error": f"Invalid arguments: {exc}"}, []

        sources: list[SourceChunk] = await self._retriever.retrieve(search_input.query)
        logger.info(
            "retrieval_tool.executed",
            query=search_input.query,
            chunks=len(sources),
        )
        if not sources:
            return {"chunks": [], "message": "No relevant documents found."}, []
        return {
            "chunks": [
                {
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "score": chunk.score,
                }
                for chunk in sources
            ]
        }, []
