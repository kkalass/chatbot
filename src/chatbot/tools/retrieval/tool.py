# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LLM-callable, citeable retrieval tool.

Modelling retrieval as a tool lets the LLM decide *when* to search and *how*
to formulate the query, enabling multi-hop retrieval and query reformulation
based on full conversation context — capabilities not possible with eager
pre-retrieval.

This tool implements :class:`~src.chatbot.app.citation.CiteableTool`: it owns
the LLM-side rendering of search results, contributes its own citation prompt
fragment, and validates :class:`~src.chatbot.app.citation.RawCitation` payloads
emitted by the model against its own past tool outputs.
"""

import html
import re
from enum import StrEnum
from typing import cast

import structlog
from openinference.semconv.trace import OpenInferenceMimeTypeValues, OpenInferenceSpanKindValues
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from pydantic import ValidationError

from src.chatbot.app.protocols import (
    Citation,
    DocumentCitation,
    I18nMessage,
    JsonObject,
    RawCitation,
    Retriever,
    SourceChunk,
    ToolSchema,
)
from src.chatbot.app.protocols_citeable_tool import (
    QUOTE_END_MARKER,
    QUOTE_START_MARKER,
    CitableUnit,
    CiteInstructions,
    ToolHistoryRendering,
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


class RetrievalCallKey(StrEnum):
    """Message keys for :meth:`RetrievalTool.describe_call` results and display name.

    The UI translation map must contain an entry for every value defined here.
    """

    DISPLAY_NAME = "retrieval.display_name"
    SEARCHING = "retrieval.searching"


class _SearchInput(ToolInputModel):
    query: str


_CITE_FRAGMENT = f"""#### {_TOOL_NAME}

`{_TOOL_NAME}` results are rendered as XML where each retrieved chunk
carries a ``citation_token`` attribute:

    <chunk citation_token="<token>" page="...">
      ...chunk content...
    </chunk>

To cite a chunk, copy that exact ``citation_token`` value into the marker
``ref``:

    {QUOTE_START_MARKER}{{"ref":"<token>"}}{QUOTE_END_MARKER}

Only cite a chunk if the claim is directly supported by the text in that
chunk's inner content — do not cite a chunk whose text does not contain
the information being stated. Read the chunk content first, then copy
that chunk's ``citation_token``.

A common failure mode is to default to the first ``<chunk>`` of a source.
Do not do this. The token must come from the specific ``<chunk>`` element
whose inner text supports the sentence — not from the first chunk or a
sibling chunk of the same ``<source>``.

Correct vs. incorrect attribution example:

    Given:
        <source ...>
            <chunk citation_token="AAA">Job losses are likely in some sectors.</chunk>
            <chunk citation_token="BBB">AI can also create new jobs and raise productivity.</chunk>
        </source>

    CORRECT (claim about new jobs cites BBB, whose content supports it):
        AI may create new jobs. {QUOTE_START_MARKER}{{"ref":"BBB"}}{QUOTE_END_MARKER}

    INCORRECT (claim about new jobs cites AAA, the first chunk, whose content is about job losses):
        AI may create new jobs. {QUOTE_START_MARKER}{{"ref":"AAA"}}{QUOTE_END_MARKER}"""

_CITE_REMINDER = (
    f"For {_TOOL_NAME}: only cite a chunk whose content actually contains the stated "
    "information — verify the chunk text matches the claim before copying its "
    "citation_token. Do not default to the first chunk of a source."
)


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


def _render_chunks_as_xml(content: JsonObject) -> ToolHistoryRendering:
    """Render search-result chunks as structured XML for model consumption.

    Each ``<chunk>`` carries a ``citation_token`` attribute (the content-hashed
    ``chunk_id``) immediately adjacent to its content, so smaller models can
    correctly attribute claims to the right chunk rather than picking the
    first chunk in the list. The same token is exposed as a
    :class:`CitableUnit` so the citation layer can resolve a model-emitted
    ``ref`` back to the originating :class:`SourceChunk`.
    """
    chunks_raw: object = content.get("chunks")
    if not isinstance(chunks_raw, list) or not chunks_raw:
        message: object = content.get("message", "No relevant documents found.")
        return ToolHistoryRendering(llm_content=str(message), units=())

    grouped_by_source: dict[str, list[dict[str, object]]] = {}
    for raw_chunk in cast(list[object], chunks_raw):
        if not isinstance(raw_chunk, dict):
            continue
        chunk_data = cast(dict[str, object], raw_chunk)
        source = str(chunk_data.get("source", "unknown"))
        grouped_by_source.setdefault(source, []).append(chunk_data)

    parts: list[str] = ["<search_results>\n"]
    units: list[CitableUnit] = []
    for source, source_chunks in grouped_by_source.items():
        title = source
        author: str | None = None
        year: str | None = None

        for chunk_data in source_chunks:
            raw_title = chunk_data.get("title")
            if raw_title is not None and str(raw_title).strip():
                title = str(raw_title).strip()
                break

        for chunk_data in source_chunks:
            raw_author = chunk_data.get("author")
            if raw_author is not None and str(raw_author).strip():
                author = str(raw_author).strip()
                break

        for chunk_data in source_chunks:
            raw_publication_date = chunk_data.get("publication_date")
            if raw_publication_date is None:
                continue
            match = re.search(r"(?:19|20)\d{2}", str(raw_publication_date))
            if match is not None:
                year = match.group(0)
                break

        title_attr = html.escape(title, quote=True)
        source_attr = html.escape(source, quote=True)
        year_attr = html.escape(year or "unknown", quote=True)
        author_attr = html.escape(author or "unknown", quote=True)
        parts.append(
            f'  <source title="{title_attr}" source_path="{source_attr}" '
            f'year="{year_attr}" author="{author_attr}">\n'
        )

        for chunk_data in source_chunks:
            chunk = _json_to_source_chunk(chunk_data)
            if not chunk.chunk_id:
                continue
            page: object = chunk_data.get("page")
            chunk_id_attr = html.escape(chunk.chunk_id, quote=True)
            page_attr = html.escape(str(page), quote=True) if page is not None else "unknown"
            content_escaped = html.escape(chunk.content)

            parts.append(f'    <chunk citation_token="{chunk_id_attr}" page="{page_attr}">\n')
            parts.append(f"      {content_escaped}\n")
            parts.append("    </chunk>\n")
            units.append(CitableUnit(citation_token=chunk.chunk_id, payload=chunk))

        parts.append("  </source>\n")

    parts.append("</search_results>")
    return ToolHistoryRendering(llm_content="".join(parts), units=tuple(units))


def _json_to_source_chunk(data: dict[str, object]) -> SourceChunk:
    def _opt_str(key: str) -> str | None:
        value = data.get(key)
        return str(value) if value is not None else None

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
        page=_opt_str("page"),
    )


class RetrievalTool:
    """LLM-callable, citeable tool wrapping the document retrieval layer.

    Implements :class:`~src.chatbot.app.citation.CiteableTool`: the tool owns
    its own LLM-side history rendering, citation prompt fragment, and
    raw-citation validation logic.

    Args:
        retriever: Retrieval backend satisfying the
            :class:`~src.chatbot.app.protocols.Retriever` Protocol.
    """

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever
        self.display_name = I18nMessage(key=RetrievalCallKey.DISPLAY_NAME, args={})
        self.schema = ToolSchema(
            name=_TOOL_NAME,
            description="""Search the document corpus for information relevant to a query.

Call this tool when the user's request may be answered from the uploaded documents.
Returns relevant text chunks with source paths, chunk IDs, content, and similarity scores.
Note that the search is an embedding based vector search, not a keyword search.
""",
            parameters_schema=_SearchInput.model_json_schema(mode="validation"),  # type: ignore[arg-type]
        )

    async def execute(self, args: JsonObject) -> JsonObject:
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
                return error_result

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
            return result

    # --- CiteableTool ------------------------------------------------

    def describe_call(self, args: JsonObject) -> I18nMessage:
        query = str(args.get("query", ""))
        return I18nMessage(key=RetrievalCallKey.SEARCHING, args={"query": query})

    def cite_instructions(self) -> CiteInstructions:
        return CiteInstructions(prompt_fragment=_CITE_FRAGMENT, reminder_fragment=_CITE_REMINDER)

    def render_for_history(self, result: JsonObject) -> ToolHistoryRendering:
        return _render_chunks_as_xml(result)

    def enrich(self, raw: RawCitation, unit: CitableUnit) -> Citation:
        chunk = unit.payload
        # The citation layer guarantees ``unit`` was produced by this tool's
        # ``render_for_history`` and therefore carries a SourceChunk payload.
        assert isinstance(chunk, SourceChunk), (
            "RetrievalTool only produces CitableUnits with SourceChunk payloads"
        )
        return DocumentCitation(
            raw_marker_text=raw.raw_marker_text,
            citation_token=unit.citation_token,
            source=chunk.source,
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            score=chunk.score,
            title=chunk.title,
            author=chunk.author,
            publication_date=chunk.publication_date,
            source_url=chunk.source_url,
            page=chunk.page,
        )
