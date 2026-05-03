"""LLM-callable, citeable retrieval tool.

Modelling retrieval as a tool lets the LLM decide *when* to search and *how*
to formulate the query, enabling multi-hop retrieval and query reformulation
based on full conversation context — capabilities not possible with eager
pre-retrieval.

This tool implements :class:`~src.chatbot.app.citation.CiteableTool`: it owns
the LLM-side rendering of search results, contributes its own citation prompt
fragment, and validates :class:`DocumentRawCitation` payloads emitted by the
model against its own past tool outputs.
"""

import html
import re
from collections.abc import Sequence
from typing import cast

import structlog
from openinference.semconv.trace import OpenInferenceMimeTypeValues, OpenInferenceSpanKindValues
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from pydantic import ValidationError

from src.chatbot.app.citation import (
    QUOTE_END_MARKER,
    QUOTE_START_MARKER,
    CitationContext,
    CiteInstructions,
    DocumentCitation,
    DocumentRawCitation,
    RawCitation,
)
from src.chatbot.app.citation.models import Citation
from src.chatbot.app.protocols import (
    JsonObject,
    Retriever,
    SourceChunk,
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


_CITE_FRAGMENT = f"""### {_TOOL_NAME}

When a sentence is grounded in a chunk returned by `{_TOOL_NAME}`, emit
immediately after that sentence a marker block of the form:

  {QUOTE_START_MARKER}{{"kind":"document","tool_call_id":"<id>","source":"<source>","chunk_id":"<chunk_id>"}}{QUOTE_END_MARKER}

Required fields (all strings, all copied **verbatim** from the conversation
context — no re-formatting):
  - kind: must be the literal string "document"
  - tool_call_id: the `tool_call_id` attribute of the assistant's tool call
    that produced the supporting `{_TOOL_NAME}` result
  - source: the `source_path` attribute of the enclosing `<source>` XML
    element from that tool result
  - chunk_id: the `chunk_id` attribute of the specific `<chunk>` element
    whose content supports the sentence

Optional fields (omit when unsure):
  - quote_text: a literal verbatim span from the cited chunk (≤ 200 chars)
  - claim: the supported claim restated in your own words

Do not invent IDs. If the exact `tool_call_id`, `source`, or `chunk_id`
is not visible in the conversation context, do not emit a marker for that
sentence."""


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


def _normalize_source_path(source: str) -> str:
    normalized = source.strip().replace("\\", "/")
    normalized = normalized.removeprefix("./")
    normalized = normalized.lstrip("/")
    return normalized


def _format_chunks_as_xml(content: JsonObject) -> str:
    """Render search-result chunks as structured XML for model consumption.

    Placing ``chunk_id`` immediately adjacent to its content makes it
    significantly easier for smaller models to correctly attribute claims to
    the right chunk rather than picking the first chunk in the list.
    """
    chunks_raw: object = content.get("chunks")
    if not isinstance(chunks_raw, list) or not chunks_raw:
        message: object = content.get("message", "No relevant documents found.")
        return str(message)

    grouped_by_source: dict[str, list[dict[str, object]]] = {}
    for raw_chunk in cast(list[object], chunks_raw):
        if not isinstance(raw_chunk, dict):
            continue
        chunk_data = cast(dict[str, object], raw_chunk)
        source = str(chunk_data.get("source", "unknown"))
        grouped_by_source.setdefault(source, []).append(chunk_data)

    parts: list[str] = ["<search_results>\n"]
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
            chunk_id = str(chunk_data.get("chunk_id", ""))
            page: object = chunk_data.get("page")
            content_text = str(chunk_data.get("content", ""))
            chunk_id_attr = html.escape(chunk_id, quote=True)
            page_attr = html.escape(str(page), quote=True) if page is not None else "unknown"
            content_escaped = html.escape(content_text)

            parts.append(f'    <chunk chunk_id="{chunk_id_attr}" page="{page_attr}">\n')
            parts.append(f"      {content_escaped}\n")
            parts.append("    </chunk>\n")

        parts.append("  </source>\n")

    parts.append("</search_results>")
    return "".join(parts)


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


def _collect_chunks(
    tool_results: Sequence[JsonObject],
) -> dict[tuple[str, str], SourceChunk]:
    """Build ``(source, chunk_id) -> SourceChunk`` from prior search results,
    keeping the highest-scoring chunk on duplicates."""
    chunks: dict[tuple[str, str], SourceChunk] = {}
    for result in tool_results:
        raw_chunks: object = result.get("chunks")
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


def _resolve_document_chunk(
    raw: DocumentRawCitation,
    chunks: dict[tuple[str, str], SourceChunk],
) -> SourceChunk | None:
    exact = chunks.get((raw.source, raw.chunk_id))
    if exact is not None:
        return exact

    normalized_source = _normalize_source_path(raw.source)
    for (source, chunk_id), chunk in chunks.items():
        if chunk_id != raw.chunk_id:
            continue
        if _normalize_source_path(source) == normalized_source:
            return chunk

    # Last-resort fallback: chunk IDs are content hashes in this project and
    # therefore globally unique across sources. Accept only a unique match.
    candidates = [
        chunk for (_source, chunk_id), chunk in chunks.items() if chunk_id == raw.chunk_id
    ]
    if len(candidates) == 1:
        return candidates[0]

    return None


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

    def cite_instructions(self) -> CiteInstructions:
        return CiteInstructions(prompt_fragment=_CITE_FRAGMENT)

    def format_for_history(self, result: JsonObject) -> str:
        return _format_chunks_as_xml(result)

    def validate_and_enrich(
        self,
        raw: RawCitation,
        context: CitationContext,
    ) -> Citation | None:
        if not isinstance(raw, DocumentRawCitation):
            return None
        chunks = _collect_chunks(context.tool_results_for(_TOOL_NAME))
        chunk = _resolve_document_chunk(raw, chunks)
        if chunk is None:
            return None
        return DocumentCitation(
            raw_marker_text=raw.raw_marker_text,
            tool_call_id=raw.tool_call_id,
            source=chunk.source,
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            score=chunk.score,
            title=chunk.title,
            author=chunk.author,
            publication_date=chunk.publication_date,
            source_url=chunk.source_url,
            page=chunk.page,
            quote_text=raw.quote_text,
        )
