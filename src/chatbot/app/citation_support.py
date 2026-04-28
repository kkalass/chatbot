"""Shared helpers for citation validation and citation-pass prompt construction."""

import json
import re
from collections.abc import Sequence
from typing import cast

from src.chatbot.app.protocols import ChatMessage, SourceChunk, ToolCallInfo

_SEARCH_TOOL_NAME = "search_documents"
_CITATION_TOOL_NAME = "cite_sources"


def collect_search_chunks(
    history: tuple[ChatMessage, ...],
    *,
    search_call_ids: set[str] | None = None,
) -> dict[tuple[str, str], SourceChunk]:
    """Build a map from ``(source, chunk_id)`` to retrieved chunks from history.

    If ``search_call_ids`` is provided, only tool results correlated to those
    call IDs are considered. Otherwise all ``search_documents`` calls in the
    given history are used.
    """

    call_ids: set[str]
    if search_call_ids is None:
        call_ids = set()
        for msg in history:
            if msg.role != "assistant" or not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                if tc.name == _SEARCH_TOOL_NAME:
                    call_ids.add(tc.call_id)
    else:
        call_ids = set(search_call_ids)

    if not call_ids:
        return {}

    chunks: dict[tuple[str, str], SourceChunk] = {}
    for msg in history:
        if msg.role != "tool":
            continue
        if msg.tool_call_id is None or msg.tool_call_id not in call_ids:
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


def render_search_results_for_prompt(
    chunks: Sequence[SourceChunk],
    *,
    max_chunks: int = 12,
    max_content_chars: int = 1200,
) -> str:
    """Render retrieved chunks for the dedicated citation-pass prompt."""

    if not chunks:
        return "(no search results available)"

    sorted_chunks = sorted(
        chunks,
        key=lambda c: (-c.score, c.source, c.chunk_id),
    )
    lines: list[str] = []
    for idx, chunk in enumerate(sorted_chunks[:max_chunks], start=1):
        content = _truncate_text(chunk.content.strip(), max_chars=max_content_chars)
        lines.extend(
            [
                f"- result_index: {idx}",
                f"  source: {chunk.source}",
                f"  chunk_id: {chunk.chunk_id}",
                f"  score: {chunk.score:.6f}",
                "  content: |",
                _indent_text(content, prefix="    "),
            ]
        )

    if len(sorted_chunks) > max_chunks:
        lines.append(f"- truncated_results: {len(sorted_chunks) - max_chunks}")

    return "\n".join(lines)


def parse_serialized_citation_tool_call(text: str) -> ToolCallInfo | None:
    """Parse a serialized ``cite_sources`` call from plain model text.

    Some smaller models output a JSON object in assistant text instead of using
    the native tool-call channel. This parser recovers a ``ToolCallInfo`` for
    ``cite_sources`` from that text when possible.
    """

    candidates = _candidate_json_payloads(text)
    for candidate in candidates:
        parsed = _parse_citation_tool_call_dict(candidate)
        if parsed is not None:
            return ToolCallInfo(
                name=_CITATION_TOOL_NAME,
                arguments=parsed,
                call_id=_CITATION_TOOL_NAME,
            )

    pairs = re.findall(
        r'"source"\s*:\s*"([^"]+)"\s*,\s*"(?:chunk_id|id)"\s*:\s*"([^"]+)"',
        text,
    )
    if not pairs:
        return None

    citations = [{"source": source, "chunk_id": chunk_id} for source, chunk_id in pairs]
    return ToolCallInfo(
        name=_CITATION_TOOL_NAME,
        arguments={"citations": citations},
        call_id=_CITATION_TOOL_NAME,
    )


def _json_to_source_chunk(data: dict[str, object]) -> SourceChunk:
    """Reconstruct a ``SourceChunk`` from serialized retrieval-tool payload."""

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
    )


def _truncate_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def _indent_text(text: str, *, prefix: str) -> str:
    parts = text.splitlines() or [""]
    return "\n".join(f"{prefix}{line}" for line in parts)


def _candidate_json_payloads(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    candidates: list[str] = [stripped]

    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", stripped, re.IGNORECASE)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    first_brace = stripped.find("{")
    if first_brace >= 0:
        candidates.append(stripped[first_brace:])

    return _dedupe_strings(candidates)


def _parse_citation_tool_call_dict(raw: str) -> dict[str, object] | None:
    for candidate in (raw, _balance_braces(raw)):
        data = _try_parse_json_object(candidate)
        if data is None:
            continue

        call_name, arguments = _extract_tool_call_parts(data)
        if call_name != _CITATION_TOOL_NAME or not isinstance(arguments, dict):
            continue

        normalized = _normalize_citation_arguments(cast(dict[str, object], arguments))
        if normalized is not None:
            return normalized

    return None


def _extract_tool_call_parts(payload: dict[str, object]) -> tuple[str | None, object | None]:
    name = payload.get("name")
    if isinstance(name, str):
        parameters = payload.get("parameters")
        arguments = payload.get("arguments")
        return name, parameters if parameters is not None else arguments

    function_obj = payload.get("function")
    if isinstance(function_obj, dict):
        function_dict = cast(dict[str, object], function_obj)
        fn_name = function_dict.get("name")
        if isinstance(fn_name, str):
            parameters = function_dict.get("parameters")
            arguments = function_dict.get("arguments")
            return fn_name, parameters if parameters is not None else arguments

    return None, None


def _normalize_citation_arguments(arguments: dict[str, object]) -> dict[str, object] | None:
    raw_citations: object = arguments.get("citations")
    if isinstance(raw_citations, str):
        parsed = _try_parse_json_object(raw_citations)
        if parsed is not None:
            maybe_citations = parsed.get("citations")
            if isinstance(maybe_citations, list):
                raw_citations = cast(list[object], maybe_citations)
        else:
            try:
                decoded: object = json.loads(raw_citations)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                raw_citations = cast(list[object], decoded)

    if not isinstance(raw_citations, list):
        return None

    citations: list[dict[str, str]] = []
    for item in cast(list[object], raw_citations):
        if not isinstance(item, dict):
            continue
        item_dict = cast(dict[str, object], item)
        source = item_dict.get("source")
        chunk_id: object | None = item_dict.get("chunk_id")
        if chunk_id is None:
            chunk_id = item_dict.get("id")
        if not isinstance(source, str) or not isinstance(chunk_id, str):
            continue
        citations.append({"source": source, "chunk_id": chunk_id})

    return {"citations": citations}


def _balance_braces(raw: str) -> str:
    open_count = raw.count("{")
    close_count = raw.count("}")
    if close_count >= open_count:
        return raw
    return f"{raw}{'}' * (open_count - close_count)}"


def _try_parse_json_object(raw: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return cast(dict[str, object], parsed) if isinstance(parsed, dict) else None


def _dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
