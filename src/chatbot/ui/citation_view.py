"""Citation rendering helpers for the Chainlit UI.

Formats citations for two presentation surfaces:
- individual side-panel elements for inspected chunks
- a compact, deduplicated source list appended to the answer bubble
"""

import json
from collections.abc import Sequence
from textwrap import dedent

from src.chatbot.app.protocols import SourceChunk, ToolCitationEvent


def build_citation_name(chunk: SourceChunk) -> str:
    """Return a compact element label without exposing opaque chunk ids."""
    name = _display_title(chunk)
    if chunk.page:
        return f"{name} (p. {chunk.page})"
    return name


def build_citation_content(chunk: SourceChunk) -> str:
    """Return structured Markdown for a citation side-panel element."""
    header = _link_or_text(_display_title(chunk), chunk.source_url)
    lines = [f"### {header}", ""]

    if chunk.author:
        lines.append(f"**Author:** {chunk.author}  ")
    if chunk.publication_date:
        lines.append(f"**Date:** {chunk.publication_date}  ")
    if chunk.page:
        lines.append(f"**Page:** {chunk.page}  ")
    if not chunk.author and not chunk.title:
        lines.append(f"**Source:** {chunk.source}  ")

    lines.extend(["", "**Excerpt**", "", _normalize_excerpt(chunk.content)])
    return "\n".join(lines)


def build_citation_markdown(chunks: Sequence[SourceChunk]) -> str:
    """Return a compact deduplicated Markdown source list for the answer bubble."""
    unique_chunks = _deduplicate_sources(chunks)
    if not unique_chunks:
        return ""

    lines = ["---", "**Sources**", ""]
    for index, chunk in enumerate(unique_chunks, start=1):
        lines.append(f"{index}. {_build_source_list_item(chunk)}")
    return "\n".join(lines)


def build_tool_citation_name(tool_citation: ToolCitationEvent) -> str:
    """Return a compact label for a non-retrieval tool citation."""
    return f"Tool: {tool_citation.tool_name}"


def build_tool_citation_content(tool_citation: ToolCitationEvent) -> str:
    """Return structured Markdown for a tool citation side-panel element.

    Top-level keys of the result are rendered as a property list; each value is
    formatted as inline JSON so nested objects stay readable without extra nesting.
    Only called for successful results (no ``error`` key).
    """
    lines = [f"### {tool_citation.tool_name}", ""]
    for key, value in tool_citation.result.items():
        if isinstance(value, (dict, list)):
            formatted = f"`{json.dumps(value, ensure_ascii=False)}`"
        else:
            formatted = str(value)
        lines.append(f"**{key}:** {formatted}  ")
    return "\n".join(lines)


def build_tool_citation_markdown(tool_citations: Sequence[ToolCitationEvent]) -> str:
    """Return a compact deduplicated Markdown list for tool-backed claims."""
    unique_tool_citations = _deduplicate_tool_citations(tool_citations)
    if not unique_tool_citations:
        return ""

    lines = ["---", "**Tools**", ""]
    for index, tool_citation in enumerate(unique_tool_citations, start=1):
        lines.append(f"{index}. {tool_citation.tool_name}")
    return "\n".join(lines)


def _display_title(chunk: SourceChunk) -> str:
    if chunk.title:
        return chunk.title

    metadata_parts = [part for part in (chunk.author, chunk.publication_date) if part]
    if metadata_parts:
        return " - ".join(metadata_parts)

    return chunk.source


def _normalize_excerpt(content: str) -> str:
    normalized = dedent(content).strip()
    return normalized or content.strip()


def _build_source_list_item(chunk: SourceChunk) -> str:
    title = chunk.title
    parts: list[str] = []

    if title:
        parts.append(_link_or_text(title, chunk.source_url))
    elif chunk.author:
        author_label = _link_or_text(chunk.author, chunk.source_url)
        parts.append(author_label)
    else:
        parts.append(_link_or_text(chunk.source, chunk.source_url))

    if chunk.author and chunk.author != title:
        parts.append(chunk.author)
    if chunk.publication_date:
        parts.append(chunk.publication_date)
    if chunk.page:
        parts.append(f"p. {chunk.page}")
    if not chunk.author and not chunk.title and chunk.source not in parts:
        parts.append(chunk.source)

    return " - ".join(parts)


def _link_or_text(label: str, source_url: str | None) -> str:
    if source_url:
        return f"[{label}]({source_url})"
    return label


def _deduplicate_sources(chunks: Sequence[SourceChunk]) -> list[SourceChunk]:
    seen: set[tuple[str | None, str | None, str | None, str | None, str | None, str]] = set()
    deduplicated: list[SourceChunk] = []
    for chunk in chunks:
        key = (
            chunk.title,
            chunk.author,
            chunk.publication_date,
            chunk.source_url,
            chunk.page,
            chunk.source,
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(chunk)
    return deduplicated


def _deduplicate_tool_citations(
    tool_citations: Sequence[ToolCitationEvent],
) -> list[ToolCitationEvent]:
    seen: set[str] = set()
    deduplicated: list[ToolCitationEvent] = []
    for tool_citation in tool_citations:
        key = tool_citation.tool_call_id
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(tool_citation)
    return deduplicated
