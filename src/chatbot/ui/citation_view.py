# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Citation rendering helpers for the Chainlit UI.

Renders both :class:`DocumentCitation` (retrieved chunks) and
:class:`ToolCitation` (non-document tool results) for two presentation surfaces:
- individual side-panel elements for inspected citations
- a compact, deduplicated source list appended to the answer bubble
"""

import json
from collections.abc import Sequence
from textwrap import dedent

from src.chatbot.app.protocols import (
    Citation,
    DocumentCitation,
    NumberedCitation,
    ToolCitation,
)
from src.chatbot.ui.i18n_messages import resolve_message

# -- Display labels ------------------------------------------------------


def build_citation_name(numbered: NumberedCitation, *, lang: str = "en") -> str:
    """Compact side-panel label."""
    citation = numbered.citation
    match citation:
        case DocumentCitation():
            name = _document_display_title(citation)
            if citation.page:
                return f"{name} (p. {citation.page})"
            return name
        case ToolCitation():
            return _tool_display_label(citation, lang=lang)


def build_citation_content(numbered: NumberedCitation, *, lang: str = "en") -> str:
    """Structured Markdown for a side-panel element."""
    citation = numbered.citation
    match citation:
        case DocumentCitation():
            return _build_document_content(citation)
        case ToolCitation():
            return _build_tool_content(citation, lang=lang)


# -- Bubble appendix -----------------------------------------------------


def build_citation_markdown(numbered: Sequence[NumberedCitation], *, lang: str = "en") -> str:
    """Markdown source list for the answer bubble.

    Items are ordered by their assigned reference number, so the answer's
    inline ``[N]`` markers map directly to entries in the appendix.
    """
    if not numbered:
        return ""

    sorted_items = sorted(numbered, key=lambda nc: nc.reference_number)
    lines = ["---", "**Sources**", ""]
    for item in sorted_items:
        lines.append(f"{item.reference_number}. {_build_appendix_item(item.citation, lang=lang)}")
    return "\n".join(lines)


# -- Internal: DocumentCitation -----------------------------------------


def _document_display_title(citation: DocumentCitation) -> str:
    if citation.title:
        return citation.title
    metadata_parts = [part for part in (citation.author, citation.publication_date) if part]
    if metadata_parts:
        return " - ".join(metadata_parts)
    return citation.source


def _build_document_content(citation: DocumentCitation) -> str:
    header = _link_or_text(_document_display_title(citation), citation.source_url)
    lines = [f"### {header}", ""]
    if citation.author:
        lines.append(f"**Author:** {citation.author}  ")
    if citation.publication_date:
        lines.append(f"**Date:** {citation.publication_date}  ")
    if citation.page:
        lines.append(f"**Page:** {citation.page}  ")
    if not citation.author and not citation.title:
        lines.append(f"**Source:** {citation.source}  ")
    lines.extend(["", "**Excerpt**", "", _normalize_excerpt(citation.content)])
    return "\n".join(lines)


def _build_document_appendix_item(citation: DocumentCitation) -> str:
    title = citation.title
    parts: list[str] = []

    if title:
        parts.append(_link_or_text(title, citation.source_url))
    elif citation.author:
        parts.append(_link_or_text(citation.author, citation.source_url))
    else:
        parts.append(_link_or_text(citation.source, citation.source_url))

    if citation.author and citation.author != title:
        parts.append(citation.author)
    if citation.publication_date:
        parts.append(citation.publication_date)
    if citation.page:
        parts.append(f"p. {citation.page}")
    if not citation.author and not citation.title and citation.source not in parts:
        parts.append(citation.source)

    return " - ".join(parts)


# -- Internal: ToolCitation ---------------------------------------------


def _build_tool_content(citation: ToolCitation, *, lang: str = "en") -> str:
    """Render a tool result as a property list; only called for successful results."""
    heading = _tool_display_label(citation, lang=lang)
    lines = [f"### {heading}", ""]
    for key, value in citation.result.items():
        if isinstance(value, (dict, list)):
            formatted = f"`{json.dumps(value, ensure_ascii=False)}`"
        else:
            formatted = str(value)
        lines.append(f"**{key}:** {formatted}  ")
    return "\n".join(lines)


# -- Shared --------------------------------------------------------------


def _build_appendix_item(citation: Citation, *, lang: str = "en") -> str:
    match citation:
        case DocumentCitation():
            return _build_document_appendix_item(citation)
        case ToolCitation():
            return _tool_display_label(citation, lang=lang)


def _tool_display_label(citation: ToolCitation, *, lang: str = "en") -> str:
    """Resolve display name from I18nMessage if present, fall back to tool_name."""
    if citation.display_name is not None:
        return resolve_message(citation.display_name, lang=lang)
    return citation.tool_name


def _normalize_excerpt(content: str) -> str:
    normalized = dedent(content).strip()
    return normalized or content.strip()


def _link_or_text(label: str, source_url: str | None) -> str:
    if source_url:
        return f"[{label}]({source_url})"
    return label
