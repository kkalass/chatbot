# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Citation rendering helpers shared between the Chainlit UI and offline evaluation.

Renders both :class:`DocumentCitation` (retrieved chunks) and
:class:`ToolCitation` (non-document tool results) for two presentation surfaces:
- individual side-panel elements for inspected citations
- a compact, deduplicated source list appended to the answer bubble

The stream-assembly helpers (:func:`format_text_chunk`,
:func:`format_citation_marker`) have no Chainlit dependency and are used by
both ``src/chatbot/ui/app.py`` (streaming) and ``eval/run_experiment.py``
(offline evaluation) to produce identical output.
"""

import json
from collections.abc import Callable, Sequence
from enum import StrEnum
from textwrap import dedent

from src.chatbot.contracts.citation import (
    Citation,
    DocumentCitation,
    NumberedCitation,
    ToolCitation,
)
from src.chatbot.contracts.i18n import I18nMessage


class CitationViewKey(StrEnum):
    """Message keys for citation rendering — used by both the UI and offline evaluation."""

    PANEL_TITLE = "citation_view.panel_title"
    PAGE_ABBREVIATION = "citation_view.page_abbreviation"
    AUTHOR_LABEL = "citation_view.author_label"
    DATE_LABEL = "citation_view.date_label"
    PAGE_LABEL = "citation_view.page_label"
    SOURCE_LABEL = "citation_view.source_label"
    EXCERPT_LABEL = "citation_view.excerpt_label"
    FIGURE_LABEL = "citation_view.figure_label"
    FIGURE_NAME = "citation_view.figure_name"


# -- Display labels ------------------------------------------------------


def build_citation_name(
    numbered: NumberedCitation,
    *,
    translate: Callable[[I18nMessage], str],
) -> str:
    """Compact label for a citation, prefixed with the session-stable reference number."""
    citation = numbered.citation
    match citation:
        case DocumentCitation():
            name = _document_display_title(citation)
            if citation.page:
                page = translate(
                    I18nMessage(key=CitationViewKey.PAGE_ABBREVIATION, args={"page": citation.page})
                )
                base = f"{name} ({page})"
            else:
                base = name
        case ToolCitation():
            base = _tool_display_label(citation, translate=translate)
    return f"[{numbered.reference_number}] {base}"


def build_side_panel_label(*, translate: Callable[[I18nMessage], str]) -> str:
    """Localised title for the citations side panel."""
    return translate(I18nMessage(key=CitationViewKey.PANEL_TITLE, args={}))


def build_citation_content(
    numbered: NumberedCitation,
    *,
    translate: Callable[[I18nMessage], str],
    image_markdown_src: str | None = None,
) -> str:
    """Structured Markdown for a citation entry in the side panel.

    The section heading is prefixed with the session-stable reference number
    so the entry is identifiable in the aggregated panel view.

    When *image_markdown_src* is provided and the citation carries an
    ``image_path``, the figure is embedded directly as a markdown image so
    callers do not need a separate ``cl.Image`` element.
    """
    citation = numbered.citation
    ref = numbered.reference_number
    match citation:
        case DocumentCitation():
            return _build_document_content(
                citation,
                ref=ref,
                translate=translate,
                image_markdown_src=image_markdown_src,
            )
        case ToolCitation():
            return _build_tool_content(citation, ref=ref, translate=translate)


# -- Stream assembly helpers (used by UI and eval) -----------------------


def format_text_chunk(
    chunk: str,
    pending_whitespace: str,
) -> tuple[list[str], str]:
    """Return stream tokens for a text chunk while buffering trailing whitespace.

    Trailing whitespace is held back so a following ``[N]`` reference can be
    rendered directly after the preceding sentence without an inserted newline.
    """
    stripped = chunk.rstrip(" \t\r\n")
    if not stripped:
        return [], f"{pending_whitespace}{chunk}"

    tokens: list[str] = []
    if pending_whitespace:
        tokens.append(pending_whitespace)

    trailing_whitespace = chunk[len(stripped) :]
    tokens.append(stripped)

    return tokens, trailing_whitespace


def format_citation_marker(
    nc: NumberedCitation,
    pending_whitespace: str,
) -> tuple[list[str], str]:
    """Return a ``[n]`` token while keeping trailing whitespace buffered.

    This avoids inserting blank lines between consecutive citation references
    when the model emits multiple marker blocks separated by newlines.
    """
    return [f"_({nc.reference_number})_"], pending_whitespace


# -- Bubble appendix -----------------------------------------------------


def build_citation_markdown(
    numbered: Sequence[NumberedCitation],
    *,
    translate: Callable[[I18nMessage], str],
) -> str:
    """Markdown source list for the answer bubble.

    Items are ordered by their assigned reference number, so the answer's
    inline ``[N]`` markers map directly to entries in the appendix.
    """
    if not numbered:
        return ""

    heading = translate(I18nMessage(key=CitationViewKey.PANEL_TITLE, args={}))
    sorted_items = sorted(numbered, key=lambda nc: nc.reference_number)
    lines = ["---", f"**{heading}**", ""]
    for item in sorted_items:
        lines.append(
            f"{item.reference_number}. {_build_appendix_item(item.citation, translate=translate)}"
        )
    return "\n".join(lines)


# -- Internal: DocumentCitation -----------------------------------------


def _document_display_title(citation: DocumentCitation) -> str:
    if citation.title:
        return citation.title
    metadata_parts = [part for part in (citation.author, citation.publication_date) if part]
    if metadata_parts:
        return " - ".join(metadata_parts)
    return citation.source


def _build_document_content(
    citation: DocumentCitation,
    *,
    ref: int,
    translate: Callable[[I18nMessage], str],
    image_markdown_src: str | None = None,
) -> str:
    header = _link_or_text(_document_display_title(citation), citation.source_url)
    lines = [f"### {ref}. {header}", ""]
    if citation.author:
        lines.append(
            f"**{translate(I18nMessage(key=CitationViewKey.AUTHOR_LABEL, args={}))}** {citation.author}  "
        )
    if citation.publication_date:
        lines.append(
            f"**{translate(I18nMessage(key=CitationViewKey.DATE_LABEL, args={}))}** {citation.publication_date}  "
        )
    if citation.page:
        lines.append(
            f"**{translate(I18nMessage(key=CitationViewKey.PAGE_LABEL, args={}))}** {citation.page}  "
        )
    if not citation.author and not citation.title:
        lines.append(
            f"**{translate(I18nMessage(key=CitationViewKey.SOURCE_LABEL, args={}))}** {citation.source}  "
        )
    if citation.image_path:
        figure_label = translate(I18nMessage(key=CitationViewKey.FIGURE_LABEL, args={}))
        figure_name = translate(
            I18nMessage(key=CitationViewKey.FIGURE_NAME, args={"ref": str(ref)})
        )
        if image_markdown_src:
            lines.extend(["", f"![{figure_name}]({image_markdown_src})", ""])
        else:
            lines.append(f"**{figure_label}** {figure_name}  ")
    excerpt_label = translate(I18nMessage(key=CitationViewKey.EXCERPT_LABEL, args={}))
    lines.extend(["", f"**{excerpt_label}**", "", _normalize_excerpt(citation.content)])
    return "\n".join(lines)


def _build_document_appendix_item(
    citation: DocumentCitation, *, translate: Callable[[I18nMessage], str]
) -> str:
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
        parts.append(
            translate(
                I18nMessage(key=CitationViewKey.PAGE_ABBREVIATION, args={"page": citation.page})
            )
        )
    if not citation.author and not citation.title and citation.source not in parts:
        parts.append(citation.source)

    return " - ".join(parts)


# -- Internal: ToolCitation ---------------------------------------------


def _build_tool_content(
    citation: ToolCitation, *, ref: int, translate: Callable[[I18nMessage], str]
) -> str:
    """Render a tool result as a property list; only called for successful results."""
    heading = _tool_display_label(citation, translate=translate)
    lines = [f"### {ref}. {heading}", ""]
    for key, value in citation.result.items():
        if isinstance(value, (dict, list)):
            formatted = f"`{json.dumps(value, ensure_ascii=False)}`"
        else:
            formatted = str(value)
        lines.append(f"**{key}:** {formatted}  ")
    return "\n".join(lines)


# -- Shared --------------------------------------------------------------


def _build_appendix_item(citation: Citation, *, translate: Callable[[I18nMessage], str]) -> str:
    match citation:
        case DocumentCitation():
            return _build_document_appendix_item(citation, translate=translate)
        case ToolCitation():
            return _tool_display_label(citation, translate=translate)


def _tool_display_label(citation: ToolCitation, *, translate: Callable[[I18nMessage], str]) -> str:
    """Resolve display name from I18nMessage if present, fall back to tool_name."""
    if citation.display_name is not None:
        return translate(citation.display_name)
    return citation.tool_name


def _normalize_excerpt(content: str) -> str:
    normalized = dedent(content).strip()
    return normalized or content.strip()


def _link_or_text(label: str, source_url: str | None) -> str:
    if source_url:
        return f"[{label}]({source_url})"
    return label
