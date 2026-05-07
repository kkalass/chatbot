# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the Chainlit citation rendering helpers."""

from src.chatbot.contracts.citation import DocumentCitation, NumberedCitation, ToolCitation
from src.chatbot.contracts.i18n import I18nMessage
from src.chatbot.ui.citation_view import (
    build_citation_content,
    build_citation_markdown,
    build_citation_name,
)
from src.chatbot.ui.i18n_messages import resolve_message


def _translate_en(msg: I18nMessage) -> str:
    return resolve_message(msg, lang="en")


def _doc(
    *,
    title: str | None = None,
    author: str | None = None,
    publication_date: str | None = None,
    page: str | None = None,
    source: str = "docs/a.md",
    source_url: str | None = None,
    content: str = "an excerpt",
) -> DocumentCitation:
    return DocumentCitation(
        raw_marker_text="<m>",
        citation_token="c1",
        source=source,
        chunk_id="c1",
        content=content,
        score=0.9,
        title=title,
        author=author,
        publication_date=publication_date,
        source_url=source_url,
        page=page,
    )


class TestBuildCitationName:
    def test_uses_title_when_present(self) -> None:
        citation = NumberedCitation(reference_number=3, citation=_doc(title="My Doc"))
        assert build_citation_name(citation, translate=_translate_en) == "[3] My Doc"

    def test_appends_page_when_present(self) -> None:
        citation = NumberedCitation(reference_number=3, citation=_doc(title="T", page="42"))
        assert build_citation_name(citation, translate=_translate_en) == "[3] T (p. 42)"

    def test_falls_back_to_author_date_then_source(self) -> None:
        citation = NumberedCitation(
            reference_number=3,
            citation=_doc(author="A", publication_date="2024"),
        )
        assert build_citation_name(citation, translate=_translate_en) == "[3] A - 2024"

        citation = NumberedCitation(reference_number=4, citation=_doc(source="x.md"))
        assert build_citation_name(citation, translate=_translate_en) == "[4] x.md"

    def test_tool_citation_falls_back_to_tool_name(self) -> None:
        cit = ToolCitation(
            raw_marker_text="<m>",
            citation_token="tok-vac-1",
            tool_name="get_vacation_days",
            result={"days": 30},
        )
        numbered = NumberedCitation(reference_number=2, citation=cit)
        assert build_citation_name(numbered, translate=_translate_en) == "[2] get_vacation_days"

    def test_tool_citation_uses_display_name(self) -> None:
        cit = ToolCitation(
            raw_marker_text="<m>",
            citation_token="tok-vac-1",
            tool_name="get_vacation_days",
            result={"days": 30},
            display_name=I18nMessage(key="vacation_days.display_name", args={}),
        )
        numbered = NumberedCitation(reference_number=2, citation=cit)
        assert build_citation_name(numbered, translate=_translate_en) == "[2] Vacation Days Service"


class TestBuildCitationContent:
    def test_document_includes_excerpt(self) -> None:
        rendered = build_citation_content(
            NumberedCitation(reference_number=3, citation=_doc(title="T", content="hello world")),
            translate=_translate_en,
        )
        assert "### 3." in rendered
        assert " T" in rendered
        assert "hello world" in rendered
        assert "**Excerpt**" in rendered

    def test_document_with_url_renders_markdown_link(self) -> None:
        rendered = build_citation_content(
            NumberedCitation(
                reference_number=3,
                citation=_doc(title="T", source_url="https://e.com/d"),
            ),
            translate=_translate_en,
        )
        assert "### 3." in rendered
        assert "[T](https://e.com/d)" in rendered

    def test_tool_citation_renders_property_list(self) -> None:
        cit = ToolCitation(
            raw_marker_text="<m>",
            citation_token="tok-vac-2",
            tool_name="get_vacation_days",
            result={"total_days": 30, "remaining_days": 20},
        )
        rendered = build_citation_content(
            NumberedCitation(reference_number=2, citation=cit), translate=_translate_en
        )
        assert "### 2." in rendered
        assert "get_vacation_days" in rendered
        assert "**total_days:** 30" in rendered
        assert "**remaining_days:** 20" in rendered


class TestBuildCitationMarkdown:
    def test_empty_returns_empty_string(self) -> None:
        assert build_citation_markdown([], translate=_translate_en) == ""

    def test_orders_by_reference_number(self) -> None:
        a = NumberedCitation(reference_number=2, citation=_doc(title="A"))
        b = NumberedCitation(reference_number=1, citation=_doc(title="B"))

        rendered = build_citation_markdown([a, b], translate=_translate_en)

        idx_a = rendered.find("A")
        idx_b = rendered.find("B")
        assert 0 < idx_b < idx_a
        assert rendered.startswith("---\n**Sources**")

    def test_tool_citation_appendix_uses_tool_name(self) -> None:
        cit = ToolCitation(
            raw_marker_text="<m>",
            citation_token="tok-vac-3",
            tool_name="get_vacation_days",
            result={"days": 30},
        )
        rendered = build_citation_markdown(
            [NumberedCitation(reference_number=1, citation=cit)], translate=_translate_en
        )
        assert "1. get_vacation_days" in rendered


def _doc_with_image(*, ref: int, image_path: str | None) -> NumberedCitation:
    return NumberedCitation(
        reference_number=ref,
        citation=DocumentCitation(
            raw_marker_text="<m>",
            citation_token=f"c{ref}",
            source="corpus/x.pdf",
            chunk_id=f"c{ref}",
            content="excerpt",
            score=0.9,
            kind="image_description" if image_path else "text",
            image_path=image_path,
        ),
    )


class TestDocumentContentWithFigure:
    def test_figure_label_appears_when_image_path_set(self) -> None:
        nc = _doc_with_image(ref=4, image_path="/tmp/x.png")
        rendered = build_citation_content(nc, translate=_translate_en)

        assert "**Figure:**" in rendered
        assert "Figure [4]" in rendered

    def test_no_figure_label_for_text_only_citation(self) -> None:
        nc = _doc_with_image(ref=4, image_path=None)
        rendered = build_citation_content(nc, translate=_translate_en)

        assert "**Figure:**" not in rendered
