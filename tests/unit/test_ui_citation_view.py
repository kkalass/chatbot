"""Tests for the Chainlit citation rendering helpers."""

from src.chatbot.app.citation import (
    DocumentCitation,
    NumberedCitation,
    ToolCitation,
)
from src.chatbot.ui.citation_view import (
    build_citation_content,
    build_citation_markdown,
    build_citation_name,
)


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
        tool_call_id="tc1",
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
        assert build_citation_name(_doc(title="My Doc")) == "My Doc"

    def test_appends_page_when_present(self) -> None:
        assert build_citation_name(_doc(title="T", page="42")) == "T (p. 42)"

    def test_falls_back_to_author_date_then_source(self) -> None:
        assert build_citation_name(_doc(author="A", publication_date="2024")) == "A - 2024"
        assert build_citation_name(_doc(source="x.md")) == "x.md"

    def test_tool_citation_uses_tool_name(self) -> None:
        cit = ToolCitation(
            raw_marker_text="<m>",
            tool_call_id="tc1",
            tool_name="get_vacation_days",
            result={"days": 30},
        )
        assert build_citation_name(cit) == "Tool: get_vacation_days"


class TestBuildCitationContent:
    def test_document_includes_excerpt(self) -> None:
        rendered = build_citation_content(_doc(title="T", content="hello world"))
        assert "### T" in rendered
        assert "hello world" in rendered
        assert "**Excerpt**" in rendered

    def test_document_with_url_renders_markdown_link(self) -> None:
        rendered = build_citation_content(_doc(title="T", source_url="https://e.com/d"))
        assert "[T](https://e.com/d)" in rendered

    def test_tool_citation_renders_property_list(self) -> None:
        cit = ToolCitation(
            raw_marker_text="<m>",
            tool_call_id="tc1",
            tool_name="get_vacation_days",
            result={"total_days": 30, "remaining_days": 20},
        )
        rendered = build_citation_content(cit)
        assert "### get_vacation_days" in rendered
        assert "**total_days:** 30" in rendered
        assert "**remaining_days:** 20" in rendered


class TestBuildCitationMarkdown:
    def test_empty_returns_empty_string(self) -> None:
        assert build_citation_markdown([]) == ""

    def test_orders_by_reference_number(self) -> None:
        a = NumberedCitation(reference_number=2, citation=_doc(title="A"))
        b = NumberedCitation(reference_number=1, citation=_doc(title="B"))

        rendered = build_citation_markdown([a, b])

        idx_a = rendered.find("A")
        idx_b = rendered.find("B")
        assert 0 < idx_b < idx_a
        assert rendered.startswith("---\n**Sources**")

    def test_tool_citation_appendix_uses_tool_name(self) -> None:
        cit = ToolCitation(
            raw_marker_text="<m>",
            tool_call_id="tc1",
            tool_name="get_vacation_days",
            result={"days": 30},
        )
        rendered = build_citation_markdown([NumberedCitation(reference_number=1, citation=cit)])
        assert "1. get_vacation_days" in rendered
