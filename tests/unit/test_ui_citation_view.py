"""Unit tests for UI citation rendering helpers."""

from src.chatbot.app.protocols import SourceChunk
from src.chatbot.ui.citation_view import (
    build_citation_content,
    build_citation_markdown,
    build_citation_name,
)


class TestCitationView:
    def test_build_citation_name_prefers_title(self) -> None:
        chunk = SourceChunk(
            content="c",
            source="doc.txt",
            score=0.9,
            chunk_id="42",
            title="Nice Title",
        )

        assert build_citation_name(chunk) == "Nice Title"

    def test_build_citation_name_falls_back_to_source(self) -> None:
        chunk = SourceChunk(
            content="c",
            source="doc.txt",
            score=0.9,
            chunk_id="42",
        )

        assert build_citation_name(chunk) == "doc.txt"

    def test_build_citation_content_includes_optional_metadata_when_present(self) -> None:
        chunk = SourceChunk(
            content="Chunk body",
            source="doc.txt",
            score=0.9,
            chunk_id="42",
            author="Alice",
            publication_date="2024-10-01",
            source_url="https://example.com/doc",
        )

        content = build_citation_content(chunk)

        assert "### doc.txt" not in content
        assert "### Nice Title" not in content
        assert "### [Alice - 2024-10-01](https://example.com/doc)" in content
        assert "**Author:** Alice" in content
        assert "**Date:** 2024-10-01" in content
        assert "[Open source](https://example.com/doc)" not in content
        assert "**Source:** doc.txt" not in content
        assert content.endswith("Chunk body")

    def test_build_citation_content_omits_missing_optional_metadata(self) -> None:
        chunk = SourceChunk(
            content="Chunk body",
            source="doc.txt",
            score=0.9,
            chunk_id="42",
        )

        content = build_citation_content(chunk)

        assert "**Author:**" not in content
        assert "**Date:**" not in content
        assert "[Open source](" not in content
        assert "**Source:** doc.txt" in content

    def test_build_citation_content_uses_title_as_header_and_dedents_excerpt(self) -> None:
        chunk = SourceChunk(
            content="    Line one\n        Line two",
            source="doc.txt",
            score=0.9,
            chunk_id="42",
            title="Nice Title",
        )

        content = build_citation_content(chunk)

        assert content.startswith("### Nice Title")
        assert "**Source:** doc.txt" not in content
        assert "\n    Line one" not in content
        assert content.endswith("Line one\n    Line two")

    def test_build_citation_content_links_title_when_source_url_present(self) -> None:
        chunk = SourceChunk(
            content="Chunk body",
            source="doc.txt",
            score=0.9,
            chunk_id="42",
            title="Executive Order 14110",
            source_url="https://example.com/eo-14110",
        )

        content = build_citation_content(chunk)

        assert content.startswith("### [Executive Order 14110](https://example.com/eo-14110)")
        assert "[Open source](" not in content

    def test_build_citation_markdown_deduplicates_sources_and_links_url(self) -> None:
        first = SourceChunk(
            content="Chunk A",
            source="doc.txt",
            score=0.9,
            chunk_id="42",
            title="Nice Title",
            author="Alice",
            publication_date="2024-10-01",
            source_url="https://example.com/doc",
        )
        duplicate_same_doc = SourceChunk(
            content="Chunk B",
            source="doc.txt",
            score=0.8,
            chunk_id="43",
            title="Nice Title",
            author="Alice",
            publication_date="2024-10-01",
            source_url="https://example.com/doc",
        )
        fallback = SourceChunk(
            content="Chunk C",
            source="fallback.txt",
            score=0.7,
            chunk_id="44",
        )

        markdown = build_citation_markdown([first, duplicate_same_doc, fallback])

        assert markdown.count("1. ") == 1
        assert markdown.count("2. ") == 1
        assert "[Nice Title](https://example.com/doc) - Alice - 2024-10-01" in markdown
        assert "fallback.txt" in markdown
        assert markdown.count("Nice Title") == 1
