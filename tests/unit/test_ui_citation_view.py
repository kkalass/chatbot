"""Unit tests for UI citation rendering helpers."""

from src.chatbot.app.protocols import QuoteReferenceEvent, SourceChunk, SourceCitationEvent
from src.chatbot.ui.app import collect_unique_citation_chunks
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

    def test_build_citation_name_includes_page_when_present(self) -> None:
        chunk = SourceChunk(
            content="c",
            source="doc.txt",
            score=0.9,
            chunk_id="42",
            title="Nice Title",
            page="7",
        )

        assert build_citation_name(chunk) == "Nice Title (p. 7)"

    def test_build_citation_content_includes_optional_metadata_when_present(self) -> None:
        chunk = SourceChunk(
            content="Chunk body",
            source="doc.txt",
            score=0.9,
            chunk_id="42",
            author="Alice",
            publication_date="2024-10-01",
            source_url="https://example.com/doc",
            page="4",
        )

        content = build_citation_content(chunk)

        assert "### doc.txt" not in content
        assert "### Nice Title" not in content
        assert "### [Alice - 2024-10-01](https://example.com/doc)" in content
        assert "**Author:** Alice" in content
        assert "**Date:** 2024-10-01" in content
        assert "**Page:** 4" in content
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
        assert "**Page:**" not in content
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
            page="2",
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
            page="2",
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
        assert "[Nice Title](https://example.com/doc) - Alice - 2024-10-01 - p. 2" in markdown
        assert "fallback.txt" in markdown
        assert markdown.count("Nice Title") == 1

    def test_build_citation_markdown_keeps_same_source_for_different_pages(self) -> None:
        page_1 = SourceChunk(
            content="Chunk A",
            source="doc.pdf",
            score=0.9,
            chunk_id="42",
            title="Policy",
            page="1",
        )
        page_2 = SourceChunk(
            content="Chunk B",
            source="doc.pdf",
            score=0.8,
            chunk_id="43",
            title="Policy",
            page="2",
        )

        markdown = build_citation_markdown([page_1, page_2])

        assert markdown.count("1. ") == 1
        assert markdown.count("2. ") == 1
        assert "Policy - p. 1" in markdown
        assert "Policy - p. 2" in markdown


class TestQuoteReferenceEventToken:
    def test_reference_token_format(self) -> None:
        event = QuoteReferenceEvent(reference_number=3, canonical_key="search:id:src:cid")
        assert f"[{event.reference_number}]" == "[3]"

    def test_reference_numbers_are_sequential(self) -> None:
        tokens = [f"[{QuoteReferenceEvent(reference_number=n, canonical_key=f'k{n}').reference_number}]" for n in range(1, 5)]
        assert tokens == ["[1]", "[2]", "[3]", "[4]"]


class TestCollectUniqueCitationChunks:
    def _chunk(self, source: str, chunk_id: str, content: str = "body") -> SourceChunk:
        return SourceChunk(content=content, source=source, score=0.9, chunk_id=chunk_id)

    def test_empty_events_returns_empty(self) -> None:
        assert collect_unique_citation_chunks([]) == []

    def test_single_event_preserves_order(self) -> None:
        a = self._chunk("doc.txt", "1")
        b = self._chunk("doc.txt", "2")
        event = SourceCitationEvent(validated=(a, b))
        result = collect_unique_citation_chunks([event])
        assert result == [a, b]

    def test_deduplicates_same_source_and_chunk_id_across_events(self) -> None:
        a = self._chunk("doc.txt", "1")
        b = self._chunk("doc.txt", "2")
        # Same (source, chunk_id) pair in two separate events — second occurrence is dropped.
        event1 = SourceCitationEvent(validated=(a,))
        event2 = SourceCitationEvent(validated=(a, b))
        result = collect_unique_citation_chunks([event1, event2])
        assert result == [a, b]

    def test_preserves_first_seen_order_across_events(self) -> None:
        a = self._chunk("a.txt", "1")
        b = self._chunk("b.txt", "1")
        c = self._chunk("c.txt", "1")
        event1 = SourceCitationEvent(validated=(b, a))
        event2 = SourceCitationEvent(validated=(c,))
        result = collect_unique_citation_chunks([event1, event2])
        assert result == [b, a, c]

    def test_different_chunk_ids_same_source_are_kept_separate(self) -> None:
        a = self._chunk("doc.txt", "1")
        b = self._chunk("doc.txt", "2")
        event = SourceCitationEvent(validated=(a, b))
        result = collect_unique_citation_chunks([event])
        assert len(result) == 2
