"""Unit tests for UI citation rendering helpers."""

from src.chatbot.app.protocols import (
    QuoteReferenceEvent,
    SourceChunk,
    SourceCitationEvent,
    ToolCitationEvent,
)
from src.chatbot.ui.app import (
    collect_unique_citation_chunks,
    collect_unique_tool_citations,
    consume_quote_reference_event,
    consume_text_chunk,
)
from src.chatbot.ui.citation_view import (
    build_citation_content,
    build_citation_markdown,
    build_citation_name,
    build_tool_citation_content,
    build_tool_citation_markdown,
    build_tool_citation_name,
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

    def test_build_citation_markdown_keeps_distinct_chunks_even_with_same_metadata(self) -> None:
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
        assert markdown.count("3. ") == 1
        assert (
            markdown.count("[Nice Title](https://example.com/doc) - Alice - 2024-10-01 - p. 2") == 2
        )
        assert "fallback.txt" in markdown
        assert markdown.count("Nice Title") == 2

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

    def test_build_tool_citation_name_uses_tool_name(self) -> None:
        tool_citation = ToolCitationEvent(
            tool_call_id="v1",
            tool_name="get_vacation_days",
            result={},
        )

        assert build_tool_citation_name(tool_citation) == "Tool: get_vacation_days"

    def test_build_tool_citation_content_renders_result_properties(self) -> None:
        tool_citation = ToolCitationEvent(
            tool_call_id="v1",
            tool_name="get_vacation_days",
            result={"remaining_days": 15, "total_days": 25, "used_days": 10},
        )

        content = build_tool_citation_content(tool_citation)

        assert "get_vacation_days" in content
        assert "**remaining_days:**" in content
        assert "15" in content
        assert "**total_days:**" in content
        assert "25" in content

    def test_build_tool_citation_markdown_deduplicates_tool_entries(self) -> None:
        first = ToolCitationEvent(
            tool_call_id="v1",
            tool_name="get_vacation_days",
            result={"remaining_days": 15},
        )
        duplicate = ToolCitationEvent(
            tool_call_id="v1",
            tool_name="get_vacation_days",
            result={"remaining_days": 15},
        )

        markdown = build_tool_citation_markdown([first, duplicate])

        assert markdown.count("1. ") == 1
        assert "get_vacation_days" in markdown


class TestQuoteReferenceEventToken:
    def test_reference_token_format(self) -> None:
        event = QuoteReferenceEvent(reference_number=3, canonical_key="search:id:src:cid")
        assert f"[{event.reference_number}]" == "[3]"

    def test_reference_numbers_are_sequential(self) -> None:
        tokens = [
            f"[{QuoteReferenceEvent(reference_number=n, canonical_key=f'k{n}').reference_number}]"
            for n in range(1, 5)
        ]
        assert tokens == ["[1]", "[2]", "[3]", "[4]"]


class TestInlineReferenceRenderingOrder:
    def test_text_chunk_buffers_trailing_whitespace(self) -> None:
        tokens, pending = consume_text_chunk("Statement.\n\n", "")

        assert tokens == ["Statement."]
        assert pending == "\n\n"

    def test_text_chunk_flushes_existing_pending_whitespace_before_new_text(self) -> None:
        tokens, pending = consume_text_chunk("Next sentence.", "\n")

        assert tokens == ["\n", "Next sentence."]
        assert pending == ""

    def test_quote_reference_is_emitted_before_pending_whitespace(self) -> None:
        event = QuoteReferenceEvent(reference_number=3, canonical_key="search:id:src:cid")

        tokens, pending = consume_quote_reference_event(event, "\n\n")

        assert tokens == ["[3]", "\n\n"]
        assert pending == ""

    def test_whitespace_only_chunks_do_not_flush_before_quote_reference(self) -> None:
        tokens_1, pending_1 = consume_text_chunk("Statement.\n", "")
        tokens_2, pending_2 = consume_text_chunk("\n", pending_1)
        event = QuoteReferenceEvent(reference_number=1, canonical_key="search:id:src:cid")
        tokens_3, pending_3 = consume_quote_reference_event(event, pending_2)

        assert tokens_1 == ["Statement."]
        assert tokens_2 == []
        assert tokens_3 == ["[1]", "\n\n"]
        assert pending_3 == ""


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


class TestCollectUniqueToolCitations:
    def test_deduplicates_same_tool_citation(self) -> None:
        tool_citation = ToolCitationEvent(
            tool_call_id="v1",
            tool_name="get_vacation_days",
            result={"remaining_days": 15},
        )

        result = collect_unique_tool_citations([tool_citation, tool_citation])

        assert result == [tool_citation]
