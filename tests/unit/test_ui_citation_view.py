"""Unit tests for UI citation rendering helpers."""

from src.chatbot.app.protocols import SourceChunk
from src.chatbot.ui.citation_view import build_citation_content, build_citation_name


class TestCitationView:
    def test_build_citation_name_prefers_title(self) -> None:
        chunk = SourceChunk(
            content="c",
            source="doc.txt",
            score=0.9,
            chunk_id="42",
            title="Nice Title",
        )

        assert build_citation_name(chunk) == "Nice Title [42]"

    def test_build_citation_name_falls_back_to_source(self) -> None:
        chunk = SourceChunk(
            content="c",
            source="doc.txt",
            score=0.9,
            chunk_id="42",
        )

        assert build_citation_name(chunk) == "doc.txt [42]"

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

        assert "Author: Alice" in content
        assert "Publication date: 2024-10-01" in content
        assert "Source link: https://example.com/doc" in content
        assert "Source file: doc.txt" in content
        assert content.endswith("Chunk body")

    def test_build_citation_content_omits_missing_optional_metadata(self) -> None:
        chunk = SourceChunk(
            content="Chunk body",
            source="doc.txt",
            score=0.9,
            chunk_id="42",
        )

        content = build_citation_content(chunk)

        assert "Author:" not in content
        assert "Publication date:" not in content
        assert "Source link:" not in content
        assert "Source file: doc.txt" in content
